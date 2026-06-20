# Reactant NUFFT — custom global-memory type-2 interp kernel

**Date:** 2026-06-19
**Branch:** `ptiede-nffttest`
**Status:** Design approved, pending spec review

## Goal

Make the Reactant NUFFT type-2 path (image → visibilities) competitive with
cuFINUFFT at large visibility counts (M ≈ 10⁶), where it currently loses ~2.5–3×.

This is a **forward-only prototype** whose primary purpose is to answer "can a
custom kernel embedded in Reactant close the M=10⁶ gap?" The existing
differentiable StableHLO path stays the default and is untouched; the kernel is
opt-in. Gradients are **not** implemented in this milestone, but the design is
shaped so the reverse-mode adjoint drops in cleanly later (see §6).

## Background / why

Prior benchmarking (`bench/SUMMARY.md`, GPU RTX 5090, eps=1e-9, 2D) established:

- Reactant **beats** cuFINUFFT 2–5.7× at M ≤ 10⁵ (no kernel-launch floor).
- cuFINUFFT **wins ~2.5–3×** at M = 10⁶.
- The gap is isolated to the type-2 interpolation: it lowers to
  `gather → dot_general` over a materialized `(M, w²)` buffer, costing
  `a≈0.0166` per stencil cell vs cuFINUFFT's `a≈0.00698` (~2.4×/cell). It is
  shared by both the old and refactored Reactant impls — not a regression.
- An "accumulate over stencil cells" reformulation (still pure HLO) was tried
  and reverted: it regressed exec ~2× at M=10⁶. SUMMARY conclusion: "closing
  the residual ~1.6× would need a custom kernel."

**Feasibility unlock (verified this session):** Reactant 0.2.266 (dev checkout
at `/mnt/ptiede/.julia/dev/Reactant`) ships `ReactantKernelAbstractionsExt` and
`ReactantCUDAExt`. A `KernelAbstractions.@kernel` launched on
`KernelAbstractions.get_backend(::Reactant array)` traces into the `@jit`
program and, on CUDA, uses the native kernel path (`raise=false`). Confirmed by
`Reactant/test/integration/kernelabstractions.jl` (matmul/square kernels with
loops, `@index(Global)`, register accumulation, `@inbounds`, `@Const`). This is
exactly the shape a per-point interp kernel needs.

## Approach (chosen)

**A — Custom KA type-2 interp kernel, global-memory (GM) method.** One thread
per `(point m, transform t)`. Each thread evaluates its `w^D` Horner stencil
weights, loops the `w^D` oversampled-grid cells reading `fw` from global memory,
accumulates one `Complex{T}` in registers, and writes one output element. This
removes the materialized `(M, w²)` gather and the `dot_general` — the exact cost
SUMMARY isolated. This mirrors cuFINUFFT's own GM method.

Approaches considered and deferred/rejected:

- **B — Shared-memory subgrid-tiling kernel** (cuFINUFFT's SM method): best
  perf, but needs cooperative shared-memory tiling + per-tile binning. Deferred
  as the stretch goal once A proves the thesis.
- **C — `julia_callback` to NonuniformFFTs/cuFINUFFT GPU**
  (proven in `Reactant/test/integration/nfft_callback.jl`): "use someone else's
  kernel," adds host-callback overhead, and `NonuniformFFTsAlg` already exists.
  Kept only as a comparison point, not the answer.

## Components & data flow

Everything upstream of the gather is reused unchanged: `setpts` metadata
(`prep.base[d]`, `prep.frac[d]` in original point order), `plan.horner_coefs`
`(w, deg+1)`, `plan.ngrid`, `plan.nspread = w`, the deconvolution, corner embed,
and FFT in `execute_type2.jl`. Only the final `_interp(prep, fw, ntrans)` call
branches.

### 1. New file `ext/VLBISkyModelsReactantExt/nufft/interp_kernel.jl`

- `@kernel function _interp_gm_kernel!(out, @Const(fw_vec), base/frac per dim,
  @Const(coefs), ngrid, w, nflat, ntrans)` — `ndrange = (M, ntrans)`.
  - Parameterized on `Val(w)` and `Val(D)` so per-dim weight vectors are static
    `MVector{w}` and the `w^D` cell loop is statically nested.
  - Per thread: compute `t = 2*frac[d] - 1`, Horner-evaluate weights into
    `MVector{w}` per dim; loop nested `w^D` cells computing wrapped linear index
    `lin = 1 + Σ_d ngrid_prefix_d * mod(base[d] + k_d, ngrid[d])`; accumulate
    `acc += fw_vec[lin + (t-1)*nflat] * Π_d weight_d[k_d]`; `out[m,t] = acc`.
- Shared device-`@inline` helpers factored for reuse by the future adjoint:
  - `_horner_eval(coefs, frac) -> weight` (scalar, one cell),
  - `_wrap(base_d, k, ngrid_d)`,
  - per-cell weight/linear-index combiners.
- `horner_coefs` is moved onto the device once (e.g. cached `to_rarray`) and
  passed as `@Const`.

### 2. `_interp` branch (`spread_interp.jl`)

```
function _interp(prep, fw, ntrans)
    if prep.plan.interp_method === :ka
        out = similar(vec(fw), complex(T), (M, ntrans))
        launch _interp_gm_kernel! over (M, ntrans)
        return out
    else
        # existing chunked HLO gather (default, differentiable) — unchanged
    end
end
```

The KA path needs no chunking, no padding, and no `Reactant.@trace for` — the
kernel covers all M points directly. Points are read in **original order**
(like the current HLO interp), so output rows align with input rows; no
permutation gather. Bin-sort for memory coalescing is a later optimization, not
a correctness requirement.

### 3. Plan / API threading

- Add `interp_method::Symbol` to `VLBISkyModels.ReactantNUFFTAlg`
  (`src/fourierdomain/nuft/nfft_reactant.jl`), default `:hlo`. Keyword on the
  constructor; documented values `:hlo` (default, differentiable) and `:ka`
  (forward-only kernel).
- Add `interp_method::Symbol` to `NUFFTPlan`
  (`ext/.../nufft/plan.jl`), populated from `opts.interp_method` in
  `plan_nufft` (also overridable via the existing `kwargs` pass-through).

Decision: extend the existing alg/plan with a field rather than introducing a
separate algorithm type (keeps the public surface and dispatch unchanged).

## Error handling

- **AD guard:** the `:ka` path has no adjoint yet. Hitting it under reverse-mode
  AD must **error clearly** (message naming the missing adjoint and pointing to
  this design), never silently return zero/garbage gradients. Implement via the
  existing custom-rule machinery (the package already special-cases `_nuft` for
  ForwardDiff via `_frule_nuft` and marks plan accessors
  `EnzymeRules.inactive`): add a reverse rule on the `:ka` `_nuft`/`_interp`
  that throws until the adjoint kernel exists.
- **Backend guard:** if `:ka` is requested but the kernel cannot raise/lower on
  the active backend, surface a clear error rather than a deep KA/LLVM failure.
- **Validation guards:** keep existing `@assert`s on shapes/types.

## Testing & validation

1. **Standalone micro-benchmark first** (proves the thesis in isolation before
   any wiring): add a `:ka` variant to `bench/bench_kernel_reactant.jl` — bare
   type-2 on identical inputs vs cuFINUFFT (same N×N image, same M points/seed,
   eps=1e-9, iflag=+1). Correctness via the existing `xcheck_*.jl` (target: max
   rel err vs HLO/cuFINUFFT ≲ 1e-13 on identical inputs), then on-device exec
   timing (min over 11 runs, scalar-reduction sync) for
   N ∈ {64,256,1024} × M ∈ {1e5, 1e6}.
2. **Parity:** `:ka` vs the HLO `_interp` and vs the NFFT.jl reference
   (`bench/verify_reactant.jl`, `test/reactant.jl`) to the same tolerances
   already recorded (4.4e-10 … 1.5e-9).
3. **Workflow:** rerun the SUMMARY M ∈ {1e5,1e6} × N ∈ {64,256,1024} exec table
   through the public `FourierDualDomain`/`visibilitymap` path with `:ka`.

Success criterion for the prototype: a clear, reproducible improvement in
on-device exec at M=10⁶ vs the `:hlo` path, ideally landing within ~1.5× of
cuFINUFFT (full closure is the SM-method stretch goal).

## Gradient-later constraints (do not implement now)

The reverse-mode adjoint of type-2 interp (gather) is type-1 spread
(scatter-add). To keep that path cheap to add later:

- Factor the device stencil math (Horner eval, index wrap, per-cell weight) into
  shared `@inline` helpers so a future `_spread_atomic_kernel!` (atomic
  scatter-add via `Atomix.@atomic`) reuses them verbatim.
- Plan to attach a **custom Enzyme/ChainRules reverse rule** on the `:ka`
  `_nuft` that invokes the adjoint kernel — matching the existing `_frule_nuft`
  pattern — rather than relying on Enzyme differentiating through the KA kernel.
- The AD guard error (above) is the placeholder for this rule.

## Risks

- KA kernel raising/lowering on CUDA with `Complex{T}` arithmetic + nested
  Horner loops. Mitigate: start from the proven matmul/square pattern, smallest
  case first, inspect `@code_hlo`.
- `Val(w)`/`Val(D)` specialization correctness and getting `horner_coefs` onto
  the device cleanly inside the trace.
- Run-to-run exec variance at M=10⁶ already noted for the HLO path; measure with
  the same on-device min protocol to stay comparable.

## Out of scope

- Type-1 standalone fast path, SM-method tiling, gradients/adjoint kernel, and
  non-CUDA backends for the `:ka` path. All are follow-ups.
