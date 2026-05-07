#==============================================================================
Type-2 execute: zero-pad+deconvolve → iFFT → interpolate.

Mirror of execute_type1. The only non-Base call is `@opcall gather` for the
batched stencil read; everything else is plain Julia broadcasting / slicing.
==============================================================================#

# ---------- corner embed + post-FFT shift -----------------------------------
#
# fk_deconv has the centered-mode layout produced by `_central_view`:
# along each spatial dim, the first `half = nmodes[d] ÷ 2` entries correspond
# to negative modes and the remaining `n_pos = nmodes[d] - half` to
# non-negative modes.
#
# The *centered* embed places those into the (ngrid_d,) array as
#   F_n = [non-neg modes; zeros; neg modes]
# i.e. F_n = circshift(corner_pad(fk_deconv), -half) along each spatial dim.
# Naively this requires 2^D pairs of dynamic_slice / dynamic_update_slice.
#
# We use the FFT shift theorem to avoid the corner shuffle. Computing
#   G = corner_pad(fk_deconv)        (zeros at the high end of each spatial axis)
#   FW = fft_or_bfft(G, 1:D)
# gives the same result as `fft_or_bfft(F_n, 1:D)` *up to* a per-axis
# linear-phase factor on the spatial-domain output. We apply that phase
# post-FFT in `_post_fft_shift_t2` — the math is in the docstring there.

function _corner_embed(fk_deconv, nmodes::NTuple{D,Int}, ngrid::NTuple{D,Int}) where {D}
    if all(nmodes .== ngrid)
        return fk_deconv
    end
    CT = Reactant.unwrapped_eltype(eltype(fk_deconv))
    R = D + 1
    fk_t = Reactant.promote_to(Reactant.TracedRArray{CT,R}, fk_deconv)
    high = Int64[ntuple(d -> ngrid[d] - nmodes[d], Val(D))..., 0]
    zero_val = Reactant.promote_to(Reactant.TracedRNumber{CT}, zero(CT))
    return Reactant.Ops.@opcall pad(fk_t, zero_val; high=high)
end

# Multiply `fw` (post-FFT spatial output, shape (ngrid..., ntrans)) by the
# separable phase factor that converts an FFT-of-corner-padded result into
# the FFT-of-centered-embedded result.
#
# Derivation. With F_n = circshift(G, -half_d) along axis d, the shift
# theorem gives
#   fft (iflag=-1):  fft(F_n)[n]  = fft(G)[n]  * exp(+2πi * half_d * n_d / ngrid_d)
#   bfft (iflag=+1): bfft(F_n)[n] = bfft(G)[n] * exp(-2πi * half_d * n_d / ngrid_d)
# i.e. phase[n_d] = exp(-iflag * 2πi * half_d * n_d / ngrid_d). The full
# multi-D factor is the outer product of per-axis phases — applied as D
# separable broadcasts so the (ngrid..., 1) outer product is never
# materialized as a constant.
#
# Skip when nmodes == ngrid (no embed → no shift, matching the old contract).
function _post_fft_shift_t2(fw, plan::NUFFTPlan{T,D}) where {T,D}
    if all(plan.nmodes .== plan.ngrid)
        return fw
    end
    R = D + 1
    for d in 1:D
        half_d = plan.nmodes[d] ÷ 2
        half_d == 0 && continue
        ngrid_d = plan.ngrid[d]
        ang = (-T(plan.iflag) * 2 * T(pi) * T(half_d) / T(ngrid_d)) .* collect(T, 0:(ngrid_d - 1))
        shape = ntuple(i -> i == d ? ngrid_d : 1, R)
        v = reshape(cis.(ang), shape...)
        fw = fw .* v
    end
    return fw
end

# ---------- batched gather for stencil reads --------------------------------

function _gather_spatial(
    fw::Reactant.AnyTracedRArray{CT,Dp1},
    gather_idx::AbstractMatrix,
    ::Val{D},
    ntrans::Int,
) where {CT,Dp1,D}
    @assert Dp1 == D + 1
    res = Reactant.Ops.@opcall gather(
        fw,
        Reactant.promote_to(Reactant.TracedRArray{Int,2}, gather_idx);
        offset_dims=Int64[2],                                # ntrans dim of result
        collapsed_slice_dims=collect(Int64, 1:D),            # spatial dims of fw
        operand_batching_dims=Int64[],
        start_indices_batching_dims=Int64[],
        start_index_map=collect(Int64, 1:D),
        index_vector_dim=Int64(2),
        slice_sizes=Int64[ntuple(_ -> 1, D)..., ntrans],
    )
    return res                                                # (Nupd, ntrans)
end

# ---------- main entry point ------------------------------------------------

"""
    execute_type2(prep, fk) -> c

Type-2 NUFFT: uniform → nonuniform.
- `fk::AbstractArray{<:Complex}` of shape `nmodes...` or `(nmodes..., ntrans)`.
- Returns `c` of shape `(M,)` or `(M, ntrans)` matching the input rank.

Designed to be called inside `Reactant.@jit`.
"""
function execute_type2(prep::NUFFTSetPts{T,D}, fk::AbstractArray) where {T,D}
    plan = prep.plan
    @assert nufft_type(plan) == 2 "Plan was not built for type-2"
    @assert size(fk)[1:D] == plan.nmodes "fk shape mismatch with plan.nmodes"

    squeeze_out = ndims(fk) == D
    fk_full = squeeze_out ? reshape(fk, plan.nmodes..., 1) : fk
    ntrans = size(fk_full, D + 1)

    return _execute_type2_impl(prep, fk_full, ntrans, squeeze_out)
end

function _execute_type2_impl(
    prep::NUFFTSetPts{T,D}, fk_full::AbstractArray, ntrans::Int, squeeze_out::Bool
) where {T,D}
    plan = prep.plan
    w = plan.nspread
    ngrid = plan.ngrid
    M_pad = prep.M_pad

    # 1. Deconvolve by separable phi_hat product.
    phih = _phi_hat_tensor(plan)
    fk_dec = fk_full ./ phih

    # 2. Embed into oversampled grid (corner pad — single XLA op).
    fw_hat = _corner_embed(fk_dec, plan.nmodes, ngrid)

    # 3. (Inverse-)FFT (same sign convention as type-1).
    fw = plan.iflag < 0 ?
         AbstractFFTs.fft(fw_hat, 1:D) :
         AbstractFFTs.bfft(fw_hat, 1:D)

    # 3a. Apply FFT-shift phase to recover centered-mode semantics
    #     (compensates for corner_embed not doing the circshift).
    fw = _post_fft_shift_t2(fw, plan)

    # 4. Gather + weighted sum.
    coefs = Reactant.promote_to(Reactant.TracedRArray{T,2}, plan.horner_coefs)
    offsets_row = reshape(collect(0:(w - 1)), 1, w)
    wpd = ntuple(d -> _per_dim_weights(coefs, prep.frac_sorted[d]), Val(D))
    idx_per_dim = ntuple(d -> _per_dim_indices(prep.base_sorted[d], ngrid[d], offsets_row), Val(D))
    c_sorted = _gather_sum_traced(fw, idx_per_dim, wpd, M_pad, w, ntrans, Val(D))

    # 5. Inverse permutation, returning length M (un-padded).
    c = c_sorted[prep.invperm, :]
    return squeeze_out ? dropdims(c; dims=2) : c
end

# Single per-point @trace-for body: for each j (and each transform t),
# accumulate sum_{k1..kD} (prod_d wpd[d][j,k_d]) * fw[i_1, ..., i_D, t].
#
# - The outer `j` loop is `Reactant.@trace for ... track_numbers=false`,
#   lowered to one MLIR while loop instead of M_pad unrolls.
#   `track_numbers=false` is required for nested-loop state propagation —
#   without it the macro promotes scalar accumulators into traced numbers
#   that lose updates across iterations.
# - The inner `k_d` loops are plain Julia `for` loops over `1:w` and so
#   unroll at trace time. Nested `@trace for` inside the j body would not
#   propagate `s` correctly today, and statically unrolling `w^D` per-cell
#   reads is what we want anyway: it lets XLA fuse the per-j stencil into
#   a single per-thread sweep.
# - The `t` (ntrans) loop is plain Julia and lives *inside* the @trace for
#   body, so all writes to `out` happen within a single while loop. Putting
#   the `t` loop outside the @trace for would split writes across separate
#   while invocations, and updates from earlier invocations don't propagate
#   into the next (the @trace for macro doesn't write back captured
#   external arrays into the outer-scope binding between invocations).
# - Reads from `fw` use linear indexing into a flattened view (`fw_flat`)
#   with a static per-t byte offset baked in. Scalar `fw[i_1, ..., i_D, t]`
#   with traced spatial indices and a Julia-Int trailing `t` is currently
#   miscompiled in Reactant: the trailing const dim is ignored and every t
#   reads the t=1 plane. Linear indexing dodges the bug and is what XLA
#   would compute anyway.

function _gather_sum_traced(
    fw, idx_per_dim::NTuple{1}, wpd::NTuple{1},
    M_pad::Int, w::Int, ntrans::Int, ::Val{1},
)
    # D=1 stays on the @trace for per-point body: fw is small enough to live
    # in L2 (e.g. 4096 × 8 B = 32 KB at the target row), so M·w scalar reads
    # are effectively cache-free — and a wide gather here was 8× slower
    # (0.12 → 0.98 ms at M=1e6 N=4096) because it materialized a 56 MB
    # (M, w, ntrans) intermediate that's DRAM-bound to write/read.
    out = similar(fw, eltype(fw), (M_pad, ntrans))
    fill!(out, zero(eltype(fw)))
    idx1, w1 = idx_per_dim[1], wpd[1]
    N1 = size(fw, 1)
    fw_flat = reshape(fw, N1 * ntrans)
    Reactant.@trace track_numbers = false for j in 1:M_pad
        for t in 1:ntrans
            t_off = (t - 1) * N1
            s = zero(eltype(fw))
            for k in 1:w
                wk = @allowscalar w1[j, k]
                i  = @allowscalar idx1[j, k]
                v  = @allowscalar fw_flat[i + t_off]
                s  = s + wk * v
            end
            @allowscalar out[j, t] = s
        end
    end
    return out
end

function _gather_sum_traced(
    fw, idx_per_dim::NTuple{2}, wpd::NTuple{2},
    M_pad::Int, w::Int, ntrans::Int, ::Val{2},
)
    # Wide-stencil gather: read each point's full (w, w) stencil in one
    # batched gather (slice_size = [w, w, ntrans]), then broadcast-multiply
    # by the per-point outer-product weights and reduce. Beats a per-cell
    # @trace for body 2.45× isolated (D=2 M=1e6 N=256²) — coalesced
    # contiguous reads instead of M·w² scattered single-element reads.
    #
    # Requires fw to be circular-padded by (w-1) on each spatial axis so that
    # `start = base[j]+1` is always a valid in-bounds slice corner even when
    # the stencil straddles the wrap boundary. We recover the raw `base`
    # from `idx_per_dim[d][:, 1] - 1` (the leftmost cell of the stencil,
    # already wrapped 1-based).
    idx1, idx2 = idx_per_dim
    w1, w2     = wpd
    N1, N2 = size(fw, 1), size(fw, 2)

    fw_padded = _circular_pad_spatial(fw, w, Val(2))            # (N1+w-1, N2+w-1, ntrans)
    base1_p1  = vec(idx1[:, 1:1])                               # (M,) 1-based start dim 1
    base2_p1  = vec(idx2[:, 1:1])                               # (M,) 1-based start dim 2
    one_int   = ones(Int, M_pad)
    start_idx = hcat(reshape(base1_p1, M_pad, 1),
                     reshape(base2_p1, M_pad, 1),
                     reshape(one_int,  M_pad, 1))               # (M, 3)

    vals = Reactant.Ops.@opcall gather(
        fw_padded,
        Reactant.promote_to(Reactant.TracedRArray{Int,2}, start_idx);
        offset_dims                  = Int64[2, 3, 4],
        collapsed_slice_dims         = Int64[],
        operand_batching_dims        = Int64[],
        start_indices_batching_dims  = Int64[],
        start_index_map              = Int64[1, 2, 3],
        index_vector_dim             = Int64(2),
        slice_sizes                  = Int64[w, w, ntrans],
    )                                                            # (M, w, w, ntrans)

    # Reduce by explicit trace-time unrolled mul-add over the w² stencil
    # cells. `sum(...; dims=)` would let XLA fold this into a dot_general
    # that segfaults the enzyme DotGeneralSimplify pass for this shape.
    s = fill(zero(eltype(fw)), (M_pad, ntrans))
    for k1 in 1:w
        wk1 = reshape(vec(w1[:, k1:k1]), M_pad, 1)                       # (M, 1)
        for k2 in 1:w
            wk2  = reshape(vec(w2[:, k2:k2]), M_pad, 1)                  # (M, 1)
            cell = reshape(vals[:, k1, k2, :], M_pad, ntrans)            # (M, ntrans)
            s = s .+ (wk1 .* wk2) .* cell
        end
    end
    return s
end

# Circular-pad fw on each spatial axis by (w-1) cells so a single batched
# gather with slice_size [w, ..., w, ntrans] starting at base+1 covers any
# stencil — including those that wrap around. Padding cells alias to the
# beginning of each axis. Trailing ntrans axis is not padded.
@inline function _circular_pad_spatial(fw, w::Int, ::Val{1})
    return vcat(fw, fw[1:(w - 1), :])
end
@inline function _circular_pad_spatial(fw, w::Int, ::Val{2})
    a = cat(fw, fw[:, 1:(w - 1), :];  dims=2)
    return cat(a, a[1:(w - 1), :, :]; dims=1)
end
@inline function _circular_pad_spatial(fw, w::Int, ::Val{3})
    a = cat(fw, fw[:, :, 1:(w - 1), :]; dims=3)
    b = cat(a,  a[:, 1:(w - 1), :, :];  dims=2)
    return cat(b, b[1:(w - 1), :, :, :]; dims=1)
end

function _gather_sum_traced(
    fw, idx_per_dim::NTuple{3}, wpd::NTuple{3},
    M_pad::Int, w::Int, ntrans::Int, ::Val{3},
)
    # The wide-gather pattern that wins for D=2 (w²=49) does not transfer:
    # at D=3 w=7 means w³=343 cells per point, the gather materializes a
    # M·w³·ntrans intermediate (~2.7 GB at M=10⁶, ComplexF32) that's
    # DRAM-bound, and the trace-time unroll of 343 broadcast mul-adds
    # blows the register file (ptxas reports ~500 B spill stores/loads).
    # Empirically 20–30% slower than the scalar @trace-for path below at
    # M ∈ {10⁵, 10⁶}.
    out = similar(fw, eltype(fw), (M_pad, ntrans))
    fill!(out, zero(eltype(fw)))
    idx1, idx2, idx3 = idx_per_dim
    w1, w2, w3       = wpd
    N1, N2, N3 = size(fw, 1), size(fw, 2), size(fw, 3)
    fw_flat = reshape(fw, N1 * N2 * N3 * ntrans)
    Reactant.@trace track_numbers = false for j in 1:M_pad
        for t in 1:ntrans
            t_off = (t - 1) * N1 * N2 * N3
            s = zero(eltype(fw))
            for k1 in 1:w
                wk1 = @allowscalar w1[j, k1]
                i1  = @allowscalar idx1[j, k1]
                for k2 in 1:w
                    wk2 = @allowscalar w2[j, k2]
                    i2  = @allowscalar idx2[j, k2]
                    for k3 in 1:w
                        wk3 = @allowscalar w3[j, k3]
                        i3  = @allowscalar idx3[j, k3]
                        v   = @allowscalar fw_flat[i1 + (i2 - 1) * N1 + (i3 - 1) * N1 * N2 + t_off]
                        s   = s + (wk1 * wk2 * wk3) * v
                    end
                end
            end
            @allowscalar out[j, t] = s
        end
    end
    return out
end
