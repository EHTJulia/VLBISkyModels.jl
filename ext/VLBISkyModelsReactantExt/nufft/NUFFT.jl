#==============================================================================
ReactantNUFFT — non-uniform FFT (types 1 and 2) implemented as Reactant traces.

Architecture mirrors FINUFFT / cuFINUFFT:
  plan_nufft        — host-only, picks sigma/w, oversampled grid, Horner coefs,
                      phi_hat tables. Cheap, runs once.
  set_nufft_points  — bin-sort points, compute per-dim base/frac. One traced
                      Reactant function compiled per (M, T, D) signature.
  execute_nufft     — type-1: spread + FFT + crop+deconvolve;
                      type-2: deconvolve+embed + iFFT + interpolate.
                      One traced Reactant function compiled per
                      (M, ntrans, T, D, type, iflag, ngrid) signature.

Everything is regular Julia array code traced through Reactant — no
`Reactant.Ops` / `@opcall` usage. See the `_scatter_add!` warning in
spread_interp.jl for the one known performance blocker (type-1 spread).

Public API: plan_nufft, set_nufft_points, execute_nufft, nufft_type1,
nufft_type2, VLBISkyModels.ReactantNUFFTAlg, NUFFTPlan, NUFFTSetPts, plus direct_type{1,2}
brute-force references for tests.
==============================================================================#

include("kernel.jl")
include("options.jl")
include("plan.jl")
include("setpts.jl")
include("spread_interp.jl")
include("execute_type1.jl")
include("execute_type2.jl")
include("api.jl")
