#==============================================================================
Type-2 execute: deconvolve → embed into oversampled grid → (i)FFT →
interpolate.

Mirror of execute_type1, all regular Julia traced by Reactant. The
interpolation gather lives in spread_interp.jl.
==============================================================================#

# ---------- corner embed -----------------------------------------------------
#
# Exact transpose of `_central_view`: place the centered-mode layout
# `[neg_half; non_neg_half]` into the 2^D corners of a zeroed oversampled
# grid (non-negative modes at the front, negative modes at the back of each
# spatial axis). Static slices + setindex!, lowered to `stablehlo.slice` +
# `stablehlo.dynamic_update_slice`.
function _corner_embed(fk, nmodes::NTuple{D, Int}, ngrid::NTuple{D, Int}) where {D}
    if all(nmodes .== ngrid)
        return fk
    end
    @assert ndims(fk) == D + 1
    out = similar(fk, (ngrid..., size(fk, D + 1)))
    fill!(out, zero(eltype(fk)))
    halves = nmodes .÷ 2
    n_poss = nmodes .- halves
    for sign_bits in 0:(1 << D - 1)
        is_neg = ntuple(d -> ((sign_bits >> (d - 1)) & 1) == 1, Val(D))
        any(d -> is_neg[d] && halves[d] == 0, 1:D) && continue
        src = ntuple(
            d -> is_neg[d] ?
                (1:halves[d]) :
                ((halves[d] + 1):nmodes[d]), Val(D)
        )
        dst = ntuple(
            d -> is_neg[d] ?
                ((ngrid[d] - halves[d] + 1):ngrid[d]) :
                (1:n_poss[d]), Val(D)
        )
        out[dst..., :] = fk[src..., :]
    end
    return out
end

# ---------- main entry point ------------------------------------------------

"""
    execute_type2(prep, fk) -> c

Type-2 NUFFT: uniform → nonuniform.
- `fk::AbstractArray{<:Complex}` of shape `nmodes...` or `(nmodes..., ntrans)`.
- Returns `c` of shape `(M,)` or `(M, ntrans)` matching the input rank.

Designed to be called inside `Reactant.@jit`.
"""
function execute_type2(prep::NUFFTSetPts{T, D}, fk::AbstractArray) where {T, D}
    plan = prep.plan
    @assert nufft_type(plan) == 2 "Plan was not built for type-2"
    @assert size(fk)[1:D] == plan.nmodes "fk shape mismatch with plan.nmodes"

    squeeze_out = ndims(fk) == D
    fk_full = squeeze_out ? reshape(fk, plan.nmodes..., 1) : fk
    ntrans = size(fk_full, D + 1)

    return _execute_type2_impl(prep, fk_full, ntrans, squeeze_out)
end

function _execute_type2_impl(
        prep::NUFFTSetPts{T, D}, fk_full::AbstractArray, ntrans::Int, squeeze_out::Bool
    ) where {T, D}
    plan = prep.plan

    # 1. Deconvolve by the separable phi_hat product.
    fk_dec = fk_full ./ _phi_hat_tensor(plan)

    # 2. Embed central modes into the corners of the oversampled grid.
    fw_hat = _corner_embed(fk_dec, plan.nmodes, plan.ngrid)

    # 3. (Inverse-)FFT (same sign convention as type-1).
    fw = plan.iflag < 0 ?
        AbstractFFTs.fft(fw_hat, 1:D) :
        AbstractFFTs.bfft(fw_hat, 1:D)

    # 4. Gather each point's stencil and contract against the kernel weights.
    c = _interp(prep, fw, ntrans)

    return squeeze_out ? dropdims(c; dims = 2) : c
end
