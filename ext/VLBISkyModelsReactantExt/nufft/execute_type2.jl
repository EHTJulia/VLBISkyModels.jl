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
    @assert ndims(fk) > D
    out = similar(fk, (ngrid..., size(fk)[(D + 1):end]...))
    rest = ntuple(_ -> Colon(), ndims(fk) - D)
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
        out[dst..., rest...] = fk[src..., rest...]
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
    fw = _type2_grid(prep.plan, fk_full)
    # Gather each point's stencil and contract against the kernel weights.
    c = _interp(prep, fw, ntrans)
    return squeeze_out ? dropdims(c; dims = 2) : c
end

# The oversampled grids of `fk_full :: (nmodes..., ntrans)`: deconvolve by the separable
# phi_hat product, embed the central modes into the grid corners, and (inverse-)FFT with the
# type-1 sign convention.
function _type2_grid(plan::NUFFTPlan{T, D}, fk_full::AbstractArray) where {T, D}
    fk_dec = fk_full ./ _phi_hat_tensor(plan)
    fw_hat = _corner_embed(fk_dec, plan.nmodes, plan.ngrid)
    return plan.iflag < 0 ?
        AbstractFFTs.fft(fw_hat, 1:D) :
        AbstractFFTs.bfft(fw_hat, 1:D)
end

"""
    execute_type2(prep, fk, plane, batched = ()) -> c

Type-2 NUFFT of a stack of planes `fk :: (nmodes..., planedims...)` in which each point
reads only its own plane. Without `batched`, `plane :: (M, K)` holds point `j`'s index
along each of the `K` plane dims. Otherwise the points are laid out as `size(plane)[1:end-1]`
(in the order of `prep`'s points), its dim `i + 1` is plane dim `batched[i]` of `fk`, and
`plane[..., :]` holds the index along each plane dim, `1` along those in `batched`. Returns
`c :: (M,)`.
Every plane is deconvolved and FFTed in one batch; the interpolation gathers each point's
stencil from its own plane.
"""
function execute_type2(prep::NUFFTSetPts{T, D}, fk::AbstractArray, plane::AbstractArray, batched::Tuple = ()) where {T, D}
    @assert nufft_type(prep.plan) == 2 "Plan was not built for type-2"
    @assert ndims(fk) == D + size(plane, ndims(plane)) && size(fk)[1:D] == prep.plan.nmodes "fk must be (nmodes..., planedims...)"
    @assert ndims(plane) == length(batched) + 2 && prod(size(plane)[1:(end - 1)]) == prep.M "need one plane index per point"
    return _interp_plane(prep, _type2_grid(prep.plan, fk), plane, batched)
end
