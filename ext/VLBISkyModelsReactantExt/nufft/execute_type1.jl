#==============================================================================
Type-1 execute: spread → FFT → crop central modes → deconvolve.

All steps are regular Julia (broadcasts, slicing, AbstractFFTs) traced by
Reactant. The spread's scatter-add lives in spread_interp.jl and is currently
the performance blocker — see the warning on `_scatter_add!`.
==============================================================================#

# ---------- central-mode crop (with periodic wrap) --------------------------
#
# Read the 2^D corners of fw_hat (non-negative modes from the front, negative
# modes from the back of the oversampled grid) into the centered-mode layout
# `[neg_half; non_neg_half]` along each dim. Static slices + setindex!, which
# Reactant lowers to `stablehlo.slice` + `stablehlo.dynamic_update_slice`.
function _central_view(fw_hat, nmodes::NTuple{D, Int}, ngrid::NTuple{D, Int}) where {D}
    if all(nmodes .== ngrid)
        return fw_hat
    end
    @assert ndims(fw_hat) == D + 1
    out = similar(fw_hat, (nmodes..., size(fw_hat, D + 1)))
    fill!(out, zero(eltype(fw_hat)))
    halves = nmodes .÷ 2
    n_poss = nmodes .- halves
    for sign_bits in 0:(1 << D - 1)
        is_neg = ntuple(d -> ((sign_bits >> (d - 1)) & 1) == 1, Val(D))
        any(d -> is_neg[d] && halves[d] == 0, 1:D) && continue
        src = ntuple(
            d -> is_neg[d] ?
                ((ngrid[d] - halves[d] + 1):ngrid[d]) :
                (1:n_poss[d]), Val(D)
        )
        dst = ntuple(
            d -> is_neg[d] ?
                (1:halves[d]) :
                ((halves[d] + 1):nmodes[d]), Val(D)
        )
        out[dst..., :] = fw_hat[src..., :]
    end
    return out
end

# ---------- phi_hat outer-product for deconvolution -------------------------
#
# (nmodes_1, ..., nmodes_D, 1) host tensor of per-dim phi_hat products; the
# trailing singleton is the ntrans axis. Enters the trace as a constant.
function _phi_hat_tensor(plan::NUFFTPlan{T, D}) where {T, D}
    factors = ntuple(D) do d
        shape = ntuple(i -> i == d ? plan.nmodes[d] : 1, Val(D + 1))
        return reshape(plan.phi_hat[d], shape...)
    end
    return reduce(.*, factors)
end

# ---------- main entry point ------------------------------------------------

"""
    execute_type1(prep, c) -> fk

Type-1 NUFFT: nonuniform → uniform.
- `c::AbstractArray{<:Complex}` of shape `(M,)` or `(M, ntrans)`.
- Returns `fk` of shape `nmodes...` (when input was `(M,)`) or
  `(nmodes..., ntrans)` (when input was `(M, ntrans)`).

Designed to be called inside `Reactant.@jit`.
"""
function execute_type1(prep::NUFFTSetPts{T, D}, c::AbstractArray) where {T, D}
    plan = prep.plan
    @assert nufft_type(plan) == 1 "Plan was not built for type-1"
    @assert size(c, 1) == prep.M "Strength count mismatch with prepared points"

    squeeze_out = ndims(c) == 1
    cmat = squeeze_out ? reshape(c, prep.M, 1) : c
    ntrans = size(cmat, 2)

    return _execute_type1_impl(prep, cmat, ntrans, squeeze_out)
end

function _execute_type1_impl(
        prep::NUFFTSetPts{T, D}, cmat::AbstractMatrix, ntrans::Int, squeeze_out::Bool
    ) where {T, D}
    plan = prep.plan

    # 1. Spread bin-sorted points onto the oversampled grid.
    fw = _spread(prep, cmat, ntrans)

    # 2. FFT (sign per iflag).
    fw_hat = plan.iflag < 0 ?
        AbstractFFTs.fft(fw, 1:D) :
        AbstractFFTs.bfft(fw, 1:D)

    # 3. Crop central modes (with periodic wrap).
    fw_central = _central_view(fw_hat, plan.nmodes, plan.ngrid)

    # 4. Deconvolve by the separable phi_hat product.
    fk = fw_central ./ _phi_hat_tensor(plan)

    return squeeze_out ? dropdims(fk; dims = D + 1) : fk
end
