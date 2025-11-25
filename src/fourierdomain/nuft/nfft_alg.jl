export NFFTAlg

"""
    NFFTAlg
Uses a non-uniform FFT to compute the visibilitymap.
You can optionally pass uv which are the uv positions you will
compute the NFFT at. This can allow for the NFFT plan to be cached improving
performance

# Fields
$(FIELDS)

"""
Base.@kwdef struct NFFTAlg{T, N, F} <: NUFT
    """
    Amount to pad the image
    """
    padfac::Int = 1
    """
    relative tolerance of the NFFT
    """
    reltol::T = 1.0e-9
    """
    NFFT interpolation algorithm.
    """
    precompute::N = NFFT.TENSOR
    """
    Flag passed to inner AbstractFFT. The fastest FFTW is FFTW.MEASURE but takes the longest
    to precompute
    """
    fftflags::F = FFTW.MEASURE
end

# This a new function is overloaded to handle when NUFTPlan has plans
# as dictionaries in the case of Ti or Fr case

_rotatex(u, v, rm) = dot(rm[1, :], SVector(u, v))
_rotatey(u, v, rm) = dot(rm[2, :], SVector(u, v))

function plan_nuft_spatial(
        alg::NFFTAlg, imagegrid::AbstractRectiGrid,
        visdomain::UnstructuredDomain
    )
    visp = domainpoints(visdomain)
    uv2 = similar(visp.U, (2, length(visdomain)))
    dpx = pixelsizes(imagegrid)
    dx = dpx.X
    dy = dpx.Y
    rm = ComradeBase.rotmat(imagegrid)'
    # Here we flip the sign because the NFFT uses the -2pi convention
    uv2[1, :] .= -_rotatex.(visp.U, visp.V, Ref(rm)) .* dx
    uv2[2, :] .= -_rotatey.(visp.U, visp.V, Ref(rm)) .* dy
    (; reltol, precompute, fftflags) = alg
    plan = plan_nfft(NFFTBackend(), uv2, size(imagegrid)[1:2]; reltol, precompute, fftflags)
    return plan
end

function make_phases(::NFFTAlg, imgdomain::AbstractRectiGrid, visdomain::UnstructuredDomain)
    dx, dy = pixelsizes(imgdomain)
    x0, y0 = phasecenter(imgdomain)
    visp = domainpoints(visdomain)
    u = visp.U
    v = visp.V
    rm = ComradeBase.rotmat(imgdomain)'
    # Correct for the nFFT phase center and the img phase center
    return cispi.(
        (
            _rotatex.(u, v, Ref(rm)) .* (dx - 2 * x0) .+
                _rotatey.(u, v, Ref(rm)) .* (dy - 2 * y0)
        )
    )
end

# Allow NFFT to work with ForwardDiff.

# We split on a strided array since NFFT.jl only works on those
# and for StridedArrays we can potentially save an allocation
@inline function _nuft!(out::StridedArray, A, b::StridedArray)
    _jlnuft!(out, A, b)
    return nothing
end

@inline function _nuft!(out::AbstractArray, A, b::AbstractArray)
    tmp = similar(out)
    _jlnuft!(tmp, A, b)
    out .= tmp
    return nothing
end

@inline function _jlnuft!(out, A, b)
    mul!(out, A, b)
    return nothing
end

# Adding new NUFFT methods should overload this for Enzyme to work
function _jlnuft_adjointadd!(dI, A::NFFT.AbstractNFFTPlan, dv)
    dI .+= real.(A' * dv)
    return nothing
end


function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(_jlnuft!)},
        ::Type{RT},
        out::Annotation{<:AbstractArray{<:Complex}},
        A::Const,
        b::Annotation{<:AbstractArray{<:Real}}
    ) where {RT}
    # Forward rule does not have to return any primal or shadow since the original function returned nothing
    func.val(out.val, A.val, b.val)
    if EnzymeRules.width(config) == 1
        func.val(out.dval, A.val, b.dval)
    else
        ntuple(EnzymeRules.width(config)) do i
            Base.@_inline_meta
            return func.val(out.dval[i], A.val, b.dval[i])
        end
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_jlnuft!)}, ::Type{<:Const},
        out::Annotation,
        A::Annotation,
        b::Annotation{<:AbstractArray{<:Real}}
    )
    isa(A, Const) ||
        throw(ArgumentError("A must be a constant in NFFT. We don't support dynamic plans"))
    primal = EnzymeRules.needs_primal(config) ? out.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? out.dval : nothing
    cache_out = EnzymeRules.overwritten(config)[2] ? out : nothing
    cache_b = EnzymeRules.overwritten(config)[4] ? b : nothing
    tape = (cache_out, cache_b)
    _jlnuft!(out.val, A.val, b.val)
    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end


function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_jlnuft!)},
        ::Type{RT}, tape,
        out::Annotation, A::Annotation,
        b::Annotation{<:AbstractArray{<:Real}}
    ) where {RT}

    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    # cache_A = tape
    # if !(EnzymeRules.overwritten(config)[3])
    #     cache_A = A.val
    # end
    isa(A, Const) ||
        throw(ArgumentError("A must be a constant in NFFT. We don't support dynamic plans"))

    # There is no gradient to propagate so short
    if isa(out, Const)
        return (nothing, nothing, nothing)
    end

    outfwd = EnzymeRules.overwritten(config)[2] ? tape[1] : out
    bfwd = EnzymeRules.overwritten(config)[4] ? tape[2] : b

    # This is so Enzyme batch mode works
    dbs = if EnzymeRules.width(config) == 1
        (bfwd.dval,)
    else
        bfwd.dval
    end

    douts = if EnzymeRules.width(config) == 1
        (outfwd.dval,)
    else
        outfwd.dval
    end
    for (db, dout) in zip(dbs, douts)
        # TODO open PR on NFFT so we can do this in place.
        _jlnuft_adjointadd!(db, A.val, dout)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end
