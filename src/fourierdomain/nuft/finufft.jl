export FINUFFTAlg

"""
    FINUFFTAlg
Uses the Flatiron non-uniform FFT FINUFFT to compute the visibilitymap.

# Fields
$(FIELDS)

"""
struct FINUFFTAlg{GPU, T, N, F} <: NUFT
    """
    Amount to pad the image
    """
    padfac::Int
    """
    relative tolerance of the NFFT (FINUFFT eps)
    """
    reltol::T
    """
    how many threads to use
    """
    threads::N
    """
    Flag passed to inner AbstractFFT. The fastest FFTW is FFTW.MEASURE but takes the longest
    to precompute
    """
    fftflags::F
end

"""
    FINUFFTAlg(padfac::Int = 1, reltol::T = 1.0e-9, threads::N = 1,
               fftflags::F = FFTW.MEASURE, gpu::Bool = false)

Use the FINUFFT software for the non-uniform FFT.

Keyword arguments:
- `padfac::Int = 1`: Amount to pad the image
- `reltol::T = 1.0e-9`: Relative tolerance of the NFFT (FINUFFT eps)
- `threads::N = 1`: How many threads to use
- `fftflags::F = FFTW.MEASURE`: Flag passed to inner AbstractFFT. The fastest FFTW is `FFTW.MEASURE` but takes the longest
  to precompute
- `gpu::Bool = false`: Whether to use the GPU version of FINUFFT. If `true` then CUDA must be installed and loaded.

"""
function FINUFFTAlg(;
        padfac::Int = 1, reltol::T = 1.0e-9, threads::N = 1,
        fftflags::F = FFTW.MEASURE, gpu::Bool = false
    ) where {T, N <: Integer, F}
    return FINUFFTAlg{gpu, T, N, F}(padfac, reltol, threads, fftflags)
end

# This is an internal struct that holds the plans for the forward and inverse plans.
# We need this for the adjoint where we need to use the inverse plan
mutable struct FINUFFTPlan{T, VS, IS, P1, P2, A}
    const vsize::VS # The size of the visibilities
    const isize::IS # The size of the image
    """The forward img->vis plan."""
    forward::P1
    """The inverse vis->img plan."""
    adjoint::P1
    const ccache::A # Complex image cache that should prevent allocations
end

function FINUFFTPlan(
        vsize::VS, isize::IS, forward::P1, adjoint::P2,
        compleximg::A
    ) where {VS, IS, P1, P2, A}
    T = typeof(compleximg[1].re)
    return FINUFFTPlan{T, VS, IS, P1, P2, A}(vsize, isize, forward, adjoint, compleximg)
end

Base.eltype(::FINUFFTPlan{T}) where {T} = Complex{T}
vissize(p::FINUFFTPlan) = p.vsize

struct AdjointFINPlan{P}
    plan::P
end
vissize(p::AdjointFINPlan) = p.plan.isize

Base.adjoint(p::FINUFFTPlan) = AdjointFINPlan(p)
Base.adjoint(p::AdjointFINPlan) = p.plan

EnzymeRules.inactive_type(::Type{<:AdjointFINPlan}) = true
EnzymeRules.inactive_type(::Type{<:FINUFFTPlan}) = true

# Internal function that saves an allocation by using internal cache
function _adjjlnuftadd! end

@noinline function getcache(A::FINUFFTPlan)
    return A.ccache
end

@noinline function getcache(A::AdjointFINPlan)
    return A.plan.ccache
end


@inline function _nuft!(out::StridedArray, A::FINUFFTPlan, b::StridedArray)
    _jlnuft!(out, A, b)
    return nothing
end


function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(_jlnuft!)},
        ::Type{RT},
        out::Annotation{<:AbstractArray{<:Complex}},
        A::Const{<:FINUFFTPlan},
        b::Annotation{<:AbstractArray{<:Real}}
    ) where {RT}
    # Forward rule does not have to return any primal or shadow since the original function returned nothing
    _jlnuft(out.val, A.val, b.val)

    if EnzymeRules.width(config) == 1
        _jlnuft(out.dval, A.val, b.dval)
    else
        ntuple(EnzymeRules.width(config)) do i
            Base.@_inline_meta
            return _jlnuft(out.dval[i], A.val, b.dval[i])
        end
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_jlnuft!)}, ::Type{<:Const},
        out::Annotation,
        A::Annotation{<:FINUFFTPlan},
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
        out::Annotation, A::Annotation{<:FINUFFTPlan},
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
        _adjjlnuftadd!(db, A.val, dout)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end
