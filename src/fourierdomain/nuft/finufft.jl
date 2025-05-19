export FINUFFTAlg

"""
    FINUFFTAlg
Uses the Flatiron non-uniform FFT FINUFFT to compute the visibilitymap.

# Fields
$(FIELDS)

"""
Base.@kwdef struct FINUFFTAlg{T, N, F} <: NUFT
    """
    Amount to pad the image
    """
    padfac::Int = 1
    """
    relative tolerance of the NFFT (FINUFFT eps)
    """
    reltol::T = 1.0e-9
    """
    how many threads to use
    """
    threads::N = 1
    """
    Flag passed to inner AbstractFFT. The fastest FFTW is FFTW.MEASURE but takes the longest
    to precompute
    """
    fftflags::F = FFTW.MEASURE
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

# EnzymeRules.inactive_type(::Type{<:AdjointFINPlan}) = true
# EnzymeRules.inactive_type(::Type{<:FINUFFTPlan}) = true
