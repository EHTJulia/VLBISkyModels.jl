export NonuniFFTAlg

"""
    NonuniFFTAlg
Uses the NonuniformFFTs NUFT to compute transforms to visibility space.

! warn 
To use this you must have loaded NonuniformFFTs.jl in your environment

# Fields
$(FIELDS)

"""
Base.@kwdef struct NonuniFFTAlg{B, T, N, F} <: NUFT
    """
    Backend for the computation.
    """
    backend::B = nothing # default nothing will get replaced with CPU() from kernel abstractions
    """
    Amount to pad the image
    """
    padfac::Int = 1
    """
    NUFFT kernel size parameter
    """
    m::Int = 1
    """
    NUFFT oversampling factor
    """
    sigma::T = 2.0
    """
    Flag passed to inner FFT for FFTW backend
    """
    fftflags::F = FFTW.MEASURE
end

# This is an internal struct that holds the plans for the forward and inverse plans.
# We need this for the adjoint where we need to use the inverse plan
struct NonuniformPlan{T, P1, P2, A}
    """The forward img->vis plan."""
    forward::P1
    """The inverse vis->img plan."""
    adjoint::P1
    ccache::A # Complex image cache that should prevent allocations
end


Base.eltype(::NonuniformPlan{T}) where {T} = Complex{T}
Base.size(p::NonuniformPlan) = p.vsize
EnzymeRules.inactive_type(::Type{<:NonuniformPlan}) = true
