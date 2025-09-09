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