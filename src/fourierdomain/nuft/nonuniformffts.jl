export NonuniformFFTAlg

"""
    NonuniFFTAlg
Uses the NonuniformFFTs NUFT to compute transforms to visibility space.

! warn 
  To use this you must have loaded NonuniformFFTs.jl in your session/REPL

# Fields
$(FIELDS)

"""
Base.@kwdef struct NonuniformFFTAlg{B, T, F} <: NUFT
    """
    Backend for the computation.
    """
    backend::B = nothing # default nothing will get replaced with CPU() from kernel abstractions
    """
    Amount to pad the image
    """
    padfac::Int = 1
    """
    NUFFT kernel size parameter. If negative will automatically decide based on reltol
    """
    m::Int = -1
    """
    The relative tolerance of the NUFFT kernel. If m is -1 this will be used to decide m.
    """
    reltol::T = 1e-9
    """
    NUFFT oversampling factor
    """
    sigma::T = 2.0
    """
    Flag passed to inner FFT for FFTW backend
    """
    fftflags::F = FFTW.MEASURE
end