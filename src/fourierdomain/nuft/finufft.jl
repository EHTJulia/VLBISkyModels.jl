export FINUFFTAlg

"""
    FINUFFTAlg
Uses the Flatiron non-uniform FFT FINUFFT to compute the visibilitymap.

# Fields
$(FIELDS)

"""
struct FINUFFTAlg{T, N, F} <: NUFT
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
               fftflags::F = FFTW.MEASURE)

Use the FINUFFT software for the non-uniform FFT.

Keyword arguments:
- `padfac::Int = 1`: Amount to pad the image
- `reltol::T = 1.0e-9`: Relative tolerance of the NFFT (FINUFFT eps)
- `threads::N = 1`: How many threads to use
- `fftflags::F = FFTW.MEASURE`: Flag passed to inner AbstractFFT. The fastest FFTW is `FFTW.MEASURE` but takes the longest
  to precompute
"""
function FINUFFTAlg(;
        padfac::Int = 1, reltol::T = 1.0e-9, threads::N = 1,
        fftflags::F = FFTW.MEASURE
    ) where {T, N <: Integer, F}
    return FINUFFTAlg{T, N, F}(padfac, reltol, threads, fftflags)
end

# This is an internal struct that holds the plans for the forward and inverse plans.
# We need this for the adjoint where we need to use the inverse plan
mutable struct FINUFFTPlan{P1, VS, IS, A}
    """The forward img->vis plan."""
    forward::P1
    """The inverse vis->img plan."""
    adjoint::P1
    const vsize::VS # The size of the visibilities
    const isize::IS # The size of the image
    const ccache::A # Complex image cache that should prevent allocations
end

function FINUFFTPlan(
        vsize::VS, isize::IS, forward::P1, adjoint::P1,
        compleximg::A
    ) where {VS, IS, P1, A}
    T = typeof(compleximg[1].re)
    return FINUFFTPlan{ A}(vsize, isize, forward, adjoint, compleximg)
end

Base.eltype(A::FINUFFTPlan) = eltype(getcache(A))
vissize(p::FINUFFTPlan) = p.vsize

EnzymeRules.inactive_type(::Type{<:FINUFFTPlan}) = true


@noinline function getcache(A::FINUFFTPlan)
    return A.ccache
end

# Internal method to make a FINUFFT plan
function make_plan_finufft end