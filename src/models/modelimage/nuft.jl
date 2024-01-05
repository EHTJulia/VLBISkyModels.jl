padfac(alg::NUFT) = alg.padfac

function padimage(::NUFT, img::IntensityMapTypes)
    #pf = padfac(alg)
    #cimg = convert(Matrix{Complex{eltype(img)}}, img.img)
    return img
    # ny,nx = size(img)
    # nnx = nextpow(2, pf*nx)
    # nny = nextpow(2, pf*ny)
    # nsx = nnx÷2-nx÷2
    # nsy = nny÷2-ny÷2
    # cimg = convert(Matrix{Complex{eltype(img)}}, img)
    # return PaddedView(zero(eltype(cimg)), cimg,
    #                   (1:nnx, 1:nny),
    #                   (nsx+1:nsx+nx, nsy+1:nsy+ny)
    #                  )
end

padimage(alg::ObservedNUFT, img::IntensityMapTypes) = padimage(alg.alg, img)


function create_cache(alg::ObservedNUFT, grid::AbstractGrid, pulse::Pulse=DeltaPulse())

    # make nuft plan
    plan = plan_nuft(alg, grid)
    # get phases and pulse functions
    phases = make_phases(alg, grid, pulse)

    return create_cache(alg, plan, phases, grid, pulse)
end

function create_cache(alg::NUFT, grid::AbstractGrid, pulse::Pulse=DeltaPulse())
    return NUFTCache(alg, nothing, nothing, pulse, grid)
end


function checkuv(uv, u, v)
    @assert u == @view(uv[1,:]) "Specified u don't match uv in cache. Did you pass the correct u,v?"
    @assert v == @view(uv[2,:]) "Specified v don't match uv in cache. Did you pass the correct u,v?"
end

function Serialization.serialize(s::Serialization.AbstractSerializer, cache::NUFTCache{<:ObservedNUFT})
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.alg)
    Serialization.serialize(s, cache.pulse)
    Serialization.serialize(s, cache.grid)

end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:NUFTCache{<:ObservedNUFT}})
    alg = Serialization.deserialize(s)
    pulse = Serialization.deserialize(s)
    grid = Serialization.deserialize(s)
    return create_cache(alg, grid, pulse)
end




"""
    NFFTAlg
Uses a non-uniform FFT to compute the visibilities.
You can optionally pass uv which are the uv positions you will
compute the NFFT at. This can allow for the NFFT plan to be cached improving
performance

# Fields
$(FIELDS)

"""
Base.@kwdef struct NFFTAlg{T,N,F} <: NUFT
    """
    Amount to pad the image
    """
    padfac::Int = 1
    """
    Kernel size parameters. This controls the accuracy of NFFT you do not usually need to change this
    """
    m::Int = 4
    """
    Over sampling factor. This controls the accuracy of NFFT you do not usually need to change this.
    """
    σ::T = 2.0
    """
    Window function for the NFFT. You do not usually need to change this
    """
    window::Symbol = :kaiser_bessel
    """
    NFFT interpolation algorithm.
    """
    precompute::N=NFFT.POLYNOMIAL
    """
    Flag block partioning should be used to speed up computation
    """
    blocking::Bool = true
    """
    Flag if the node should be sorted in a lexicographic way
    """
    sortNodes::Bool = false
    """
    Flag if the deconvolve indices should be stored, Currently required for GPU
    """
    storeDeconvolutionIdx::Bool = true
    """
    Flag passed to inner AbstractFFT. The fastest FFTW is FFTW.MEASURE but takes the longest
    to precompute
    """
    fftflags::F = FFTW.MEASURE
end
include(joinpath(@__DIR__, "nfft_alg.jl"))

"""
    DFTAlg
Uses a discrete fourier transform. This is not very efficient for larger images. In those cases
 NFFTAlg or FFTAlg are more reasonable. For small images this is a reasonable choice especially
since it's easy to define derivatives.
"""
struct DFTAlg <: NUFT end
include(joinpath(@__DIR__, "dft_alg.jl"))
