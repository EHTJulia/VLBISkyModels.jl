export create_cache, DFTAlg

abstract type FourierTransform end

abstract type NUFT <: FourierTransform end

"""
    $(TYPEDEF)

This defines an abstract cache that can be used to
hold or precompute some computations.
"""
abstract type AbstractCache end


"""
    getgrid(c::AbstractCache)

Returns the grid used to compute the image cache
"""
getgrid(c::AbstractCache) = c.grid



"""
    create_cache(alg::AbstractFourierTransform, grid::AbstractGrid)

Creates a Fourier transform cache for the image grid using algorithm `alg`. For non-analytic visibility models this
can significantly speed up computations.

# Examples

```julia-repl
julia> u,v = rand(100), rand(100)
julia> g = imagepixels(μas2rad(100.0), μas2rad(100.0), 256, 256)
julia> cache = create_cache(NFFTAlg(u, v), g) # create a cache using a NUFFT this is fast and accurate
julia> cache = create_cache(FFTAlg(), g)      # create a cahce using a FFT. Fast but not as accurate
julia> cache = create_cache(DFTAlg(u, v), g)  # create a cache using the DTFT. Slow but accurate
```
"""
function create_cache end


"""
    $(TYPEDEF)

Use an FFT to compute the approximate numerical visibilities of a model.
For a DTFT see [`DFTAlg`](@ref DFTAlg) or for an NFFT [`NFFTAlg`](@ref NFFTAlg)

# Fields
$(FIELDS)

"""
Base.@kwdef struct FFTAlg <: FourierTransform
    """
    The amount to pad the image by.
    Note we actually round up to the nearest factor
    of 2, but this will be improved in the future to use
    small primes
    """
    padfac::Int = 2
end

"""
    $(TYPEDEF)
The cache used when the `FFT` algorithm is used to compute
visibilties. This is an internal type and is not part of the public API
"""
struct FFTCache{A<:FFTAlg,P,Pu,G,Guv} <: AbstractCache
    alg::A # FFT algorithm
    plan::P # FFT plan or matrix
    pulse::Pu
    grid::G
    gridUV::Guv
end

function Base.show(io::IO, a::T) where {T<:AbstractCache}
    st = split("$T", "{")[1]
    println(io, st, ": ")
    as = split("$(typeof(a.alg))", '{')[1]
    println(io, "\tFT algorithm: $as")
    println(io, "\tpulse: $(typeof(a.pulse))")
    sg = split("$(typeof(a.grid))", '{')[1]
    print(io, "\tdomain: ", sg, "$(propertynames(a.grid))")
end


include(joinpath(@__DIR__, "fft_alg.jl"))

"""
    $(TYPEDEF)

Container type for a non-uniform Fourier transform (NUFT).
This stores the uv-positions that the model will be sampled at in the Fourier domain,
allowing certain transformtion factors (e.g., NUFT matrix) to be cached.

This is an internal type, an end user should instead create this using [`NFFTAlg`](@ref NFFTAlg)
or [`DFTAlg`](@ref DFTAlg).
"""
struct ObservedNUFT{A<:NUFT, T} <: NUFT
    """
    Which NUFT algorithm to use (e.g. NFFTAlg or DFTAlg)
    """
    alg::A
    """
    uv positions of the NUFT transform. This is used for precomputation.
    """
    uv::T
end
padfac(a::ObservedNUFT) = padfac(a.alg)

"""
    $(TYPEDEF)

Internal type used to store the cache for a non-uniform Fourier transform (NUFT).

The user should instead create this using the [`create_cache`](@ref create_cache) function.
"""
struct NUFTCache{A,P,M,PI,G} <: AbstractCache
    alg::A # which algorithm to use
    plan::P #NUFT matrix or plan
    phases::M #FT phases needed to phase center things
    pulse::PI #pulse function to make continuous image
    grid::G # image grid
end
include(joinpath(@__DIR__, "nuft.jl"))


include(joinpath(@__DIR__, "modelimage.jl"))
