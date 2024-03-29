export modelimage

abstract type AbstractModelImage{M} <: ComradeBase.AbstractModel end

"""
    $(TYPEDEF)

Container for non-analytic model that contains a image cache which will hold the image,
a Fourier transform cache, which usually an instance of a <: FourierCache.

# Note

This is an internal implementation detail that shouldn't usually be called directly.
Instead the user should use the exported function `modelimage`, for example

```julia
using Comrade
m = ExtendedRing(20.0, 5.0)

# This creates an version where the image is dynamically specified according to the
# radial extent of the image
mimg = modelimage(m) # you can also optionally pass the number of pixels nx and ny

# Or you can create an IntensityMap
img = intensitymap(m, 100.0, 100.0, 512, 512)
mimg = modelimage(m, img)

# Or precompute a cache
cache = create_cache(FFTAlg(), img)
mimg = modelimage(m, cache)
```
"""
struct ModelImage{M,C} <: AbstractModelImage{M}
    model::M
    cache::C

    function ModelImage(model::AbstractModel, cache::FFTCache)
        minterp = InterpolatedModel(model, cache)
        return new{typeof(minterp), typeof(cache)}(minterp, image, cache)
    end

    function ModelImage(model::ModelImage{<:InterpolatedModel}, cache::FFTCache)
        minterp = InterpolatedModel(model.model.model, cache)
        return new{typeof(minterp), typeof(cache)}(minterp, image, cache)
    end


    function ModelImage(model::AbstractModel, cache::AbstractCache)
        return new{typeof(model), typeof(cache)}(model, image, cache)
    end

end
@inline visanalytic(::Type{<:ModelImage{M}}) where {M} = NotAnalytic()
@inline visanalytic(::Type{<:ModelImage{M, <:FFTCache}}) where {M} = IsAnalytic()
@inline imanalytic(::Type{<:ModelImage{M}}) where {M} = imanalytic(M)
@inline isprimitive(::Type{<:ModelImage{M}}) where {M} = isprimitive(M)
@inline ispolarized(::Type{<:ModelImage{M}}) where {M} = ispolarized(M)

# Default to using the cache grid for simplicity
intensitymap(m::ModelImage) = intensitymap(m, getgrid(m))

using Enzyme: EnzymeRules

# Now we define a bunch of getters and all set them to be non-differentiable
# since they should all be static
getplan(m::ModelImage{M, <:NUFTCache}) where {M} = m.cache.plan
EnzymeRules.inactive(::typeof(getplan), args...) = nothing
ChainRulesCore.@non_differentiable getplan(m)

getgrid(m::ModelImage) = m.cache.grid
ChainRulesCore.@non_differentiable getgrid(m::ModelImage)
EnzymeRules.inactive(::typeof(getgrid), args...) = nothing


getphases(m::ModelImage{M, <:NUFTCache}) where {M} = m.cache.phases
EnzymeRules.inactive(::typeof(getphases), args...) = nothing


@inline function visibility_point(mimg::ModelImage{M,<:FFTCache}, u, v, time, freq) where {M}
    return mimg.model.sitp(u, v)
end



function Base.show(io::IO, mi::ModelImage)
   #io = IOContext(io, :compact=>true)
   #s = summary(mi)
   ci = first(split(summary(mi.cache), "{"))
   println(io, "ModelImage(")
   println(io, "\tmodel: ", mi.model)
   si = split("$(typeof(mi.image))", ",")[1]*"}"*"$(size(mi.image))"
   println(io, "\timage: ", si)
   println(io, "\tcache: ", ci)
   print(io, ")")
end

model(m::AbstractModelImage) = m.model
flux(mimg::ModelImage) = flux(intensitymap(mimg, getgrid(mimg)))

# function intensitymap(mimg::ModelImage)
#     intensitymap!(mimg.image, mimg.model)
#     mimg.image
# end

intensitymap(mimg::ModelImage, g::ComradeBase.AbstractGrid) = intensitymap(model(mimg), g)
intensitymap!(img::IntensityMap, mimg::ModelImage) = intensitymap!(img, model(mimg))

radialextent(m::ModelImage) = hypot(fieldofview(getgrid(m))...)/2

#@inline visibility_point(m::AbstractModelImage, u, v) = visibility_point(model(m), u, v)

@inline intensity_point(m::AbstractModelImage, p) = intensity_point(model(m), p)

using Static

"""
    modelimage(model::AbstractModel, grid::AbstractGrid; alg=NFTAlg(), pulse=DeltaPulse(), thread=false)

Construct a `ModelImage` from a `model`, `grid` that specifies the domain of the image.
The keyword arguments are:
  - `alg`: specify the type of Fourier transform algorithm we will use. Default if the non-uniform FFT
  - `pulse`: Specify the pulse for the image model, the default is `DeltaPulse`

# Notes
For analytic models this is a no-op and returns the model.
For non-analytic models this creates a `ModelImage` object which uses `alg` to compute
the non-analytic Fourier transform.
"""
@inline function modelimage(model, grid::AbstractGrid; alg::FourierTransform=NFFTAlg(), pulse=DeltaPulse())
    return modelimage(model, grid, alg, pulse)
end

@inline function modelimage(model::M, grid::AbstractGrid, alg::FourierTransform, pulse=DeltaPulse()) where {M}
    return modelimage(visanalytic(M), model, grid, alg, pulse)
end

@inline function modelimage(::IsAnalytic, model, args...; kwargs...)
    return model
end

function _modelimage(model, grid, alg, pulse)
    cache = create_cache(alg, grid, pulse)
    return ModelImage(model, cache)
end

@inline function modelimage(::NotAnalytic, model::AbstractModel, cache::FFTCache)
    return ModelImage(model, cache)
end




@inline function modelimage(::NotAnalytic, model,
                            grid::AbstractGrid,
                            alg::FourierTransform=FFTAlg(),
                            pulse = DeltaPulse(),
                            )
    _modelimage(model, grid, alg, pulse)
end

"""
    modelimage(model, cache::AbstractCach))

Construct a `ModelImage` from the `model` and using a precompute Fourier transform `cache`.
You can optionally specify th which will compute the internal image buffer using
the`.

# Example

```julia-repl
julia> m = ExtendedRing(10.0)
julia> cache = create_cache(DFTAlg(), IntensityMap(zeros(128, 128), 50.0, 50.0)) # used threads to make the image
julia> mimg = modelimage(m, cache, true)
```

# Notes
For analytic models this is a no-op and returns the model.

"""
@inline function modelimage(model::M, cache::AbstractCache) where {M}
    return modelimage(visanalytic(M), model, cache, static(thread))
end


@inline function modelimage(::NotAnalytic, model, cache::AbstractCache)
    return ModelImage(model, cache)
end


function nocachevis(m::ModelImage{M,<:NUFTCache}, u, v, time, freq) where {M}
    alg = ObservedNUFT(m.cache.alg, vcat(u', v'))
    cache = create_cache(alg, getgrid(grid))
    m = @set m.cache = cache
    return visibilities_numeric(m, u, v, time, freq)
end




ChainRulesCore.@non_differentiable checkuv(alg, u::AbstractArray, v::AbstractArray)
EnzymeRules.inactive(::typeof(checkuv), args...) = nothing

function visibilities_numeric(m::ModelImage{M,<:NUFTCache{A}},
                      u, v, time, freq) where {M,A<:ObservedNUFT}
    checkuv(m.cache.alg.uv, u, v)
    img = intensitymap(m)
    vis =  nuft(getplan(m), ComradeBase.baseimage(img))
    return conj.(vis).*getphases(m)
end

function visibilities_numeric(m::ModelImage{M,<:NUFTCache{A}},
                      u, v, time, freq) where {M,A<:NUFT}
    return nocachevis(m, u, v, time, freq)
end
