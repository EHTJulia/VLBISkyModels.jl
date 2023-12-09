export modelimage

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
struct ModelImage{M,I,C} <: AbstractModelImage{M}
    model::M
    image::I
    cache::C

    function ModelImage(model::AbstractModel, image::IntensityMapTypes, cache::FFTCache)
        minterp = InterpolatedModel(model, cache)
        return new{typeof(minterp), typeof(image), typeof(cache)}(minterp, image, cache)
    end

    function ModelImage(model::ModelImage{<:InterpolatedModel}, image::IntensityMapTypes, cache::FFTCache)
        minterp = InterpolatedModel(model.model.model, cache)
        return new{typeof(minterp), typeof(image), typeof(cache)}(minterp, image, cache)
    end


    function ModelImage(model::AbstractModel, image::IntensityMapTypes, cache::AbstractCache)
        return new{typeof(model), typeof(image), typeof(cache)}(model, image, cache)
    end

end
@inline visanalytic(::Type{<:ModelImage{M}}) where {M} = NotAnalytic()
@inline visanalytic(::Type{<:ModelImage{M, I, <:FFTCache}}) where {M,I} = IsAnalytic()
@inline imanalytic(::Type{<:ModelImage{M}}) where {M} = imanalytic(M)
@inline isprimitive(::Type{<:ModelImage{M}}) where {M} = isprimitive(M)
@inline ispolarized(::Type{<:ModelImage{M}}) where {M} = ispolarized(M)


@inline function visibility_point(mimg::ModelImage{M,I,<:FFTCache}, u, v, time, freq) where {M,I}
    return mimg.model.sitp(u, v)
end




function Base.show(io::IO, mi::ModelImage)
   #io = IOContext(io, :compact=>true)
   #s = summary(mi)
   ci = first(split(summary(mi.cache), "{"))
   println(io, "ModelImage")
   println(io, "\tmodel: ", summary(mi.model))
   println(io, "\timage: ", summary(mi.image))
   println(io, "\tcache: ", ci)
end

model(m::AbstractModelImage) = m.model
flux(mimg::ModelImage) = flux(mimg.image)

# function intensitymap(mimg::ModelImage)
#     intensitymap!(mimg.image, mimg.model)
#     mimg.image
# end

intensitymap(mimg::ModelImage, g::ComradeBase.AbstractDims) = intensitymap(mimg.model, g)
intensitymap!(img::IntensityMap, mimg::ModelImage) = intensitymap!(img, mimg.model)

radialextent(m::ModelImage) = hypot(fieldofview(m.image)...)/2

#@inline visibility_point(m::AbstractModelImage, u, v) = visibility_point(model(m), u, v)

@inline intensity_point(m::AbstractModelImage, p) = intensity_point(model(m), p)

using Static

"""
    modelimage(model::AbstractModel, image::AbstractIntensityMap, alg=FFTAlg())

Construct a `ModelImage` from a `model`, `image` and the optionally
specified visibility algorithm `alg`

# Notes
For analytic models this is a no-op and returns the model.
For non-analytic models this creates a `ModelImage` object which uses `alg` to compute
the non-analytic Fourier transform.
"""
@inline function modelimage(model::M, grid::AbstractDims, alg::FourierTransform=FFTAlg(), pulse=DeltaPulse(), thread::Union{Bool, StaticBool}=false) where {M}
    return modelimage(visanalytic(M), model, grid, alg, pulse, static(thread))
end

@inline function modelimage(::IsAnalytic, model, args...; kwargs...)
    return model
end

function _modelimage(model, grid, alg, pulse, thread::StaticBool)
    image = intensitymap(model, grid, thread)
    cache = create_cache(alg, grid, pulse)
    return ModelImage(model, image, cache)
end

@inline function modelimage(::NotAnalytic, model::AbstractModel, cache::FFTCache, thread::StaticBool)
    img = intensitymap(model, cache.grid, thread)
    return ModelImage(model, img, cache)
end




@inline function modelimage(::NotAnalytic, model,
                            grid::AbstractDims,
                            alg::FourierTransform=FFTAlg(),
                            pulse = DeltaPulse(),
                            thread::StaticBool = False()
                            )
    _modelimage(model, grid, alg, pulse, thread)
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
@inline function modelimage(model::M, cache::AbstractCache, thread::Union{Bool, StaticBool}=false) where {M}
    return modelimage(visanalytic(M), model, cache, static(thread))
end


@inline function modelimage(::NotAnalytic, model, cache::AbstractCache, thread::StaticBool)
    img = intensitymap(model, cache.grid, thread)
    return ModelImage(model, img, cache)
end


"""
    modelimage(m;
               fovx=2*radialextent(m),
               fovy=2*radialextent(m),
               nx=512,
               ny=512,
               alg=FFTAlg(),
               pulse=ComradeBase.DeltaPulse(),
                )

Construct a `ModelImage` where just the model `m` is specified.

If `fovx` or `fovy` aren't given `modelimage` will *guess* a reasonable field of view based
on the `radialextent` function. `nx` and `ny` are the number of pixels in the x and y
direction. The `pulse` is the pulse used for the image and `alg`

# Notes
For analytic models this is a no-op and returns the model.

"""
function modelimage(m::M;
                    fovx = 2*radialextent(m),
                    fovy = 2*radialextent(m),
                    nx = 512,
                    ny = 512,
                    x0 = zero(fovx),
                    y0 = zero(fovx),
                    alg=FFTAlg(),
                    pulse = DeltaPulse(),
                    thread::Bool = false
                    ) where {M}
    if visanalytic(M) == IsAnalytic()
        return m
    else
        dims = imagepixels(fovx, fovy, nx, ny, x0, y0)
        return modelimage(m, dims, alg, pulse, thread)
    end
end

function nocachevis(m::ModelImage{M,I,<:NUFTCache}, u, v, time, freq) where {M,I<:IntensityMap}
    alg = ObservedNUFT(m.cache.alg, vcat(u', v'))
    cache = create_cache(alg, m.cache.grid)
    m = @set m.cache = cache
    return visibilities_numeric(m, u, v, time, freq)
end


using Enzyme: EnzymeRules
getplan(m::ModelImage{M, I, <:NUFTCache}) where {M, I} = m.cache.plan
EnzymeRules.inactive(::typeof(getplan), args...) = nothing
ChainRulesCore.@non_differentiable getplan(m)
EnzymeRules.inactive(::typeof(checkuv), args...) = nothing
getphases(m::ModelImage{M, I, <:NUFTCache}) where {M,I} = m.cache.phases
EnzymeRules.inactive(::typeof(getphases), args...) = nothing

#using ReverseDiff
#using NFFT
#ReverseDiff.@grad_from_chainrules nuft(A, b::ReverseDiff.TrackedArray)
#ReverseDiff.@grad_from_chainrules nuft(A, b::Vector{<:ReverseDiff.TrackedReal})


ChainRulesCore.@non_differentiable checkuv(alg, u::AbstractArray, v::AbstractArray)

function visibilities_numeric(m::ModelImage{M,I,<:NUFTCache{A}},
                      u, v, time, freq) where {M,I<:IntensityMap,A<:ObservedNUFT}
    checkuv(m.cache.alg.uv, u, v)
    vis =  nuft(getplan(m), complex(ComradeBase.baseimage(m.image)))
    return conj.(vis).*getphases(m)
end

function visibilities_numeric(m::ModelImage{M,I,<:NUFTCache{A}},
                      u, v, time, freq) where {M,I<:StokesIntensityMap,A<:ObservedNUFT}
    checkuv(m.cache.alg.uv, u, v)
    visI =  conj.(nuft(getplan(m), complex(ComradeBase.baseimage(stokes(m.image, :I))))).*getphases(m)
    visQ =  conj.(nuft(getplan(m), complex(ComradeBase.baseimage(stokes(m.image, :Q))))).*getphases(m)
    visU =  conj.(nuft(getplan(m), complex(ComradeBase.baseimage(stokes(m.image, :U))))).*getphases(m)
    visV =  conj.(nuft(getplan(m), complex(ComradeBase.baseimage(stokes(m.image, :V))))).*getphases(m)
    r = StructArray{StokesParams{eltype(visI)}}((I=visI, Q=visQ, U=visU, V=visV))
    return r
end


function visibilities_numeric(m::ModelImage{M,I,<:NUFTCache{A}},
                      u, v, time, freq) where {M,I,A<:NUFT}
    return nocachevis(m, u, v, time, freq)
end

function _frule_vis(m::ModelImage{M,<:SpatialIntensityMap{<:ForwardDiff.Dual{T,V,P}},<:NUFTCache{O}}) where {M,T,V,P,A<:NFFTAlg,O<:ObservedNUFT{A}}
    p = m.cache.plan
    # Compute the fft
    bimg = parent(m.image)
    buffer = ForwardDiff.value.(bimg)
    xtil = p*complex.(buffer)
    out = similar(buffer, Complex{ForwardDiff.Dual{T,V,P}})
    # Now take the deriv of nuft
    ndxs = ForwardDiff.npartials(first(m.image))
    dxtils = ntuple(ndxs) do n
        buffer .= ForwardDiff.partials.(m.image, n)
        p * complex.(buffer)
    end
    out = similar(xtil, Complex{ForwardDiff.Dual{T,V,P}})
    for i in eachindex(out)
        dual = getindex.(dxtils, i)
        prim = xtil[i]
        red = ForwardDiff.Dual{T,V,P}(real(prim), ForwardDiff.Partials(real.(dual)))
        imd = ForwardDiff.Dual{T,V,P}(imag(prim), ForwardDiff.Partials(imag.(dual)))
        out[i] = Complex(red, imd)
    end
    return out
end

function visibilities_numeric(m::ModelImage{M,<:SpatialIntensityMap{<:ForwardDiff.Dual{T,V,P}},<:NUFTCache{O}},
    u::AbstractArray,
    v::AbstractArray,
    time,
    freq) where {M,T,V,P,A<:NFFTAlg,O<:ObservedNUFT{A}}
    checkuv(m.cache.alg.uv, u, v)
    # Now reconstruct everything

    vis = _frule_vis(m)
    return conj.(vis).*m.cache.phases
end
