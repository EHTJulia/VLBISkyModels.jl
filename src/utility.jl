export centroid_mean, center_image, convolve!, convolve, regrid, smooth

function centroid_mean(imgs::AbstractVector{<:IntensityMap})
    mimg = mapreduce(+, imgs) do img
        return center_image(img)
    end
    return mimg ./ length(imgs)
end

"""
    center_image(img::SpatialIntensityMap)

centers the `img` such that the centroid of the image is approximately at the origin.
"""
function center_image(img::SpatialIntensityMap)
    x, y = centroid(img)
    return modify(img, Shift(-x, -y))
end

# This is an internal struct that is use to modify IntensityMaps so that we can hook into
# VLBISkyModels image modifier interface.
struct InterpolatedImage{I,P} <: AbstractModel
    img::I
    itp::P
    function InterpolatedImage(img::SpatialIntensityMap)
        itp = BilinearInterpolator(img.X, img.Y, img, StrictBoundaries())
        return new{typeof(img),typeof(itp)}(img, itp)
    end
end

imanalytic(::Type{<:InterpolatedImage}) = IsAnalytic()
visanalytic(::Type{<:InterpolatedImage}) = NotAnalytic()
ispolarized(::Type{<:InterpolatedImage{<:IntensityMap{T}}}) where {T<:Real} = NotPolarized()
function ispolarized(::Type{<:InterpolatedImage{<:IntensityMap{T}}}) where {T<:StokesParams}
    return IsPolarized()
end

@inline function intensity_point(m::InterpolatedImage, p)
    g = axisdims(m.img)
    dx, dy = pixelsizes(g)
    X, Y = g.X, g.Y
    (X[begin] > p.X || p.X > X[end]) && return zero(eltype(m.img))
    (Y[begin] > p.Y || p.Y > Y[end]) && return zero(eltype(m.img))
    return m.itp(p.X, p.Y) / (dx * dy)
end
function ModifiedModel(img::SpatialIntensityMap,
                       transforms::NTuple{N,ModelModifier}) where {N}
    ms = ModifiedModel(InterpolatedImage(img), transforms)
    return intensitymap(ms, axisdims(img))
end

"""
    modify(img::IntensityMap, transforms...)

This modifies the `img` by applying the `transforms...` returning a transformed `IntensityMap`

!!! note
Unlike when `modify` is applied to a `<:AbstractModel` this returns an already modified image.
"""
modify(img::SpatialIntensityMap, transforms...) = ModifiedModel(img, transforms)

"""
    convolve!(img::IntensityMap, m::AbstractModel)

Convolves an `img` with a given analytic model `m`. This is useful for blurring the
image with some model. For instance to convolve a image with a Gaussian you would do
```julia
convolve!(img, Gaussian())
```

# Notes
This method does not automatically pad your image. If there is substantial flux at the boundaries
you will start to see artifacts.
"""
function convolve!(img::SpatialIntensityMap{<:Real}, m::AbstractModel)
    # short circuit if fill array since convolve is invariant
    ComradeBase.baseimage(img) isa FillArrays.Fill && return img

    @assert visanalytic(typeof(m)) isa IsAnalytic "Convolving model must have an analytic Fourier transform currently"
    # plan_rfft uses just the positive first axis to respect real conjugate symmetry
    (; X, Y) = img

    pimg = padimage(img, FFTAlg())
    # TODO maybe ask a user to pass a vis buffer as well?
    vis = fftshift(fft(ifftshift(pimg)))
    U = fftshift(fftfreq(size(pimg, 1), inv(step(X))))
    V = fftshift(fftfreq(size(pimg, 2), inv(step(Y))))
    guv = RectiGrid((U=U, V=V); executor=executor(img), header=header(img))
    puv = domainpoints(guv)

    vis .*= visibility_point.(Ref(shifted(m, step(X)/2, step(Y)/2)), puv)
    # phasedecenter!(vis, axisdims(img), griduv)
    cimg = fftshift(ifft(ifftshift(vis)))
    for i in CartesianIndices(img)
        img[i] = real(cimg[i])
    end
    return img
end

"""
    convolve(img::IntensityMap, m::AbstractModel)

Convolves an `img` with a given analytic model `m`. This is useful for blurring the
image with some model. For instance to convolve a image with a Gaussian you would do
```julia
convolve(img, Gaussian())
```

For the inplace version of the function see [`convolve!`](@ref)

# Notes
This method does not automatically pad your image. If there is substantial flux at the boundaries
you will start to see artifacts.
"""
function convolve(img::SpatialIntensityMap{<:Real}, m::AbstractModel)
    cimg = copy(img)
    return convolve!(cimg, m)
end

function convolve(img::SpatialIntensityMap{<:StokesParams}, m::AbstractModel)
    g = axisdims(img)
    bimg = copy(baseimage(img))
    cimg = IntensityMap(StructArray(bimg), g, refdims(img), name(img))
    return convolve!(cimg, m)
end

function convolve!(img::SpatialIntensityMap{<:StokesParams}, m)
    convolve!(stokes(img, :I), m)
    convolve!(stokes(img, :Q), m)
    convolve!(stokes(img, :U), m)
    convolve!(stokes(img, :V), m)
    return img
end

"""
    smooth(img::SpatialIntensityMap)

Smooths the `img` using a symmetric Gaussian with σ standard deviation.

For more flexible convolution please see [`convolve`](@ref).
"""
function smooth(img::SpatialIntensityMap, σ::Number)
    return convolve(img, modify(Gaussian(), Stretch(σ)))
end

# function convolve(img::IntensityMap, m::AbstractModel)
#     return map(x->convolve(x, m), eachslice(img, dims=(:X, :Y)))
# end

# """
#     $(SIGNATURES)

# Regrids the spatial parts of an image `img` on the new domain `g`
# """
# function regrid(img::IntensityMap, g::RectiGrid)
#     map(eachslice(img; dims=(:))) do simg
#         return regrid(simg, g)
#     end
# end

"""
    $(SIGNATURES)

Regrids the spatial parts of an image `img` on the new domain `g`
"""
function regrid(img::SpatialIntensityMap, g::RectiGrid)
    fimg = VLBISkyModels.InterpolatedImage(img)
    return intensitymap(fimg, g)
end
