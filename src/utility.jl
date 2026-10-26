export centroid_mean, center_image, convolve!, convolve, regrid, smooth

@inline function _rfft_plan(::Type{T}, shape::NTuple{N, Int}) where {T <: Real, N}
    return plan_rfft(zeros(T, shape), 1:2)
end

@inline function _irfft_plan(::Type{T}, shape::NTuple{N, Int}) where {T <: Real, N}
    spectrum_shape = (div(shape[1], 2) + 1, Base.tail(shape)...)
    return plan_irfft(zeros(Complex{T}, spectrum_shape), shape[1], 1:2)
end

EnzymeRules.inactive(::typeof(_rfft_plan), args...) = nothing
EnzymeRules.inactive(::typeof(_irfft_plan), args...) = nothing

function _convolution_uvpoints(grid, shape)
    (; X, Y) = grid
    u = U(rfftfreq(shape[1], inv(step(X))))
    v = V(fftfreq(shape[2], inv(step(Y))))
    frequency_dims = (u, v, dims(grid)[3:end]...)
    return domainpoints(rebuild(grid; dims = frequency_dims))
end

EnzymeRules.inactive(::typeof(_convolution_uvpoints), args...) = nothing

@inline function _apply_rfft!(out, plan, input)
    mul!(out, plan, input)
    return nothing
end

@inline function _apply_irfft!(out, plan, input)
    mul!(out, plan, input)
    return nothing
end

function _rfft_adjoint_add!(input_gradient, spectrum_gradient)
    weighted_spectrum_gradient = copy(spectrum_gradient)
    n = size(input_gradient, 1)
    interior = iseven(n) ? (2:(size(weighted_spectrum_gradient, 1) - 1)) :
               (2:size(weighted_spectrum_gradient, 1))
    trailing_indices = ntuple(_ -> Colon(), ndims(weighted_spectrum_gradient) - 1)
    view(weighted_spectrum_gradient, interior, trailing_indices...) ./= 2
    input_gradient .+= brfft(weighted_spectrum_gradient, n, 1:2)
    return nothing
end

function _irfft_adjoint_add!(spectrum_gradient, image_gradient)
    image_size = size(image_gradient, 1) * size(image_gradient, 2)
    transformed_gradient = rfft(image_gradient, 1:2)
    n = size(image_gradient, 1)
    interior = iseven(n) ? (2:(size(transformed_gradient, 1) - 1)) :
               (2:size(transformed_gradient, 1))
    trailing_indices = ntuple(_ -> Colon(), ndims(transformed_gradient) - 1)
    view(transformed_gradient, interior, trailing_indices...) .*= 2 / image_size
    view(transformed_gradient, 1, trailing_indices...) ./= image_size
    iseven(n) && (view(transformed_gradient, size(transformed_gradient, 1), trailing_indices...) ./= image_size)
    spectrum_gradient .+= transformed_gradient
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_apply_rfft!)},
        ::Type{<:Const},
        out::Annotation,
        plan::Annotation,
        input::Annotation,
    )
    plan isa Const || throw(ArgumentError("FFTW plans must be constant during differentiation"))
    _apply_rfft!(out.val, plan.val, input.val)
    primal = EnzymeRules.needs_primal(config) ? out.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? out.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_apply_rfft!)},
        ::Type{RT},
        _,
        out::Annotation,
        plan::Annotation,
        input::Annotation,
    ) where {RT}
    plan isa Const || throw(ArgumentError("FFTW plans must be constant during differentiation"))
    out isa Const && return (nothing, nothing, nothing)
    input isa Const && return (nothing, nothing, nothing)
    output_gradients = EnzymeRules.width(config) == 1 ? (out.dval,) : out.dval
    input_gradients = EnzymeRules.width(config) == 1 ? (input.dval,) : input.dval
    for (input_gradient, output_gradient) in zip(input_gradients, output_gradients)
        _rfft_adjoint_add!(input_gradient, output_gradient)
        fill!(output_gradient, 0)
    end
    return (nothing, nothing, nothing)
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_apply_irfft!)},
        ::Type{<:Const},
        out::Annotation,
        plan::Annotation,
        input::Annotation,
    )
    plan isa Const || throw(ArgumentError("FFTW plans must be constant during differentiation"))
    _apply_irfft!(out.val, plan.val, input.val)
    primal = EnzymeRules.needs_primal(config) ? out.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? out.dval : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
end

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_apply_irfft!)},
        ::Type{RT},
        _,
        out::Annotation,
        plan::Annotation,
        input::Annotation,
    ) where {RT}
    plan isa Const || throw(ArgumentError("FFTW plans must be constant during differentiation"))
    out isa Const && return (nothing, nothing, nothing)
    input isa Const && return (nothing, nothing, nothing)
    output_gradients = EnzymeRules.width(config) == 1 ? (out.dval,) : out.dval
    input_gradients = EnzymeRules.width(config) == 1 ? (input.dval,) : input.dval
    for (input_gradient, output_gradient) in zip(input_gradients, output_gradients)
        _irfft_adjoint_add!(input_gradient, output_gradient)
        fill!(output_gradient, 0)
    end
    return (nothing, nothing, nothing)
end

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
struct InterpolatedImage{I, P} <: AbstractModel
    img::I
    itp::P
end

function InterpolatedImage(img::IntensityMap)
    itp = RectangleGrid(map(ComradeBase.basedim, dims(img))...)
    return InterpolatedImage{typeof(img), typeof(itp)}(img, itp)
end

function InterpolatedImage(img::IntensityMap, itp::RectangleGrid)
    return InterpolatedImage{typeof(img), typeof(itp)}(img, itp)
end

imanalytic(::Type{<:InterpolatedImage}) = IsAnalytic()
visanalytic(::Type{<:InterpolatedImage}) = NotAnalytic()
ispolarized(::Type{<:InterpolatedImage{<:IntensityMap{T}}}) where {T <: Number} = NotPolarized()
function ispolarized(::Type{<:InterpolatedImage{<:IntensityMap{T}}}) where {T <: StokesParams}
    return IsPolarized()
end

@inline function intensity_point(m::InterpolatedImage, p)
    g = axisdims(m.img)
    dx, dy = pixelsizes(g)
    X, Y = g.X, g.Y
    rm = rotmat(g)'
    X2 = _rotatex(p.X, p.Y, rm)
    Y2 = _rotatey(p.X, p.Y, rm)
    (X[begin] > X2 || X2 > X[end]) && return zero(eltype(m.img))
    (Y[begin] > Y2 || Y2 > Y[end]) && return zero(eltype(m.img))
    # - sign is because we need to move into the frame of the vertical-horizontal image
    p2 = ComradeBase.update_spat(p, X2, Y2)
    return interpolate(m.itp, m.img, SVector(values(p2))) / (dx * dy)
end
function ModifiedModel(
        img::IntensityMap,
        transforms::NTuple{N, ModelModifier}
    ) where {N}
    ms = ModifiedModel(InterpolatedImage(img), transforms)
    return intensitymap(ms, axisdims(img))
end

"""
    modify(img::IntensityMap, transforms...)

This modifies the `img` by applying the `transforms...` returning a transformed `IntensityMap`

!!! note
Unlike when `modify` is applied to a `<:AbstractModel` this returns an already modified image.
"""
modify(img::IntensityMap, transforms...) = ModifiedModel(img, transforms)

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
function convolve!(img::IntensityMap{<:Real}, m::AbstractModel)
    # short circuit if fill array since convolve is invariant
    ComradeBase.baseimage(img) isa FillArrays.Fill && return img

    @assert visanalytic(typeof(m)) isa IsAnalytic "Convolving model must have an analytic Fourier transform currently"
    image = baseimage(img)
    shape = size(image)
    p = _rfft_plan(eltype(image), shape)

    # plan_rfft uses just the positive first axis to respect real conjugate symmetry
    puv = _convolution_uvpoints(axisdims(img), shape)

    # TODO maybe ask a user to pass a vis buffer as well?
    vis = Array{Complex{eltype(image)}}(undef, div(shape[1], 2) + 1, Base.tail(shape)...)
    _apply_rfft!(vis, p, image)

    # Conjugate because Comrade uses +2πi exponent.
    for index in eachindex(vis, puv)
        vis[index] *= conj(visibility_point(m, puv[index]))
    end
    pinv = _irfft_plan(eltype(image), shape)
    _apply_irfft!(image, pinv, vis)
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
function convolve(img::IntensityMap{<:Real}, m::AbstractModel)
    cimg = copy(img)
    return convolve!(cimg, m)
end

function convolve(img::IntensityMap{<:StokesParams}, m::AbstractModel)
    g = axisdims(img)
    bimg = copy(baseimage(img))
    cimg = IntensityMap(StructArray(bimg), g, refdims(img), name(img))
    return convolve!(cimg, m)
end

function convolve!(img::IntensityMap{<:StokesParams}, m)
    convolve!(stokes(img, :I), m)
    convolve!(stokes(img, :Q), m)
    convolve!(stokes(img, :U), m)
    convolve!(stokes(img, :V), m)
    return img
end

"""
    smooth(img::IntensityMap)

Smooths the `img` using a symmetric Gaussian with σ standard deviation.

For more flexible convolution please see [`convolve`](@ref).
"""
function smooth(img::IntensityMap, σ::Number)
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
function regrid(img::IntensityMap, g::RectiGrid)
    fimg = VLBISkyModels.InterpolatedImage(img)
    return intensitymap(fimg, g)
end
