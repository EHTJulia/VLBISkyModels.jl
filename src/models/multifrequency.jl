export Multifrequency, TaylorSpectral, applyspectral, applyspectral!, generatemodel,
       visibilitymap_numeric, build_imagecube, mfimagepixels

"""Abstract type to hold all multifrequency spectral models"""
abstract type AbstractSpectralModel end

"""
    $(TYPEDEF)
Spectral Model object.

# Fields
$(FIELDS)
"""
struct Multifrequency{B<:ContinuousImage,F<:Number,S<:FrequencyParams} <:
       ComradeBase.AbstractModel
    """
    Base image model (ContinuousImage only)
    """
    base::B
    """
    The base image reference frequency.
    """
    ν0::F
    """
    Multifrequency spectral model
    """
    spec::S
end

# defining the mandatory methods for a Comrade AbstractModel
visanalytic(::Type{Multifrequency{B}}) where {B} = NotAnalytic()
imanalytic(::Type{Multifrequency{B}}) where {B} = imanalytic(B)

radialextent(::Type{Multifrequency{B}}) where {B} = radialextent(B)

flux(::Type{Multifrequency{B}}) where {B} = flux(B)

intensity_point(M::Multifrequency{B}, p) where {B} = intensity_point(M.base, p)

function intensity_point(M::Multifrequency, p)
    I0 = parent(M.base)
    I_img = applyspectral(I0, M.spec, p.Fr)
    I_model = ContinuousImage(I_img, M.base.kernel)
    return intensity_point(I_model, p)
end

#visibility_point(M::Multifrequency{B},p) where {B} = visibility_point(B)

#"""
#    $(TYPEDEF)
#Taylor expansion spectral model of order n for multifrequency imaging. Applies to Continuous Image models only.
#
#This is the same multifrequency model implemented in ehtim (Chael et al., 2023).
#
# Fields
#$(FIELDS)
#"""
#struct TaylorSpectral{C<:NTuple, ν0<:Real} <: AbstractSpectralModel
#    """
#    A tuple containing the Taylor expansion coefficients.
#    c[1] is the spectral index α, c[2] is the spectral curvature β, etc.
#    The coeffecients can be either constant values, or arrays with dimensions equal to that of the base image I0.
#    """
#    c::C
#    """
#    The expansion reference frequency.
#    """
#    ν0::C
#end

function order(::TaylorSpectral{N}) where {N}
    return N
end

"""
Applies Taylor Series spectral model to image data.

Generates image data at frequency ν with respect to the reference frequency ν0.
"""
function applyspectral(I0::AbstractArray, spec::TaylorSpectral, ν::N) where {N<:Number}
    data = copy(I0)
    applyspectral!(data, spec, ν)
    return data
end

function applyspectral!(I0::AbstractArray,
                        spec::TaylorSpectral{<:Any,<:DataType,
                                             <:Tuple{Vararg{<:AbstractArray}},<:Number,
                                             <:Number}, ν::Number)
    ν0 = spec.freq0
    x = log(ν / ν0) # frequency to evaluate taylor expansion
    c = spec.index

    n = order(spec)
    xlist = ntuple(i -> x^i, n) # creating a tuple to hold the frequency powers

    for i in eachindex(I0) # doing expansion one pixel at a time
        exparg = sum(getindex.(c, i) .* xlist)
        I0[i] = I0[i] * exp(exparg)
    end
    return I0
end

function applyspectral!(I0::AbstractArray,
                        spec::TaylorSpectral{<:Any,<:DataType,<:Tuple{Vararg{<:Number}},
                                             <:Number,<:Number}, ν::Number)
    ν0 = spec.freq0
    x = log(ν / ν0) # frequency to evaluate taylor expansion
    c = spec.index

    n = order(spec)
    xlist = ntuple(i -> x^i, n) # creating a tuple to hold the frequency powers

    for i in eachindex(I0) # doing expansion one pixel at a time
        exparg = sum(c .* xlist)
        I0[i] = I0[i] * exp(exparg)
    end
    return I0
end

"""Given a multifrequency model (base image & spectral model), creates a new Continuous image model at frequency ν."""
function generatemodel(MF::Multifrequency, ν::N) where {N<:Number}
    image = parent(MF.base) # ContinuousImage -> SpatialIntensityMap
    I0 = parent(image) # SpatialIntensityMap -> Array

    data = applyspectral(I0, MF.spec, ν) # base image model, spectral model, frequency to generate new image

    new_intensitymap = IntensityMap(data, getfield(image, :grid), getfield(image, :refdims),
                                    getfield(image, :name))
    return ContinuousImage(new_intensitymap, MF.base.kernel)
end

function visibilitymap_numeric(m::Multifrequency{<:ContinuousImage},
                               grid::AbstractFourierDualDomain)
    checkspatialgrid(axisdims(m.base), grid.imgdomain) # compare base image dimensions to spatial dimensions of data cube
    imgcube = build_imagecube(m, grid.imgdomain)
    vis = applyft(forward_plan(grid), imgcube)
    return applypulse!(vis, m.base.kernel, grid)
end

function checkspatialgrid(imgdims, grid)
    return !(dims(imgdims) == dims(grid)[1:2]) &&
           throw(ArgumentError("The image dimensions in `ContinuousImage`\n" *
                               "and the spatial dimensions of the visibility grid passed to `visibilitymap`\n" *
                               "do not match. This is not currently supported."))
end

"""
Build a multifrequency image cube to hold images at all frequencies.
"""
function build_imagecube(m::Multifrequency, mfgrid::RectiGrid)
    I0 = parent(m.base) # base image IntensityMap 
    spec = m.spec
    νlist = mfgrid.Fr

    imgcube = allocate_imgmap(m.base, mfgrid) # build 3D cube of IntensityMap objects

    for i in eachindex(νlist) # setting the image each frequency to first equal the base image and then apply spectral model
        imgcube[:, :, i] .= I0
        applyspectral!(@view(imgcube[:, :, i]), spec, νlist[i]) # @view modifies existing image cube inplace --- don't create new object
    end

    return imgcube
end

function mfimagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer, νlist::Vector,
                       x0::Real=0, y0::Real=0; executor=Serial(),
                       header=ComradeBase.NoHeader())
    @assert (nx > 0) && (ny > 0) "Number of pixels must be positive"

    psizex = fovx / nx
    psizey = fovy / ny

    xitr = X(LinRange(-fovx / 2 + psizex / 2 - x0, fovx / 2 - psizex / 2 - x0, nx))
    yitr = Y(LinRange(-fovy / 2 + psizey / 2 - y0, fovy / 2 - psizey / 2 - y0, ny))
    νlist = Fr(νlist)
    grid = RectiGrid((X=xitr, Y=yitr, Fr=νlist); executor, header)
    return grid
end
