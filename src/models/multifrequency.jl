export Multifrequency, TaylorSpectral, applyspectral, applyspectral!, generatemodel, visibilitymap_numeric, build_imagecube

"""Abstract type to hold all multifrequency spectral models"""
abstract type AbstractSpectralModel end

"""
    $(TYPEDEF)
Spectral Model object.

# Fields
$(FIELDS)
"""
struct Multifrequency{B<:ComradeBase.AbstractModel, F<:Number, S<:AbstractSpectralModel} <: ComradeBase.AbstractModel
    """
    Base image model (e.g. Gaussian, ContinuousImage)
    """
    base::B
    """
    Reference frequency in Hz
    """
    ν0::F
    """
    Multifrequency spectral model
    """
    spec::S
end

# defining the mandatory methods for a Comrade AbstractModel
visanalytic(::Type{Multifrequency{B}}) where {B} = visanalytic(B)
imanalytic(::Type{Multifrequency{B}}) where {B} = imanalytic(B)

radialextent(::Type{Multifrequency{B}}) where {B} = radialextent(B)

flux(::Type{Multifrequency{B}}) where {B} = flux(B)

intensity_point(::Type{Multifrequency{B}}) where {B} = intensity_point(B)
visibility_point(::Type{Multifrequency{B}}) where {B} = visibility_point(B)


"""
    $(TYPEDEF)
Taylor expansion spectral model of order n for multifrequency imaging.

This is the same multifrequency model implemented in ehtim (Chael et al., 2023).

# Fields
$(FIELDS)
"""
struct TaylorSpectral{C<:NTuple} <: AbstractSpectralModel
    """
    Taylor expansion coefficients.
    Tuple of coefficients.
    """
    c::C
end

function order(::TaylorSpectral{<:NTuple{N}}) where N
    return N
end

"""
Applies Taylor Series spectral model to image data.

Generates image data at frequency ν with respect to the reference frequency ν0.
"""
function applyspectral(I0::AbstractArray, spec::TaylorSpectral, ν::N, ν0::N) where {N<:Number}
    data = copy(I0)
    applyspectral!(data, spec, ν, ν0)
    return data
end


function applyspectral!(I0::AbstractArray, spec::TaylorSpectral, ν::N, ν0::N) where {N<:Number}
    x = log(ν/ν0) # frequency to evaluate taylor expansion
    c = spec.c

    n = order(spec)
    xlist = ntuple(i -> x^i, n) # creating a tuple to hold the frequency powers

    for i in eachindex(I0) # doing expansion one pixel at a time
        exparg = sum(getindex.(c,i).*xlist)
        I0[i] = I0[i] * exp(exparg)
    end
    return I0
end

"""Creates a new Multifrequency object containing image at a frequency ν."""
function generatemodel(MF::Multifrequency, ν::N) where {N<:Number}
    image = parent(MF.base) # ContinuousImage -> SpatialIntensityMap
    I0 = parent(image) # SpatialIntensityMap -> Array
    
    data = applyspectral(I0,MF.spec,ν,MF.ν0) # base image model, spectral model, frequency to generate new image
    
    new_intensitymap = IntensityMap(data, getfield(image, :grid), getfield(image, :refdims), getfield(image, :name))
    new_base = ContinuousImage(new_intensitymap,MF.base.kernel)
    return Multifrequency(new_base,MF.ν0,MF.spec)
end

function visibilitymap_numeric(m::Multifrequency{<:ContinuousImage}, grid::AbstractFourierDualDomain)
    checkspatialgrid(axisdims(m.base), grid.imgdomain) # compare base image dimensions to spatial dimensions of data cube
    #@info display(grid)
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
function build_imagecube(m, mfgrid)
    I0 = parent(m.base) # base image IntensityMap 
    ν0 = m.ν0
    spec = m.spec
    νlist = mfgrid.Fr 

    imgcube = allocate_imgmap(m.base, mfgrid) # build 3D cube of IntensityMap objects

    for i in eachindex(νlist) # setting the image each frequency to first equal the base image and then apply spectral model
        imgcube[:,:,i] .= I0 
        applyspectral!(@view(imgcube[:,:,i]), spec, νlist[i], ν0) # @view modifies existing image cube inplace --- don't create new object
    end

    return imgcube
end
