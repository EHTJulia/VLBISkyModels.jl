export Multifrequency, TaylorSpectral, applyspectral, generateimage

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
function applyspectral(ν::N, ν0::N, I0::AbstractArray, spec::TaylorSpectral) where {N<:Number}
    x = log(ν/ν0) # frequency to evaluate taylor expansion
    c = spec.c

    n = order(spec)
    xlist = ntuple(i -> x^i, n) # creating a tuple to hold the frequency powers

    data = similar(I0)
    for i in eachindex(data) # doing expansion one pixel at a time
        exparg = sum(getindex.(c,i).*xlist)
        data[i] = I0[i] * exp(exparg)
    end
    return data
end

"""Creates a new Multifrequency object containing image at a frequency ν."""
function generateimage(MF::Multifrequency, ν::N) where {N<:Number}
    image = parent(MF.base) # ContinuousImage -> SpatialIntensityMap
    I0 = parent(image) # SpatialIntensityMap -> Array

    data = applyspectral(ν,MF.ν0,I0,MF.spec) # base image model, spectral model, frequency to generate new image
    
    new_base = IntensityMap(data, getfield(image, :grid), getfield(image, :refdims), getfield(image, :name))
    return Multifrequency(new_base,MF.ν0,MF.spec)
end