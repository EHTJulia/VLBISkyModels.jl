export Multifrequency, TaylorSpectral, applyspectral, generateimage

"""Abstract type to hold all multifrequency spectral models"""
abstract type AbstractSpectralModel end

"""
    $(TYPEDEF)
Spectral Model object.

# Fields
$(FIELDS)
"""
struct Multifrequency{B, F<:Number, S<:AbstractSpectralModel} <: ComradeBase.AbstractModel
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
struct TaylorSpectral{C<:AbstractArray,N<:Integer} <: AbstractSpectralModel
    """
    Taylor expansion coefficients.
    Can be a vector or an array of coefficients.
    """
    c::C
    """
    Taylor expansion order
    """
    n::N
end

"""
Applies Taylor Series spectral model to an image model.

Generates a new image model at frequency ν with respect to the reference frequency ν0.
"""
function applyspectral(ν, ν0, base::ContinuousImage, spec::TaylorSpectral)
    x = log10(ν/ν0) # frequency to evaluate taylor expansion
    data = getfield(base.img,:data)
    for i in 1:n # loop through the taylor series terms
        data .+= spec.c[i]*(x^i)
    end
    image = IntensityMap(data, getfield(base.img, :grid), getfield(base.img, :refdims), getfield(base.img, :name))
    return ContinuousImage(image,base.kernel)
end

"""Creates a new Multifrequency object containing image at a frequency ν."""
function generateimage(MF::Multifrequency, ν)
    new_base = applyspectral(ν,MF.ν0,MF.base,MF.spec) # base image model, spectral model, frequency to generate new image
    return Multifrequency(new_base,MF.ν0,spec)
end