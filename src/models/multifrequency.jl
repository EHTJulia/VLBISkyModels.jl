using ComradeBase
using VLBISkyModels
using TaylorSeries

"""Abstract type to hold all multifrequency spectral models"""
abstract type AbstractSpectralModel end

"""
Spectral Model object.
"""
struct Multifrequency{B, F, S<:AbstractSpectralModel} <: ComradeBase.AbstractModel
    """
    base::B: Base image model (e.g. Gaussian, ContinuousImage)
    """
    base::B
    """
    ν0::F: Reference frequency in Hz
    """
    ν0::F
    """
    spec::S: Multifrequency spectral model
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
Taylor expansion spectral model of order n for multifrequency imaging.

This is the same multifrequency model implemented in ehtim (Chael et al., 2023).
"""
struct TaylorSpectral{N<:Integer,C<:AbstractArray,T<:Union{AbstractArray,AbstractSeries}}<: AbstractSpectralModel
    """
    Taylor expansion coefficients.
    Can be a vector or an array of coefficients.
    """
    c::C
    """
    Taylor expansion order
    """
    n::N
    """
    Taylor expansion of order c and with coefficients c
    """
    t::T

end

function TaylorSpectral(c::C,n::N) where {C<:AbstractSeries,N<:Integer}
    t = Taylor1(c,n)
    return TaylorSpectral(c,n,t)
end

function TaylorSpectral(c::C,n::N) where {C<:AbstractArray,N<:Integer}
    t = Taylor1.(c,n) # broacast across array of coefficients
    return TaylorSpectral(c,n,t)
end

"""
Applies Taylor Series spectral model to an image model.

Generates a new image model at frequency ν with respect to the reference frequency ν0.
"""
function applyspectral(ν, ν0, base::ContinuousImage, spec::TaylorSpectral)

    data = spec.t(log10(ν/ν0)) # evaluating the taylor series

    image = IntensityMap(data, getfield(base.img, :grid), getfield(base.img, :refdims), getfield(base.img, :name))
    
    return ContinuousImage(image,base.kernel)
end

"""Makes a new Multifrequency object containing image at a frequency ν."""
function generateimage(MF::Multifrequency, ν)

    new_base = applyspectral(ν,MF.ν0,MF.base,MF.spec) # base image model, spectral model, frequency to generate new image

    return Multifrequency(new_base,MF.ν0,spec)
end