export ZeroModel

"""
    $(TYPEDEF)

Defines a model that is `empty` that is it return zero for everything.

# Notes
This returns 0 by using `FillArrays` so everything should be non-allocating
"""
struct ZeroModel{T} <: AbstractModel end

ZeroModel() = ZeroModel{Float64}()

visanalytic(::Type{<:ZeroModel}) = IsAnalytic()
imanalytic(::Type{<:ZeroModel}) = IsAnalytic()

visibility_point(::ZeroModel{T}, args...) where {T} = complex(zero(T))
intensity_point(::ZeroModel{T}, args...) where {T} = zero(T)

function visibilitymap_analytic(::ZeroModel{T}, p::AbstractSingleDomain) where {T}
    return Fill(zero(Complex{T}), length(p.U))
end
function intensitymap_analytic(::ZeroModel{T}, p::AbstractSingleDomain) where {T}
    return IntensityMap(Fill(zero(T), map(length, dims(p))), p)
end
function intensitymap_analytic(::ZeroModel{T}, p::AbstractRectiGrid) where {T}
    return IntensityMap(Fill(zero(T), map(length, dims(p))), p)
end

@inline AddModel(::ZeroModel, x) = x
@inline AddModel(x, ::ZeroModel) = x

@inline ConvolvedModel(m::ZeroModel, ::Any) = m
@inline ConvolvedModel(::Any, m::ZeroModel) = m

@inline ModifiedModel(z::ZeroModel, ::NTuple{N, <:ModelModifier}) where {N} = z

# __extract_tangent(::ZeroModel) = ZeroTangent()
