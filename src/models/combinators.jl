export add, convolved, smoothed, components

"""
    $(TYPEDEF)
Abstract type that denotes a composite model. Where we have
combined two models together.

# Implementation
Any implementation of a composite type must define the following methods:

- visibility_point
- uv_combinator
- imanalytic
- visanalytic
- ComradeBase.intensity_point if model intensity is `IsAnalytic`
- intensitymap_numeric! if model intensity is `NotAnalytic`
- intensitymap_numeric if model intensity is `NotAnalytic`
- flux
- radialextent

In addition there are additional optional methods a person can define if needed:

- intensitymap_analytic! if model intensity is `IsAnalytic`  (optional)
- intensitymap_analytic if model intensity is `IsAnalytic` (optional)
- visibilitymap_analytic if visanalytic is `IsAnalytic` (optional)
- visibilitymap_numeric  if visanalytic is `Not Analytic` (optional)
- visibilitymap_analytic! if visanalytic is `IsAnalytic` (optional)
- visibilitymap_numeric!  if visanalytic is `Not Analytic` (optional)
"""
abstract type CompositeModel{M1,M2} <: AbstractModel end

function Base.show(io::IO, m::T) where {T<:CompositeModel}
    si = split("$(T)", "{")[1]
    println(io, "$(si)(")
    println(io, "model1: ", m.m1)
    println(io, "model2: ", m.m2)
    return print(io, ")")
end

radialextent(m::CompositeModel) = max(radialextent(m.m1), radialextent(m.m2))

@inline visanalytic(::Type{<:CompositeModel{M1,M2}}) where {M1,M2} = visanalytic(M1) *
                                                                     visanalytic(M2)
@inline imanalytic(::Type{<:CompositeModel{M1,M2}}) where {M1,M2} = imanalytic(M1) *
                                                                    imanalytic(M2)
@inline ispolarized(::Type{<:CompositeModel{M1,M2}}) where {M1,M2} = ispolarized(M1) *
                                                                     ispolarized(M2)

"""
    $(TYPEDEF)

Pointwise addition of two models in the image and visibility domain.
An end user should instead call [`added`](@ref added) or `Base.+` when
constructing a model

# Example

```julia-repl
julia> m1 = Disk() + Gaussian()
julia> m2 = added(Disk(), Gaussian()) + Ring()
```
"""
struct AddModel{T1,T2} <: CompositeModel{T1,T2}
    m1::T1
    m2::T2
end

"""
    added(m1::AbstractModel, m2::AbstractModel)

Combine two models to create a composite [`AddModel`](@ref VLBISkyModels.AddModel).
This adds two models pointwise, i.e.

```julia-repl
julia> visibility(added(m1, m2), 1.0, 1.0) == visibility(m1, 1.0, 1.0) + visibility(m2, 1.0, 1.0)
m1 = Gaussian()
```
"""
@inline added(m1::AbstractModel, m2::AbstractModel) = AddModel(m1, m2)

"""
    Base.:+(m1::AbstractModel, m2::AbstractModel)

Combine two models to create a composite [`AddModel`](@ref VLBISkyModels.AddModel).
This adds two models pointwise, i.e.

```julia-repl
julia> visibility(m1 + m2, 1.0, 1.0) == visibility(m1, 1.0, 1.0) + visibility(m2, 1.0, 1.0)
m1 = Gaussian()
```
"""
Base.:+(m1::AbstractModel, m2::AbstractModel) = added(m1, m2)
Base.:-(m1::AbstractModel, m2::AbstractModel) = added(m1, -1.0 * m2)

# struct NModel{V<:AbstractVector, M<:AbstractModel}
#     m::V{M}
# end

# function visibilitymap(m::NModel, u, v)
#     f(x) = visibilitymap(x, u, v)
#     return sum(f, m.m)
# end

# function intensitymap(m::NModel, fov, dims)

# end

"""
    components(m::AbstractModel)

Returns the model components for a composite model. This
will return a Tuple with all the models you have constructed.

# Example

```julia-repl
julia> components(m)
m = Gaussian() + Disk()
```
"""
components(m::AbstractModel) = (m,)
function components(m::CompositeModel{M1,M2}) where
         {M1<:AbstractModel,M2<:AbstractModel}
    return (components(m.m1)..., components(m.m2)...)
end

flux(m::AddModel) = flux(m.m1) + flux(m.m2)

_numeric_add(m1, m2, dims) = intensitymap(m1, dims) + intensitymap(m2, dims)

function intensitymap_numeric(m::AddModel, dims::AbstractSingleDomain)
    return _numeric_add(m.m1, m.m2, dims)
end

function intensitymap_numeric(m::AddModel, dims::AbstractFourierDualDomain)
    return _numeric_add(m.m1, m.m2, dims)
end

function intensitymap_numeric!(sim::IntensityMap, m::AddModel)
    csim = copy(sim)
    intensitymap!(csim, m.m1)
    sim .= csim
    intensitymap!(csim, m.m2)
    sim .= sim .+ csim
    return nothing
end

@inline uv_combinator(::AddModel) = Base.:+
@inline xy_combinator(::AddModel) = Base.:+

# @inline function _visibilitymap(model::CompositeModel{M1,M2}, u, v, t, ν, cache) where {M1,M2}
#     _combinatorvis(visanalytic(M1), visanalytic(M2), uv_combinator(model), model, u, v, t, ν, cache)
# end

# @inline function _visibilitymap(model::M, u::AbstractArray, v::AbstractArray, args...) where {M <: CompositeModel}
#     return _visibilitymap(visanalytic(M), model, u, v, args...)
# end

# TODO the fast thing is to add the intensitymaps together and then FT
# We currently don't handle this case
@inline function visibilitymap_numeric(model::AddModel{M1,M2},
                                       p::AbstractSingleDomain) where {M1,M2}
    return _visibilitymap(visanalytic(M1), model.m1, p) .+
           _visibilitymap(visanalytic(M2), model.m2, p)
end

@inline function visibilitymap_numeric(model::AddModel{M1,M2},
                                       p::AbstractRectiGrid) where {M1,M2}
    return _visibilitymap(visanalytic(M1), model.m1, p) .+
           _visibilitymap(visanalytic(M2), model.m2, p)
end

@inline function visibilitymap_numeric(model::AddModel{M1,M2},
                                       p::AbstractFourierDualDomain) where {M1,M2}
    return _visibilitymap(visanalytic(M1), model.m1, p) .+
           _visibilitymap(visanalytic(M2), model.m2, p)
end

# @inline function _visibilitymap(::IsAnalytic, model::CompositeModel, u::AbstractArray, v::AbstractArray, args...)
#     f = uv_combinator(model)
#     return f.(visibility_point.(Ref(model.m1), u, v), visibility_point.(Ref(model.m2), u, v))
# end

# function __extract_tangent(dm::CompositeModel)
#     m1 = dm.m1
#     m2 = dm.m2
#     tm1 = __extract_tangent(m1)
#     tm2 = __extract_tangent(m2)
#     return Tangent{typeof(dm)}(; m1=tm1, m2=tm2)
# end

# function _visibilitymap(model::AddModel, u::AbstractArray, v::AbstractArray, args...)
#     return visibilitymap(model.m1, u, v) + visibilitymap(model.m2, u, v)
# end

@inline function visibility_point(model::CompositeModel{M1,M2}, p) where {M1,M2}
    f = uv_combinator(model)
    v1 = visibility_point(model.m1, p)
    v2 = visibility_point(model.m2, p)
    return f(v1, v2)
end

@inline function intensity_point(model::CompositeModel, p)
    f = xy_combinator(model)
    v1 = intensity_point(model.m1, p)
    v2 = intensity_point(model.m2, p)
    return f(v1, v2)
end

"""
    $(TYPEDEF)

Pointwise addition of two models in the image and visibility domain.
An end user should instead call [`convolved`](@ref convolved).
Also see [`smoothed(m, σ)`](@ref smoothed) for a simplified function that convolves
a model `m` with a Gaussian with standard deviation `σ`.
"""
struct ConvolvedModel{M1,M2} <: CompositeModel{M1,M2}
    m1::M1
    m2::M2
end

"""
    convolved(m1::AbstractModel, m2::AbstractModel)

Convolve two models to create a composite [`ConvolvedModel`](@ref VLBISkyModels.ConvolvedModel).

```julia-repl
julia> convolved(m1, m2)
m1 = Ring()
```
"""
convolved(m1::AbstractModel, m2::AbstractModel) = ConvolvedModel(m1, m2)

"""
    smoothed(m::AbstractModel, σ::Number)

Smooths a model `m` with a Gaussian kernel with standard deviation `σ`.

# Notes

This uses [`convolved`](@ref) to created the model, i.e.

```julia-repl
julia> convolved(m1, m2) == smoothed(m1, 1.0)
m1 = Disk()
```
"""
smoothed(m, σ::Number) = convolved(m, stretched(Gaussian(), σ, σ))

@inline imanalytic(::Type{<:ConvolvedModel}) = NotAnalytic()

@inline uv_combinator(::ConvolvedModel) = Base.:*

flux(m::ConvolvedModel) = flux(m.m1) * flux(m.m2)

@inline function visibilitymap_numeric(model::ConvolvedModel{M1,M2},
                                       p::AbstractRectiGrid) where {M1,M2}
    return _visibilitymap(visanalytic(M1), model.m1, p) .*
           _visibilitymap(visanalytic(M2), model.m2, p)
end

@inline function visibilitymap_numeric(model::ConvolvedModel{M1,M2},
                                       p::AbstractFourierDualDomain) where {M1,M2}
    return _visibilitymap(visanalytic(M1), model.m1, p) .*
           _visibilitymap(visanalytic(M2), model.m2, p)
end

@inline function visibilitymap_numeric!(vis::IntensityMap,
                                        m::ConvolvedModel{M1,M2}) where {M1,M2}
    cvis = similar(vis)
    visibilitymap!(cvis, m.m1)
    vis .= cvis
    visibilitymap!(cvis, m.m2)
    vis .*= cvis
    return nothing
end

# function intensitymap_numeric(model::ConvolvedModel, dims::ComradeBase.AbstractDomain)
#     (;X, Y) = dims
#     vis1 = visibilitymap(model.m1, dims)
#     vis2 = visibilitymap(model.m2, dims)
#     U = vis1.U
#     V = vis1.V
#     vis = ifftshift(parent(phasedecenter!(vis1.*vis2, X, Y, U, V)))
#     ifft!(vis)
#     return IntensityMap(real.(vis), dims)
# end

# function intensitymap_numeric!(sim::IntensityMap, model::ConvolvedModel)
#     dims = axisdims(sim)
#     (;X, Y) = dims
#     vis1 = fouriermap(model.m1, dims)
#     vis2 = fouriermap(model.m2, dims)
#     U = vis1.U
#     V = vis1.V
#     vis = ifftshift(parent(phasedecenter!((vis1.*vis2), X, Y, U, V)))
#     ifft!(vis)
#     sim .= real.(vis)
# end
