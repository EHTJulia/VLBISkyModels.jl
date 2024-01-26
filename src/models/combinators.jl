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
- visibilities_analytic if visanalytic is `IsAnalytic` (optional)
- visibilities_numeric  if visanalytic is `Not Analytic` (optional)
- visibilities_analytic! if visanalytic is `IsAnalytic` (optional)
- visibilities_numeric!  if visanalytic is `Not Analytic` (optional)
"""
abstract type CompositeModel{M1,M2} <: AbstractModel end


function Base.show(io::IO, m::T) where {T<:CompositeModel}
    si = split("$(T)", "{")[1]
    println(io, "$(si)(")
    println(io, "model1: ", m.m1)
    println(io, "model2: ", m.m2)
    print(io, ")")
end


function modelimage(::NotAnalytic,
    model::CompositeModel,
    grid::AbstractGrid,
    alg::FourierTransform=FFTAlg(),
    pulse = DeltaPulse(),
    thread::StaticBool = False())

    m1 = @set model.m1 = modelimage(model.m1, grid, alg, pulse, thread)
    @set m1.m2 = modelimage(m1.m2, grid, alg, pulse, thread)
end

function modelimage(::NotAnalytic,
    model::CompositeModel,
    cache::AbstractCache,
    thread::StaticBool = False())

    m1 = @set model.m1 = modelimage(model.m1, cache)
    @set m1.m2 = modelimage(m1.m2, cache)
end

function fouriermap(m::CompositeModel, dims::ComradeBase.AbstractGrid)
    m1 = fouriermap(m.m1, dims)
    m2 = fouriermap(m.m2, dims)
    return uv_combinator(m).(m1,m2)
end



radialextent(m::CompositeModel) = max(radialextent(m.m1), radialextent(m.m2))

@inline visanalytic(::Type{<:CompositeModel{M1,M2}}) where {M1,M2} = visanalytic(M1)*visanalytic(M2)
@inline imanalytic(::Type{<:CompositeModel{M1,M2}}) where {M1,M2} = imanalytic(M1)*imanalytic(M2)
@inline ispolarized(::Type{<:CompositeModel{M1,M2}}) where {M1,M2} = ispolarized(M1)*ispolarized(M2)

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
julia> m1 = Gaussian()
julia> m2 = Disk()
julia> visibility(added(m1,m2), 1.0, 1.0) == visibility(m1, 1.0, 1.0) + visibility(m2, 1.0, 1.0)
true
```
"""
@inline added(m1::AbstractModel, m2::AbstractModel) = AddModel(m1, m2)


"""
    Base.:+(m1::AbstractModel, m2::AbstractModel)

Combine two models to create a composite [`AddModel`](@ref VLBISkyModels.AddModel).
This adds two models pointwise, i.e.

```julia-repl
julia> m1 = Gaussian()
julia> m2 = Disk()
julia> visibility(m1+m2, 1.0, 1.0) == visibility(m1, 1.0, 1.0) + visibility(m2, 1.0, 1.0)
true
```

"""
Base.:+(m1::AbstractModel, m2::AbstractModel) = added(m1, m2)
Base.:-(m1::AbstractModel, m2::AbstractModel) = added(m1, -1.0*m2)



# struct NModel{V<:AbstractVector, M<:AbstractModel}
#     m::V{M}
# end

# function visibilities(m::NModel, u, v)
#     f(x) = visibilities(x, u, v)
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
julia> m = Gaussian() + Disk()
julia> components(m)
(Gaussian{Float64}(), Disk{Float64}())
```
"""
components(m::AbstractModel) = (m,)
components(m::CompositeModel{M1,M2}) where
    {M1<:AbstractModel, M2<:AbstractModel} = (components(m.m1)..., components(m.m2)...)

flux(m::AddModel) = flux(m.m1) + flux(m.m2)


function intensitymap_numeric(m::AddModel, dims::ComradeBase.AbstractGrid)
    sim1 = intensitymap(m.m1, dims)
    sim2 = intensitymap(m.m2, dims)
    return sim1 + sim2
end


function intensitymap_numeric!(sim::IntensityMap, m::AddModel)
    csim = deepcopy(sim)
    intensitymap!(csim, m.m1)
    sim .= csim
    intensitymap!(csim, m.m2)
    sim .= sim .+ csim
    return sim
end

@inline uv_combinator(::AddModel) = Base.:+
@inline xy_combinator(::AddModel) = Base.:+

# @inline function _visibilities(model::CompositeModel{M1,M2}, u, v, t, ν, cache) where {M1,M2}
#     _combinatorvis(visanalytic(M1), visanalytic(M2), uv_combinator(model), model, u, v, t, ν, cache)
# end

# @inline function _visibilities(model::M, u::AbstractArray, v::AbstractArray, args...) where {M <: CompositeModel}
#     return _visibilities(visanalytic(M), model, u, v, args...)
# end


@inline function visibilities_numeric(model::AddModel{M1,M2}, u, v, time, freq) where {M1, M2}
    #f = uv_combinator(model)
    return _visibilities(visanalytic(M1), model.m1, u, v, time, freq) .+
           _visibilities(visanalytic(M2), model.m2, u, v, time, freq)
end

# @inline function _visibilities(::IsAnalytic, model::CompositeModel, u::AbstractArray, v::AbstractArray, args...)
#     f = uv_combinator(model)
#     return f.(visibility_point.(Ref(model.m1), u, v), visibility_point.(Ref(model.m2), u, v))
# end


function __extract_tangent(dm::CompositeModel)
    m1 = dm.m1
    m2 = dm.m2
    tm1 = __extract_tangent(m1)
    tm2 = __extract_tangent(m2)
    return Tangent{typeof(dm)}(m1=tm1, m2=tm2)
end




# function _visibilities(model::AddModel, u::AbstractArray, v::AbstractArray, args...)
#     return visibilities(model.m1, u, v) + visibilities(model.m2, u, v)
# end


@inline function visibility_point(model::CompositeModel{M1,M2}, u, v, time, freq) where {M1,M2}
    f = uv_combinator(model)
    v1 = visibility_point(model.m1, u, v, time, freq)
    v2 = visibility_point(model.m2, u, v, time, freq)
    return f(v1,v2)
end

@inline function intensity_point(model::CompositeModel, p)
    f = xy_combinator(model)
    v1 = intensity_point(model.m1, p)
    v2 = intensity_point(model.m2, p)
    return f(v1,v2)
end



"""
    $(TYPEDEF)

Pointwise addition of two models in the image and visibility domain.
An end user should instead call [`convolved`](@ref convolved).
Also see [`smoothed(m, σ)`](@ref smoothed) for a simplified function that convolves
a model `m` with a Gaussian with standard deviation `σ`.
"""
struct ConvolvedModel{M1, M2} <: CompositeModel{M1,M2}
    m1::M1
    m2::M2
end

"""
    convolved(m1::AbstractModel, m2::AbstractModel)

Convolve two models to create a composite [`ConvolvedModel`](@ref VLBISkyModels.ConvolvedModel).

```julia-repl
julia> m1 = Ring()
julia> m2 = Disk()
julia> convolved(m1, m2)
```
"""
convolved(m1::AbstractModel, m2::AbstractModel) = ConvolvedModel(m1, m2)

"""
    smoothed(m::AbstractModel, σ::Number)
Smooths a model `m` with a Gaussian kernel with standard deviation `σ`.

# Notes
This uses [`convolved`](@ref) to created the model, i.e.
```julia-repl
julia> m1 = Disk()
julia> m2 = Gaussian()
julia> convolved(m1, m2) == smoothed(m1, 1.0)
true
```
"""
smoothed(m, σ::Number) = convolved(m, stretched(Gaussian(), σ, σ))

@inline imanalytic(::Type{<:ConvolvedModel}) = NotAnalytic()


@inline uv_combinator(::ConvolvedModel) = Base.:*

flux(m::ConvolvedModel) = flux(m.m1)*flux(m.m2)

function intensitymap_numeric(model::ConvolvedModel, dims::ComradeBase.AbstractGrid)
    (;X, Y) = dims
    vis1 = fouriermap(model.m1, dims)
    vis2 = fouriermap(model.m2, dims)
    U = vis1.U
    V = vis1.V
    vis = ifftshift(parent(phasedecenter!(vis1.*vis2, X, Y, U, V)))
    ifft!(vis)
    return IntensityMap(real.(vis), dims)
end

function AbstractFFTs.ifft!(vis::AbstractArray{<:StokesParams}, region)
    vI = stokes(vis, :I)
    vQ = stokes(vis, :Q)
    vU = stokes(vis, :U)
    vV = stokes(vis, :V)
    p = plan_ifft!(vI, region)
    p*vI
    p*vQ
    p*vU
    p*vV
    return StructArray{StokesParams{eltype(I)}}((vI, vQ, vU, vV))
end

function intensitymap_numeric!(sim::IntensityMap, model::ConvolvedModel)
    dims = axisdims(sim)
    (;X, Y) = dims
    vis1 = fouriermap(model.m1, dims)
    vis2 = fouriermap(model.m2, dims)
    U = vis1.U
    V = vis1.V
    vis = ifftshift(parent(phasedecenter!((vis1.*vis2), X, Y, U, V)))
    ifft!(vis)
    sim .= real.(vis)
end

@inline function visibilities_numeric(model::ConvolvedModel{M1,M2}, u, v, time, freq) where {M1, M2}
    return _visibilities(visanalytic(M1), model.m1, u, v, time, freq).*
           _visibilities(visanalytic(M2), model.m2, u, v, time, freq)
end



# function _combinatorvis(::IsAnalytic, ::IsAnalytic, f::F, m, u, v, t, ν, cache) where {F}
#     @assert length(u) == length(u) "Number of visibilities must equal number of points"
#     ff(u, v, t, ν) = f(visibility_point(m.m1, u, v, t, ν, cache), visibility_point(m.m2, u, v, t, ν, cache))
#     return mappedarray(ff, u, v, t, ν)
# end

# function _combinatorvis(::IsAnalytic, ::NotAnalytic, f::F, m, u, v, args...) where {F}
#     vis = _visibilities(m.m1, u, v, args...)
#     for i in eachindex(vis)
#         vis[i] = f(visibility_point(m.m2, u[i], v[i], args...), vis[i])
#     end
# end

# function _combinatorvis(::NotAnalytic, ::IsAnalytic, f::F, m, u, v, args...) where {F}
#     vis = _visibilities(m.m2, u, v, args...)
#     for i in eachindex(vis)
#         vis[i] = f(vis[i], visibility_point(m.m1, u[i], v[i], args...))
#     end
# end

# function _combinatorvis(::NotAnalytic, ::NotAnalytic, m, f::F, u, v, args...) where {F}
#     vis1 = _visibilities(m.m1, u, v, args...)
#     vis2 = _visibilities(m.m2, u, v, args...)
#     vis1 .= f.(vis1, vis2)
# end
