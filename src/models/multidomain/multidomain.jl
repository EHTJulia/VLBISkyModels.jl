export getparam, @unpack_params

"""
    abstract type DomainParams

Abstract type for multidomain i.e. time, frequency domain models. 
This is to extend existing models that are just definedin the image and 
visibility domain and automatically extend them to time and frequency domain.

The interface is simple and to extend this with your own time and frequency models,
most users will just need to define 

```julia
struct MyDomainParam{T} <: DomainParams{T} end
function build_param(param::MyDomainParam{Float64}, p)
    ...
end
where `p` is the point where the model will be evaluated at. For an
example see the [`TaylorSpectralModel`](@ref).

To evaluate the parameter family at a point `p` in the frequency and time 
domain use `build_param(param, p)` or just `param(p)`.

For a model parameterized with a `<:DomainParams` the a use should access 
the parameters with [`getparam`](@ref) or the `@unpack_params` macro.
```
"""
abstract type DomainParams{T} end

abstract type FrequencyParams{T} <: DomainParams{T} end
abstract type TimeParams{T} <: DomainParams{T} end

"""
    paramtype(::Type{<:DomainParams})

Computes the base parameter type of the DomainParams. If `!<:DomainParams` then it just returns
the type. 
"""
@inline paramtype(::Type{<:DomainParams{T}}) where {T} = paramtype(T)
@inline paramtype(T::Type{<:Any}) = T


"""
    getparam(m, s::Symbol, p)

Gets the parameter value `s` from the model `m` evaluated at the domain `p`. 
This is similar to getproperty, but allows for the parameter to be a function of the 
domain. Essentially is `m.s <: DomainParams` then `m.s` is evaluated at the parameter `p`.
If `m.s` is not a subtype of `DomainParams` then `m.s` is returned.

!!! warn
    Developers should not typically overload this function and instead
    target [`build_param`](@ref).

!!! warn
    This feature is experimental and is not considered part of the public stable API.

"""
@inline function getparam(m, s::Symbol, p)
    ps = getproperty(m, s)
    return build_param(ps, p)
end
@inline function getparam(m, ::Val{s}, p) where {s}
    return getparam(m, s, p)
end

"""
    build_param(param::DomainParams, p)

Constucts the parameters for the `param` model at the point `p`
in the (X/U, Y/V, Ti, Fr) domain. This is a required function for
any `<: DomainParams` and must return a number for the specific
parameter at the point `p`.
"""
@inline function build_param(param::Any, p)
    return param
end

@inline function build_param(param::NTuple, p)
    return map(x -> build_param(x, p), param)
end

@inline function build_param(param::AbstractArray, p)
    return map(x -> build_param(x, p), param)
end

@inline function (m::DomainParams)(p)
    return build_param(m, p)
end

include("unpack.jl")
include("freqtaylor.jl")
