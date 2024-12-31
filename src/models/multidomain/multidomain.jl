export getparam, @unpack_params

"""
    abstract type DomainParams

Abstract type for multidomain i.e. time, frequency domain models. 
This is to extend existing models that are just definedin the image and 
visibility domain and automatically extend them to time and frequency domain.

The interface is simple and to extend this with your own time and frequency models,
most users will just need to define 

```julia
struct MyDomainParam <: DomainParams end
function build_params(param::MyDomainParam, p)
    ...
end
where `p` is the point where the model will be evaluated at. For an
example see the [`TaylorSpectralModel`](@ref).

For a model parameterized with a `<:DomainParams` the a use should access 
the parameters with [`getparam`](@ref) or the `@unpack_params` macro.
```
"""
abstract type DomainParams end

abstract type FrequencyParams <: DomainParams end
abstract type TimeParams <: DomainParams end

"""
    getparam(m, s::Symbol, p)

Gets the parameter value `s` from the model `m` evaluated at the domain `p`. 
This is similar to getproperty, but allows for the parameter to be a function of the 
domain. Essentially is `m.s <: DomainParams` then `m.s` is evaluated at the parameter `p`.
If `m.s` is not a subtype of `DomainParams` then `m.s` is returned.

!!! warn
    Developers should not typically overload this function and instead
    target [`build_params`](@ref).

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

@inline function build_param(param::Any, p)
    return param
end

include("unpack.jl")
include("freqtaylor.jl")
