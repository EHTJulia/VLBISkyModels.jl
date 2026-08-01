export MultiDomainParams
import ComradeBase: DomainParams, allocate_imgmap, build_param

"""
    restrict_params(param, ix, iy)

Restrict `param` to the spatial sub-block `(ix, iy)` of an image grid, so that it can be
evaluated over part of an image without materializing the whole of it.

Internal, and not part of the [`DomainParams`](@ref) interface — it is what makes
`intensity_point` on a [`ContinuousImage`](@ref) cost the pulse's support window rather than
the whole image.

Single parameter values pass through unchanged; fields of values over the grid are viewed.
See `ComradeBase.paramtype` for the rule separating the two.

A family carrying data that varies across the image extends this method, rebuilding itself
from restricted components. The fallback returns the family unchanged, which is correct for
one whose parameters do not vary spatially; for one that does, combining its full-sized
field with a restricted base throws a `DimensionMismatch` rather than silently computing
over the wrong pixels.
"""
restrict_params(x, ix, iy) = x
restrict_params(x::AbstractArray, ix, iy) = view(x, ix, iy)
restrict_params(x::StaticArray, ix, iy) = x

# The element type produced by combining two parameter values. A polarized operand makes
# the result polarized: `promote_type` has no rule pairing a number with a `StokesParams`,
# but multiplying or offsetting a polarized value by a scalar stays polarized.
_combineelt(::Type{T}, ::Type{S}) where {T, S} = promote_type(T, S)
_combineelt(::Type{<:StokesParams{T}}, ::Type{S}) where {T, S} = StokesParams{promote_type(T, S)}
_combineelt(::Type{T}, ::Type{<:StokesParams{S}}) where {T, S} = StokesParams{promote_type(T, S)}
function _combineelt(::Type{<:StokesParams{T}}, ::Type{<:StokesParams{S}}) where {T, S}
    return StokesParams{promote_type(T, S)}
end

include("poly_spectral.jl")

### general multidomain stuffs ###

# NOTE: the `MultiDomainImage` constructors and the `ContinuousImage{<:MultiDomainParams}`
# methods live in continuous_image.jl. This file is included *before* continuous_image.jl,
# so it must not reference `ContinuousImage` (in a body or a signature) — doing so would be
# an `UndefVarError` at load. Keep this file free of `ContinuousImage`.

@doc """
    MultiDomainParams(base, models...)
    MultiDomainParams(base, models::Tuple)

Pairs a `base` parameter value with the `DomainParams` `models` describing how it varies
across the extra (frequency and/or time) domains beyond the image domain. "Multi-domain"
refers to those extra domains, not to the number of models: a chain of one model is the
common case.

`base` is the value at the reference domain point: an image array for imaging, or a scalar
for geometric modeling. The `models` (e.g. [`PolySpectral`](@ref)) are applied in order,
each transforming the result of the previous one, so
`MultiDomainParams(base, m1, m2)(p) == apply_param(apply_param(base, m1, p), m2, p)`.
Pairing a model with a base is what gives it a value: a model on its own is a
transformation and cannot be evaluated.

A chain has exactly one base. Chaining onto an existing chain extends its model tuple
rather than nesting, so `MultiDomainParams(MultiDomainParams(b, m1), m2)` and
`MultiDomainParams(b, m1, m2)` are the same model. A chain may not appear among the
`models` of another chain, since its base would have nowhere to go.

# Example
```julia
# A frequency-dependent image: base image at 230 GHz with spectral index 1.5.
MultiDomainParams(rand(64, 64), PolySpectral(1.5, 230.0e9))

# A frequency-dependent geometric parameter.
modify(Gaussian(), Stretch(MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9))))
```

See [`MultiDomainImage`](@ref) to build a `ContinuousImage` from a base image and models.
"""
struct MultiDomainParams{E, P, M <: Tuple{Vararg{DomainParams}}} <: DomainParams{E}
    base::P # the value at the reference domain point, shared by all domains
    models::M # the domain models, applied in order

    function MultiDomainParams{E, P, M}(base, models) where {E, P, M}
        return new{E, P, M}(base, models)
    end
end

MultiDomainParams(base, models...) = MultiDomainParams(base, models)

function MultiDomainParams(base, models::Tuple)
    # A chain among the models would have its base silently dropped; chains flatten instead.
    any(m -> m isa MultiDomainParams, models) && throw(
        ArgumentError(
            "a `MultiDomainParams` cannot be a model in another chain; pass its models " *
                "directly, e.g. `MultiDomainParams(base, m1, m2)`."
        )
    )
    E = _combineelt(
        paramtype(typeof(base)),
        promote_type(map(m -> paramtype(typeof(m)), models)...)
    )
    return MultiDomainParams{E}(base, models)
end

# Chaining onto an existing chain extends the model tuple, so a chain has exactly one base.
function MultiDomainParams(md::MultiDomainParams, models::Tuple)
    return MultiDomainParams(md.base, (md.models..., models...))
end

function MultiDomainParams{E}(base, models::Tuple) where {E}
    return MultiDomainParams{E, typeof(base), typeof(models)}(base, models)
end
MultiDomainParams{E}(base, models...) where {E} = MultiDomainParams{E}(base, models)

(md::MultiDomainParams)(p) = build_param(md, p)

function ComradeBase.build_param(md::MultiDomainParams, p)
    # The single materialization point of a chain: the links compose lazily, so an N-model
    # chain allocates one result rather than one per link. The base is a plain array or
    # scalar, or a chain of its own supplying the reference value.
    return Base.materialize(_applymodels(ComradeBase.build_param(md.base, p), md.models, p))
end

# Apply each model in turn, each transforming the result of the previous one. A model's
# domain-only field is built here, once, rather than inside the element-wise broadcast.
# Nothing else is materialized: a link may hand back a lazy broadcast for the caller to
# realize.
_applymodels(base, ::Tuple{}, p) = base
function _applymodels(base, models::Tuple, p)
    m = first(models)
    return _applymodels(apply_param(base, m, paramfield(m, p), p), Base.tail(models), p)
end

# Projecting a chain onto a Stokes component projects the base and every model, so a model
# carrying polarized data (such as a `StokesParams` offset) projects with it. A
# `DomainParams` used in a polarized chain must define this method.
function ComradeBase.stokes(md::MultiDomainParams, v)
    return MultiDomainParams(stokes(md.base, v), map(m -> stokes(m, v), md.models))
end

function restrict_params(md::MultiDomainParams, ix, iy)
    return MultiDomainParams(
        restrict_params(md.base, ix, iy),
        map(m -> restrict_params(m, ix, iy), md.models)
    )
end

function Base.show(io::IO, md::MultiDomainParams)
    print(io, "MultiDomainParams(")
    _showparam(io, md.base)
    for m in md.models
        print(io, ", ")
        show(io, m)
    end
    return print(io, ")")
end
