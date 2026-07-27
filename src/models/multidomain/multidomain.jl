export MultiDomainImage, MultiDomainParams, TaylorSpectral, build_param, build_param!
import ComradeBase: DomainParams, allocate_imgmap, build_param
include("poly_spectral.jl")

### general multidomain stuffs ###

# NOTE: the `MultiDomainImage` constructors and the `ContinuousImage{<:MultiDomainParams}`
# methods live in continuous_image.jl. This file is included *before* continuous_image.jl,
# so it must not reference `ContinuousImage` (in a body or a signature) — doing so would be
# an `UndefVarError` at load. Keep this file free of `ContinuousImage`.

@doc """
    MultiDomainParams(base, models...)
    MultiDomainParams(base, models::Tuple)

Pairs a `base` parameter value with one or more `DomainParams` `models` describing how it
varies across the extra (frequency/time) domains.

`base` is the value at the reference domain point: an image array for imaging, or a scalar
for geometric modeling. The `models` (e.g. [`PolySpectral`](@ref)) are applied in order,
each transforming the result of the previous one, so
`MultiDomainParams(base, m1, m2)(p) == build_param(build_param(base, m1, p), m2, p)`.
A `base` may itself be a `MultiDomainParams`, in which case it is evaluated first.

# Example
```julia
# A frequency-dependent image: base image at 230 GHz with spectral index 1.5.
MultiDomainParams(rand(64, 64), PolySpectral(1.5, 230.0e9))

# A frequency-dependent geometric parameter.
modify(Gaussian(), Stretch(MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9))))
```

See [`MultiDomainImage`](@ref) to build a `ContinuousImage` from a base image and models.
"""
struct MultiDomainParams{P, M <: Tuple{Vararg{DomainParams}}} <: DomainParams{P}
    params::P # base model parameters shared by all domains
    models::M  # tuple of domains: contains the domain-specific parameters

    function MultiDomainParams{P, M}(params, models) where {P, M}
        return new{P, M}(params, models)
    end
end

MultiDomainParams(params, domains...) = MultiDomainParams(params, domains)
function MultiDomainParams(params, domains::Tuple)
    return MultiDomainParams{typeof(params), typeof(domains)}(params, domains)
end

(md::MultiDomainParams)(p) = build_param(md, p)

@doc """
    build_param!(buffer, md::MultiDomainParams, p)

In-place form of [`build_param`](@ref): transforms `buffer` through each model in
`md.models` in turn and returns it.

!!! warning "buffer is overwritten and is the seed"
    The **first argument `buffer` is mutated in place** and is the *seed* of the
    transformation. The stored base `md.params` is **not** read by this mutating path —
    the caller is responsible for initializing `buffer` (e.g. with the base). In
    particular, do **not** pass `md.params` itself as `buffer`, or the model's stored
    base will be corrupted. Use the non-mutating `build_param(md, p)` (which seeds from
    `md.params`) when you want a fresh result.

!!! warning "immutable buffers"
    An immutable `buffer` (e.g. a scalar) cannot be updated in place; the transformed
    value is only available as the return value, so always use the returned object.
"""
function build_param!(buffer, md::MultiDomainParams, p)
    # Thread the return value: for an immutable buffer (e.g. a scalar) the per-model
    # build_param! cannot mutate and instead returns the transformed value.
    newbuf = build_param!(buffer, first(md.models), p)
    return build_param!(newbuf, MultiDomainParams(newbuf, Base.tail(md.models)), p)
end

# end the recursive loop
build_param!(buffer, ::MultiDomainParams{P, Tuple{}}, p) where {P} = buffer

function ComradeBase.build_param(md::MultiDomainParams, p)
    # Evaluate the base first: it may itself be a `DomainParams` (e.g. a nested
    # `MultiDomainParams` from chaining `MultiDomainImage` constructors); a plain
    # array or scalar base passes through unchanged.
    seed = ComradeBase.build_param(md.params, p)
    return ComradeBase.build_param(seed, md, p)
end

# The plain array (or scalar) base at the root of a possibly nested chain.
baseparams(md::MultiDomainParams) = baseparams(md.params)
baseparams(x) = x

# The element type and Stokes projection of a chain are those of its root base.
Base.eltype(::Type{<:MultiDomainParams{P}}) where {P} = eltype(P)
# Projecting onto a Stokes component projects the base and keeps the domain models,
# which therefore must act identically across Stokes components (true for scalar
# spectral/temporal models).
ComradeBase.stokes(md::MultiDomainParams, v) = MultiDomainParams(stokes(md.params, v), md.models)

function build_param(params, md::MultiDomainParams, p)
    # The 3-argument `build_param` must not alias its input: the result of the first
    # model is therefore owned here, and the remaining models update it in place rather
    # than allocating a full-sized array per step.
    newparams = build_param(params, first(md.models), p)
    return build_param!(newparams, MultiDomainParams(newparams, Base.tail(md.models)), p)
end

function build_param(params, ::MultiDomainParams{P, Tuple{}}, p) where {P}
    return params
end

"""
    TaylorSpectral(param, index, freq0::Number, p0=zero(param))

Deprecated. `TaylorSpectral` stored the base `param` inside the spectral model; the
base now lives in a [`MultiDomainParams`](@ref) and the expansion in a spectral-only
[`PolySpectral`](@ref). This constructor keeps the old base-first argument order and
returns `MultiDomainParams(param, PolySpectral(index, freq0, p0))`.
"""
function TaylorSpectral(param, index, freq0::Number, p0 = zero(param))
    Base.depwarn(
        "`TaylorSpectral(param, index, freq0, p0)` is deprecated; use " *
            "`MultiDomainParams(param, PolySpectral(index, freq0, p0))`, or " *
            "`PolySpectral(index, freq0, p0)` for the bare spectral factor.",
        :TaylorSpectral,
    )
    return MultiDomainParams(param, PolySpectral(index, freq0, p0))
end
