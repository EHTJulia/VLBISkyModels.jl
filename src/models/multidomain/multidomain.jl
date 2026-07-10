export MultiDomainImage, MultiDomainParams, build_param, build_param!
import ComradeBase: imagepixels, NoHeader, DomainParams, allocate_imgmap, build_param
include("poly_spectral.jl")

### general multidomain stuffs ###

# NOTE: the `MultiDomainImage` constructors and the `ContinuousImage{<:MultiDomainParams}`
# methods live in continuous_image.jl. This file is included *before* continuous_image.jl,
# so it must not reference `ContinuousImage` (in a body or a signature) — doing so would be
# an `UndefVarError` at load. Keep this file free of `ContinuousImage`.

struct MultiDomainParams{P, M <: Tuple{Vararg{<:DomainParams}}} <: DomainParams{P}
    params::P # base model parameters shared by all domains
    models::M  # tuple of domains: contains the domain-specific parameters

    function MultiDomainParams(params, domains...) # wrap trailing argument into a tuple
        return new{typeof(params), typeof(domains)}(params, domains)
    end

    function MultiDomainParams(params, domains::Tuple) # already pre-wrapped in a tuple
        return new{typeof(params), typeof(domains)}(params, domains)
    end
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
"""
function build_param!(buffer, md::MultiDomainParams, p)
    build_param!(buffer, first(md.models), p)
    return build_param!(buffer, MultiDomainParams(buffer, Base.tail(md.models)), p)
end

# end the recursive loop
build_param!(buffer, ::MultiDomainParams{P, Tuple{}}, p) where {P} = buffer

# build_param!(md::MultiDomainParams, p) = build_param!(mp.params, md, p)
function ComradeBase.build_param(md::MultiDomainParams, p)
    return ComradeBase.build_param(md.params, md, p)
end

function build_param(params, md::MultiDomainParams, p)
    newparams = build_param(params, first(md.models), p)
    return build_param(newparams, MultiDomainParams(newparams, Base.tail(md.models)), p)
end

function build_param(params, ::MultiDomainParams{P, Tuple{}}, p) where {P}
    return params
end

# Convenience: pair a base value with a spectral model. The base (an image array for
# imaging, or a scalar for geometric modeling) is stored in the `MultiDomainParams`;
# `PolySpectral` itself stays spectral-only.
#
# Note on dispatch: a 3-argument all-`Number` call `PolySpectral(a, b, c)` is the
# spectral-only `(index, freq0, p0)` constructor (more specific, so it wins). A scalar
# base therefore needs the 4-argument form `PolySpectral(base, index, freq0, p0)` (or a
# tuple `index`); an array base is unambiguous in any arity.
function PolySpectral(base::Union{Number, AbstractArray}, index, freq0::Number, p0 = zero(base))
    return MultiDomainParams(base, PolySpectral(index, freq0, p0))
end

@inline _getindices(index::NTuple{N, <:AbstractArray}, i) where {N} = ntuple(n -> index[n][i], Val(N))
@inline _getindices(index::NTuple, i) = index
