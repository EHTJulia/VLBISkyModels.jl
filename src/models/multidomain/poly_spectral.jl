export PolySpectral
import ComradeBase: build_param

@doc """
    PolySpectral(index::NTuple{N}, freq0::Number, p0=0.0)
    PolySpectral(index::Number, freq0::Number, p0=0.0)

A frequency-dependent [`DomainParams`](@ref) model that scales a (separately supplied)
base parameter by a polynomial expansion in log-frequency:

    base * exp(∑ₙ index[n] * log(Fr / freq0)^n) + p0

where `Fr` is the observation frequency and `freq0` is the reference frequency. The
base value is *not* stored in `PolySpectral`; it is supplied externally — by
[`MultiDomainParams`](@ref) for image cubes, or by a modifier/`build_param(base, …)`
for geometric models. Evaluated on its own (`build_param(ps, p)` / `ps(p)`),
`PolySpectral` returns the spectral factor `exp(arg) + p0` (i.e. base `1`).

# Arguments
- `index`: polynomial coefficients. An `NTuple{N}` for an order-`N` expansion, or a
  single `Number` for order-1 (a spectral index). Each coefficient may itself be an
  `AbstractArray` for spatially varying coefficients.
- `freq0`: reference frequency. At `Fr = freq0` the spectral factor is `1 + p0`.
- `p0` (optional): additive offset term. Defaults to `0`.

# Examples
```julia
# Geometric modeling: the modifier supplies the base value.
ps = PolySpectral(1.0, 230.0e9)         # spectral index = 1
modify(Gaussian(), Stretch(ps, 1.0))     # frequency-dependent stretch
ps = PolySpectral((1.0, 0.5), 230.0e9)  # order-2: index + curvature

# Imaging: pair a base image with the spectral model explicitly, e.g.
# MultiDomainParams(base_image, PolySpectral(1.5, 230.0e9)) or `MultiDomainImage`.
```

The deprecated [`TaylorSpectral`](@ref) constructor accepts the old base-first argument
order and returns a `MultiDomainParams`.
"""
# The trailing parameter `E` is the element type of the spectral factor the model
# produces; through `DomainParams{E}` it feeds `paramtype`/`eltype`/`ispolarized` when a
# bare `PolySpectral` is used as the `params` of a `ContinuousImage`.
struct PolySpectral{N, T <: NTuple{N}, F <: Number, P0, E} <: ComradeBase.FrequencyParams{E}
    index::T
    freq0::F
    p0::P0

    function PolySpectral{N, T, F, P0, E}(index, freq0, p0) where {N, T, F, P0, E}
        return new{N, T, F, P0, E}(index, freq0, p0)
    end
end

# Canonical constructor: a tuple of indices defines the expansion order.
function PolySpectral(index::NTuple{N}, freq0::Number, p0 = 0.0) where {N}
    E = promote_type(
        map(x -> eltype(typeof(x)), index)..., typeof(freq0),
        eltype(typeof(p0))
    )
    return PolySpectral{N, typeof(index), typeof(freq0), typeof(p0), E}(index, freq0, p0)
end

# a single spectral index is an order-1 expansion. Array-valued coefficients must be
# tuple-wrapped (`PolySpectral((indmap,), freq0)`): an untupled non-`Number` first
# argument is more likely a misplaced base value (bases pair with the spectral model
# via `MultiDomainParams`), so it gets a MethodError instead of a guess.
PolySpectral(index::Number, freq0::Number, p0 = 0.0) = PolySpectral((index,), freq0, p0)

# Log-frequency of the evaluation point: a scalar for a single domain point, or a small
# array reshaped along the cube's `Fr` axis for broadcasted cube construction.
_logfr(domain::PolySpectral, p) = log.(p.Fr ./ domain.freq0)

# Per-element polynomial exponent. Broadcasting this scalar kernel over `x` and the
# (scalar or spatially varying array) coefficients fuses the whole expansion into the
# enclosing broadcast, so no term-sized temporaries are materialized.
@inline function _polyarg(x, index::Vararg{Any, N}) where {N}
    return reduce(+, ntuple(n -> index[n] * x^n, Val(N)))
end

# `p0` is an additive offset in the space of the base parameter. A polarized
# (`StokesParams`) base needs a `StokesParams` offset; the scalar-zero default is the
# universal additive identity, so it is accepted for any base.
@inline _addp0(x, p0) = x + p0
@inline function _addp0(x::StokesParams, p0::Real)
    iszero(p0) && return x
    throw(
        ArgumentError(
            "A scalar nonzero `p0` cannot offset a polarized (StokesParams) base; " *
                "pass `p0` as a `StokesParams` (e.g. `zero(StokesParams{Float64})`)."
        )
    )
end

# A `StokesParams` p0 is an `AbstractVector`, so protect it from being broadcast over
# componentwise; any other p0 (scalar or spatial offset array) broadcasts as usual.
@inline _bcastp0(p0) = p0
@inline _bcastp0(p0::StokesParams) = Ref(p0)

# Evaluated on its own (no base): return the spectral factor (base == 1).
function ComradeBase.build_param(domain::PolySpectral, p)
    x = _logfr(domain, p)
    return _addp0.(exp.(_polyarg.(x, domain.index...)), _bcastp0(domain.p0))
end

# Base supplied externally (image cube via `MultiDomainParams`, or a scalar/array base).
function ComradeBase.build_param(params, domain::PolySpectral, p)
    x = _logfr(domain, p)
    return _addp0.(params .* exp.(_polyarg.(x, domain.index...)), _bcastp0(domain.p0))
end

### MUTATING VERSION (used by the recursive `MultiDomainParams` path) ###

function build_param!(param::AbstractArray, domain::PolySpectral, p)
    x = _logfr(domain, p)
    param .= _addp0.(param .* exp.(_polyarg.(x, domain.index...)), _bcastp0(domain.p0))
    return param
end

function build_param!(param::Number, domain::PolySpectral, p)
    x = _logfr(domain, p)
    return _addp0.(param .* exp.(_polyarg.(x, domain.index...)), _bcastp0(domain.p0))
end
