export PolySpectral

@doc """
    PolySpectral(index::Tuple, freq0::Number, p0=zero(freq0); link=log)
    PolySpectral(index::Number, freq0::Number, p0=zero(freq0); link=log)

A frequency-dependent [`DomainParams`](@ref) model that changes a base parameter by a
polynomial expansion in log-frequency, `η = ∑ₙ index[n] * log(Fr / freq0)^n`:

    inverse(link)(link(base) + η) + p0

where `Fr` is the observation frequency and `freq0` is the reference frequency. With the
default `link = log` this is `base * exp(η) + p0`, a power law in frequency for order 1, and
it is computed in that form, so a zero or negative base works.

The base is not stored here: like every `DomainParams`, `PolySpectral` is a transformation
and has no value until it is paired with one by [`MultiDomainParams`](@ref). Use a unit base
for a geometric parameter, an image for a cube, or build the latter with
[`MultiDomainImage`](@ref).

The expansion is scalar: against a polarized base the default link scales all four Stokes
components identically, so with the default `p0` the fractional polarization and EVPA are
frequency-independent. To give a component its own spectrum, give it its own model —
`PolarizedModel(MultiDomainImage(imgI, pulse, psI), ...)`.

# Arguments
- `index`: polynomial coefficients. A `Tuple` of length `N` for an order-`N` expansion, or
  a single `Number` for order-1 (a spectral index). Each coefficient may itself be an
  `AbstractArray` for spatially varying coefficients.
- `freq0`: reference frequency. At `Fr = freq0`, `η = 0` and the value is `base + p0`.
- `p0` (optional): additive offset term, a real value or a field of them. Defaults to a
  zero that does not widen the element type. Against a polarized base it offsets every
  Stokes component alike, so a nonzero `p0` changes the fractional polarization.
- `link` (keyword): a function with an `InverseFunctions.inverse`, such as `log`,
  `identity` or `LogExpFunctions.logit`, that sets how `η` changes the base. A link without
  an inverse throws.

# Examples
```julia
# Geometric modeling: a unit base makes the parameter the spectral factor itself.
σ = MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9))   # spectral index = 1
modify(Gaussian(), Stretch(σ, 1.0))                       # frequency-dependent stretch

# Order-2: index plus curvature.
MultiDomainParams(1.0, PolySpectral((1.0, 0.5), 230.0e9))

# Imaging: the base is the image.
MultiDomainParams(rand(64, 64), PolySpectral(1.5, 230.0e9))
```
"""
struct PolySpectral{E, T, F <: Number, P0, L} <: ComradeBase.DomainParams{E}
    index::T
    freq0::F
    p0::P0
    link::L

    function PolySpectral{E, T, F, P0, L}(index, freq0, p0, link) where {E, T, F, P0, L}
        return new{E, T, F, P0, L}(index, freq0, p0, checklink(link))
    end
end

function PolySpectral(index::Tuple, freq0::Number, p0 = zero(freq0); link = log)
    P = map(paramtype ∘ typeof, index)
    E = promote_type(P..., typeof(freq0), paramtype(typeof(p0)))
    return PolySpectral{E}(index, freq0, p0; link)
end

function PolySpectral{E}(index, freq0, p0 = zero(E); link = log) where {E}
    return PolySpectral{E, typeof(index), typeof(freq0), typeof(p0), typeof(link)}(index, freq0, p0, link)
end

function PolySpectral(index::Number, freq0::Number, p0 = zero(freq0); link = log)
    return PolySpectral((index,), freq0, p0; link)
end

function ComradeBase.paramfield(domain::PolySpectral, p)
    return polyfield(domain.link, log.(domaincoord(p, :Fr, domain) ./ domain.freq0), domain.index)
end

ComradeBase.apply_param(base, domain::PolySpectral, fac, p) = polyapply(domain.link, base, fac, domain.p0)

# The polynomial is scalar, so every Stokes component changes alike.
ComradeBase.stokes(ps::PolySpectral, v) = ps

Base.show(io::IO, ps::PolySpectral) = _showpoly(io, "PolySpectral", ps.index, ps.freq0, ps.p0, ps.link)
