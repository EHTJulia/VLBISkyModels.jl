@doc """
    PolyTemporal(index::Tuple, t0::Number, p0=zero(t0); link=log)
    PolyTemporal(index::Number, t0::Number, p0=zero(t0); link=log)

A time-dependent [`DomainParams`](@ref) model that changes a base parameter by a polynomial
expansion in time about the reference epoch `t0`, `η = ∑ₙ index[n] * (Ti - t0)^n`:

    inverse(link)(link(base) + η) + p0

where `Ti` is the time it is evaluated at. With the default `link = log` this is
`base * exp(η) + p0`, which keeps the sign of the base, as a flux needs; it is computed in
that form, so a zero or negative base works. With `link = identity` it is `base + η + p0`, a
drift such as the proper motion of a position. Like [`PolySpectral`](@ref) it has no value until
[`MultiDomainParams`](@ref) pairs it with a base, and it composes with other families in one
chain, e.g. `MultiDomainParams(img, PolyTemporal(0.1, t0), PolySpectral(1.0, ν0))`.

On a [`ContinuousImage`](@ref) the image is built one plane per `Ti` of the grid it is
evaluated on, so the model is evaluated at each plane's `Ti` value (the interval center for a
grid built with [`frames`](@ref)), and every visibility in a plane sees that plane's time.
A geometric model, or an image evaluated at a single point, uses the point's own `Ti`, except
for an image whose own grid has a `Ti` dim, which uses the point's plane.

# Arguments
- `index`: polynomial coefficients, in units of inverse time to the power `n`. A `Tuple` of
  length `N` for an order-`N` expansion, or a single `Number` for a linear rate. Each
  coefficient may itself be an `AbstractArray` for spatially varying coefficients.
- `t0`: reference epoch, in the units of `Ti`. At `Ti = t0`, `η = 0` and the value is
  `base + p0`.
- `p0` (optional): additive offset term, a real value or a field of them. Defaults to a
  zero that does not widen the element type. Against a polarized base it offsets every
  Stokes component alike.
- `link` (keyword): a function with an `InverseFunctions.inverse`, such as `log`,
  `identity` or `LogExpFunctions.logit`, that sets how `η` changes the base. A link without
  an inverse throws.

# Examples
```julia
# A source whose flux decays at a rate of 0.2 per hour from t0 = 1 h.
MultiDomainParams(1.0, PolyTemporal(-0.2, 1.0))

# A component moving 0.5 μas per hour in x from x0 at t0 = 1 h.
MultiDomainParams(x0, PolyTemporal(0.5, 1.0; link = identity))

# An image brightening in time and with a spectral index of -0.5.
MultiDomainParams(rand(64, 64), PolyTemporal(0.1, 1.0), PolySpectral(-0.5, 230.0e9))
```
"""
struct PolyTemporal{E, T, F <: Number, P0, L} <: ComradeBase.DomainParams{E}
    index::T
    t0::F
    p0::P0
    link::L

    function PolyTemporal{E, T, F, P0, L}(index, t0, p0, link) where {E, T, F, P0, L}
        return new{E, T, F, P0, L}(index, t0, p0, checklink(link))
    end
end

function PolyTemporal(index::Tuple, t0::Number, p0 = zero(t0); link = log)
    P = map(paramtype ∘ typeof, index)
    E = promote_type(P..., typeof(t0), paramtype(typeof(p0)))
    return PolyTemporal{E}(index, t0, p0; link)
end

function PolyTemporal{E}(index, t0, p0 = zero(E); link = log) where {E}
    return PolyTemporal{E, typeof(index), typeof(t0), typeof(p0), typeof(link)}(index, t0, p0, link)
end

PolyTemporal(index::Number, t0::Number, p0 = zero(t0); link = log) = PolyTemporal((index,), t0, p0; link)

function ComradeBase.paramfield(domain::PolyTemporal, p)
    return polyfield(domain.link, domaincoord(p, :Ti, domain) .- domain.t0, domain.index)
end

ComradeBase.apply_param(base, domain::PolyTemporal, fac, p) = polyapply(domain.link, base, fac, domain.p0)

# The polynomial is scalar, so every Stokes component changes alike.
ComradeBase.stokes(pt::PolyTemporal, v) = pt

Base.show(io::IO, pt::PolyTemporal) = _showpoly(io, "PolyTemporal", pt.index, pt.t0, pt.p0, pt.link)
