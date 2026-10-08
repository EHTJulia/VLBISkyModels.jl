export PolySpectral

@doc """
    PolySpectral(index::Tuple, freq0::Number, p0=zero(freq0))
    PolySpectral(index::Number, freq0::Number, p0=zero(freq0))

A frequency-dependent [`DomainParams`](@ref) model that scales a base parameter by a
polynomial expansion in log-frequency:

    base * exp(∑ₙ index[n] * log(Fr / freq0)^n) + p0

where `Fr` is the observation frequency and `freq0` is the reference frequency.

The base is not stored here: like every `DomainParams`, `PolySpectral` is a transformation
and has no value until it is paired with one by [`MultiDomainParams`](@ref). Use a unit base
for a geometric parameter, an image for a cube, or build the latter with
[`MultiDomainImage`](@ref).

The expansion is scalar: against a polarized base it scales all four Stokes components
identically, so with the default `p0` the fractional polarization and EVPA are
frequency-independent. To give a component its own spectrum, give it its own model —
`PolarizedModel(MultiDomainImage(imgI, pulse, psI), ...)`.

# Arguments
- `index`: polynomial coefficients. A `Tuple` of length `N` for an order-`N` expansion, or
  a single `Number` for order-1 (a spectral index). Each coefficient may itself be an
  `AbstractArray` for spatially varying coefficients.
- `freq0`: reference frequency. At `Fr = freq0` the spectral factor is `1 + p0`.
- `p0` (optional): additive offset term, a real value or a field of them. Defaults to a
  zero that does not widen the element type. Against a polarized base it offsets every
  Stokes component alike, so a nonzero `p0` changes the fractional polarization.

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
struct PolySpectral{E, T, F <: Number, P0} <: ComradeBase.DomainParams{E}
    index::T
    freq0::F
    p0::P0

    function PolySpectral{E, T, F, P0}(index, freq0, p0) where {E, T, F, P0}
        return new{E, T, F, P0}(index, freq0, p0)
    end
end

function PolySpectral(index::Tuple, freq0::Number, p0 = zero(freq0))
    P = map(paramtype ∘ typeof, index)
    E = promote_type(P..., typeof(freq0), paramtype(typeof(p0)))
    return PolySpectral{E}(index, freq0, p0)
end

function PolySpectral{E}(index, freq0, p0 = zero(E)) where {E}
    return PolySpectral{E, typeof(index), typeof(freq0), typeof(p0)}(index, freq0, p0)
end

function PolySpectral(index::Number, freq0::Number, p0 = zero(freq0))
    return PolySpectral((index,), freq0, p0)
end

# Coefficients are splatted so array-valued ones fuse into one broadcast.
@inline function polyarg(x, index::Vararg{Any, N}) where {N}
    return reduce(+, ntuple(n -> index[n] * x^n, Val(N)))
end

# Through the tuple: broadcasting a static vector with a traced `p0` fails under Reactant.
@inline addoffset(x::Number, p0) = x + p0
@inline addoffset(x::StaticArray, p0) = similar_type(x)(Tuple(x) .+ p0)

# Scalar coefficients: materialized once per frequency. Array coefficients: left lazy to fuse
# with `apply_param`, since the factor is as large as the result.
@inline specfactor(x, index::Tuple{Vararg{Number}}) = exp.(polyarg.(x, index...))
@inline specfactor(x, index) = Base.broadcasted(exp, Base.broadcasted(polyarg, x, index...))

function ComradeBase.paramfield(domain::PolySpectral, p)
    return specfactor(log.(p.Fr ./ domain.freq0), domain.index)
end

ComradeBase.apply_param(base, domain::PolySpectral, fac, p) = addoffset.(base .* fac, domain.p0)

# The factor is scalar, so every Stokes component scales alike.
ComradeBase.stokes(ps::PolySpectral, v) = ps

function restrict_params(ps::PolySpectral, ix, iy)
    return PolySpectral(
        map(RestrictTo(ix, iy), ps.index), ps.freq0,
        restrict_params(ps.p0, ix, iy)
    )
end

function Base.show(io::IO, ps::PolySpectral)
    print(io, "PolySpectral((")
    for (i, c) in enumerate(ps.index)
        i > 1 && print(io, ", ")
        _showparam(io, c)
    end
    # A one-element tuple needs its trailing comma to read back as a tuple.
    length(ps.index) == 1 && print(io, ",")
    print(io, "), ", ps.freq0)
    if !iszero(ps.p0)
        print(io, ", ")
        _showparam(io, ps.p0)
    end
    return print(io, ")")
end
