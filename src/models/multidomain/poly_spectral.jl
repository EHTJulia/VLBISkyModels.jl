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

# Canonical constructor: a tuple of coefficients defines the expansion order.
function PolySpectral(index::Tuple, freq0::Number, p0 = zero(freq0))
    P = map(x -> paramtype(typeof(x)), index)
    E = promote_type(P..., typeof(freq0), paramtype(typeof(p0)))
    return PolySpectral{E}(index, freq0, p0)
end

function PolySpectral{E}(index, freq0, p0 = zero(E)) where {E}
    return PolySpectral{E, typeof(index), typeof(freq0), typeof(p0)}(index, freq0, p0)
end

# A single spectral index is an order-1 expansion.
function PolySpectral(index::Number, freq0::Number, p0 = zero(freq0))
    return PolySpectral((index,), freq0, p0)
end

# Per-element polynomial exponent. The coefficients are splatted into the broadcast rather
# than passed as a tuple, so this scalar kernel fuses over both the frequency axis and
# (spatially varying) array-valued coefficients without materializing a term-sized
# temporary for each order.
@inline function polyarg(x, index::Vararg{Any, N}) where {N}
    return reduce(+, ntuple(n -> index[n] * x^n, Val(N)))
end

# A polarized value is a static vector, so its offset is elementwise: every Stokes component
# takes the same `p0`. Broadcasting inside the kernel is the only way to reach it, since the
# enclosing broadcast has already stepped down to a single parameter value.
@inline addoffset(x::Number, p0) = x + p0
@inline addoffset(x::AbstractArray, p0) = x .+ p0

# The spectral factor. `p.Fr` is a scalar for a single domain point, or reshaped along the
# cube's `Fr` axis when building a cube, so materializing here evaluates the expansion once
# per frequency rather than once per cube point. Spatially varying coefficients make the
# factor as large as the result, with nothing to reuse, so those stay lazy and fuse into the
# broadcast below instead of building a temporary.
@inline specfactor(x, index::Tuple{Vararg{Number}}) = exp.(polyarg.(x, index...))
@inline specfactor(x, index) = Base.broadcasted(exp, Base.broadcasted(polyarg, x, index...))

function ComradeBase.paramfield(domain::PolySpectral, p)
    return specfactor(log.(p.Fr ./ domain.freq0), domain.index)
end

ComradeBase.apply_param(base, domain::PolySpectral, fac, p) = addoffset.(base .* fac, domain.p0)

# The expansion is scalar, so it scales every Stokes component of a polarized base
# identically and survives projection onto a component unchanged.
ComradeBase.stokes(ps::PolySpectral, v) = ps

function restrict_params(ps::PolySpectral, ix, iy)
    return PolySpectral(
        map(x -> restrict_params(x, ix, iy), ps.index), ps.freq0,
        restrict_params(ps.p0, ix, iy)
    )
end

# A field of values prints as a summary: a full 64×64 matrix inline is never what the
# reader wants. A single value prints in full.
_showparam(io::IO, x) = show(io, x)
_showparam(io::IO, x::AbstractArray) = print(io, summary(x))
_showparam(io::IO, x::StaticArray) = show(io, x)

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
