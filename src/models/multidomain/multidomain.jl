import ComradeBase: _showparam

# Coefficients are splatted so array-valued ones fuse into one broadcast.
@inline function polyarg(x, index::Vararg{Any, N}) where {N}
    return reduce(+, ntuple(n -> index[n] * x^n, Val(N)))
end

# Through the tuple: broadcasting a static vector with a traced `p0` fails under Reactant.
@inline addoffset(x::Number, p0) = x + p0
@inline addoffset(x::StaticArray, p0) = similar_type(x)(Tuple(x) .+ p0)

"""
    AbstractLink

How a polynomial family such as [`PolySpectral`](@ref) or [`PolyTemporal`](@ref) combines
its polynomial `η = ∑ₙ index[n] * xⁿ` with the base parameter. A family's value is
`applylink(link, base, η) + p0`.

To define a new link, subtype `AbstractLink` and add a method to [`applylink`](@ref).
"""
abstract type AbstractLink end

"""
    applylink(link::AbstractLink, base, η)

The base parameter `base` changed by the polynomial value `η` according to `link`.
For a link function `g` this is `g⁻¹(g(base) + η)`.
"""
function applylink end

"""
    LogLink()

The log link: `applylink(LogLink(), base, η) = base * exp(η)`. The polynomial gives the log
of a positive factor, so the base never changes sign. This is the default of every
polynomial family.
"""
struct LogLink <: AbstractLink end

"""
    IdentityLink()

The identity link: `applylink(IdentityLink(), base, η) = base + η`, for a parameter that
drifts additively, such as a position. Against a polarized base, `η` is added to every
Stokes component.
"""
struct IdentityLink <: AbstractLink end

applylink(::LogLink, base, η) = base * exp(η)
applylink(::IdentityLink, base, η) = addoffset(base, η)

# A link splits into a part that depends only on `η` and a part that combines it with the
# base, so the first is computed once per plane rather than once per pixel.
linkfield(::AbstractLink, η) = η
linkfield(::LogLink, η) = exp(η)
linkapply(link::AbstractLink, base, f) = applylink(link, base, f)
linkapply(::LogLink, base, f) = base * f

# The link field of the polynomial. Scalar coefficients: materialized once per frequency or
# time. Array coefficients: left lazy to fuse with `apply_param`, since the field is as
# large as the result.
@inline function polyfield(link, x, index::Tuple{Vararg{Number}})
    return ((xi, c...) -> linkfield(link, polyarg(xi, c...))).(x, index...)
end
@inline function polyfield(link, x, index)
    return Base.broadcasted((xi, c...) -> linkfield(link, polyarg(xi, c...)), x, index...)
end

# `linkapply(link, base, field) + p0`, lazily.
@inline function polyapply(link, base, fac, p0)
    return Base.broadcasted(addoffset, Base.broadcasted((b, f) -> linkapply(link, b, f), base, fac), p0)
end

# The coordinate `n` of the point `p` that the family `fam` reads.
@inline function domaincoord(p, n::Symbol, fam)
    hasproperty(p, n) || throw(
        ArgumentError(
            "`$(nameof(typeof(fam)))` reads the `$n` coordinate, but it is evaluated at a point with coordinates $(keys(p)); evaluate it on a grid with a `$n` dim or a domain with a `$n` coordinate"
        )
    )
    return getproperty(p, n)
end

function _showpoly(io::IO, name, index, ref, p0, link)
    print(io, name, "((")
    for (i, c) in enumerate(index)
        i > 1 && print(io, ", ")
        _showparam(io, c)
    end
    # A one-element tuple needs its trailing comma to read back as a tuple.
    length(index) == 1 && print(io, ",")
    print(io, "), ", ref)
    if !iszero(p0)
        print(io, ", ")
        _showparam(io, p0)
    end
    link isa LogLink || print(io, "; link = ", link)
    return print(io, ")")
end

include("poly_spectral.jl")
include("poly_temporal.jl")
export PolySpectral, PolyTemporal, LogLink, IdentityLink
