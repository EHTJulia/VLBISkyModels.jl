import ComradeBase: _showparam
using InverseFunctions: inverse, NoInverse

# Coefficients are splatted so array-valued ones fuse into one broadcast.
@inline function polyarg(x, index::Vararg{Any, N}) where {N}
    return reduce(+, ntuple(n -> index[n] * x^n, Val(N)))
end

# Through the tuple: broadcasting a static vector with a traced `p0` fails under Reactant.
@inline addoffset(x::Number, p0) = x + p0
@inline addoffset(x::StaticArray, p0) = similar_type(x)(Tuple(x) .+ p0)

# A link `g` gives `g⁻¹(g(base) + η)`, split into a part that depends only on `η`, computed
# once per plane, and a part that combines it with the base. `log` multiplies by `exp(η)`, so
# a base that is zero or negative works and its gradient stays finite.
linkfield(link, η) = η
linkfield(::typeof(log), η) = exp(η)
linkapply(link, base, η) = inverse(link)(link(base) + η)
linkapply(::typeof(log), base, f) = base * f
linkapply(::typeof(identity), base, η) = addoffset(base, η)

function checklink(link)
    inverse(link) isa NoInverse && throw(
        ArgumentError("the link `$link` has no inverse; define `InverseFunctions.inverse` for it")
    )
    return link
end

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
    link === log || print(io, "; link = ", link)
    return print(io, ")")
end

include("poly_spectral.jl")
include("poly_temporal.jl")
export PolySpectral, PolyTemporal
