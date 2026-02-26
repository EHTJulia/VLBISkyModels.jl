export Gaussian,
    TBlob,
    Disk,
    MRing,
    Crescent,
    ConcordanceCrescent,
    ExtendedRing,
    Ring,
    ParabolicSegment,
    Wisp,
    Butterworth,
    SlashedDisk

# helper functions for below
@inline _getuv(p) = (p.U, p.V)
@inline _getxy(p) = (p.X, p.Y)

"""
$(TYPEDEF)
A type that defines it is a geometric model. These are usually
primitive models, and are usually analytic in Fourier and the image domain.
As a result a user only needs to implement the following methods

- `visibility_point`
- `intensity_point`
- `radialextent`

Note that if the geometric model isn't **analytic** then the usual methods listed
in [`ComradeBase.AbstractModel`](@ref) for non-analytic models need to be implemented.
"""
abstract type GeometricModel{T} <: AbstractModel end
@inline flux(::GeometricModel{T}) where {T} = one(T)

@inline visanalytic(::Type{<:GeometricModel}) = IsAnalytic()
@inline imanalytic(::Type{<:GeometricModel}) = IsAnalytic()

"""
    $(TYPEDEF)

Gaussian with unit standard deviation and flux.

By default if T isn't given, `Gaussian` defaults to `Float64`
"""
struct Gaussian{T} <: GeometricModel{T} end
Gaussian() = Gaussian{Float64}()
radialextent(::Gaussian{T}) where {T} = convert(paramtype(T), 5)
kernel_extent(m::Gaussian{T}) where {T} = radialextent(m)

@inline function intensity_point(::Gaussian{D}, p) where {D}
    x, y = _getxy(p)
    T = paramtype(D)
    return exp(-(x^2 + y^2) / 2) / T(2 * π)
end

@inline function visibility_point(::Gaussian{D}, p) where {D}
    u, v = _getuv(p)
    T = paramtype(D)
    return exp(-2 * T(π)^2 * (u^2 + v^2)) + zero(T)im
end

"""
    $(TYPEDEF)

Constructs a model whose intensity profile is the two dimensional T distriburtion
with degrees of freedom `s`. The normalization is such that the flux is unity.

The intensity profile is given by

 I(r) = Γ((s+2)/2) / (Γ(s/2) * s * π) * (1 + r²/s)^(-(s+2)/2)

where `r` is the radius from the center of the blob. As s → ∞ this approaches the
unit Gaussian.
"""
struct TBlob{T} <: GeometricModel{T}
    slope::T
    norm::T
    function TBlob(slope::Number)
        T = typeof(slope)
        norm = tblobnorm(slope)
        return new{T}(slope, norm)
    end
    function TBlob(slope::DomainParams)
        T = typeof(slope)
        return new{T}(slope, slope)
    end
end
visanalytic(::Type{<:TBlob}) = NotAnalytic()

@inline tblobnorm(s) = gamma((s + 2) / 2) * inv(gamma(s / 2)s * π)
@inline getnorm(m, p) = m.norm
@inline getnorm(s::TBlob{<:DomainParams}, p) = tblobnorm(s.norm(p))
radialextent(m::TBlob) = 5 * m.slope / (m.slope - 2)

function intensity_point(m::TBlob, p)
    x, y = _getxy(p)
    r² = x^2 + y^2
    @unpack_params slope = m(p)
    norm = getnorm(m, p)
    return norm * (1 + r² / slope)^(-(slope + 2) / 2)
end

@doc raw"""
    Disk{T}() where {T}

Tophat disk geometrical model, i.e. the intensity profile
```math
    I(x,y) = \begin{cases} \pi^{-1} & x^2+y^2 < 1 \\ 0 & x^2+y^2 \geq 0 \end{cases}
```
i.e. a unit radius and unit flux disk.

By default if T isn't given, `Disk` defaults to `Float64`
"""
struct Disk{T} <: GeometricModel{T} end
Disk() = Disk{Float64}()

@inline function intensity_point(::Disk{D}, p) where {D}
    x, y = _getxy(p)
    r = x^2 + y^2
    T = paramtype(D)
    return r < 1 ? one(T) / T(π) : zero(T)
end

@inline function visibility_point(::Disk{D}, p) where {D}
    u, v = _getuv(p)
    T = paramtype(D)
    ur = 2 * T(π) * (sqrt(u^2 + v^2) + eps(T))
    return 2 * besselj1(ur) / (ur) + zero(T)im
end

radialextent(::Disk{T}) where {T} = convert(T, 3)

@doc raw"""
    SlashedDisk{T}(slash::T) where {T}

Tophat disk geometrical model, i.e. the intensity profile
```math
    I(x,y) = \begin{cases} \pi^{-1} & x^2+y^2 < 1 \\ 0 & x^2+y^2 \geq 0 \end{cases}
```
i.e. a unit radius and unit flux disk.

By default if T isn't given, `Disk` defaults to `Float64`
"""
struct SlashedDisk{T} <: GeometricModel{T}
    slash::T
end

function intensity_point(m::SlashedDisk{D}, p) where {D}
    x, y = _getxy(p)
    T = paramtype(D)
    r2 = x^2 + y^2
    @unpack_params slash = m(p)
    s = 1 - slash
    norm = 2 / (π * (1 + s))
    ifelse(r2 < 1,
        norm / 2 * ((1 + y) + s * (1 - y)),
        zero(T)
    )
end

function visibility_point(m::SlashedDisk{D}, p) where {D}
    u, v = _getuv(p)
    T = paramtype(D)
    @unpack_params slash = m(p)
    k = 2 * T(π) * sqrt(u^2 + v^2) + eps(T)
    s = 1 - slash
    norm = 2 / (1 + s) / k

    b0outer = besselj0(k)
    b1outer = besselj1(k)
    b2outer = besselj(2, k)

    v1 = (1 + s) * b1outer
    v3 = -2im * T(π) * u * (1 - s) * (b0outer - b2outer - 2 * b1outer / k) / (2 * k)
    return norm * (v1 + v3)
end

radialextent(::SlashedDisk{T}) where {T} = convert(paramtype(T), 3)

"""
    $(TYPEDEF)

A infinitely thin ring model, whose expression in the image domain is
    I(r,θ) = δ(r - 1)/2π
i.e. a unit radius and flux delta ring.

By default if `T` isn't given, `Gaussian` defaults to `Float64`
"""
struct Ring{T} <: GeometricModel{T} end
Ring() = Ring{Float64}()
radialextent(::Ring{T}) where {T} = convert(paramtype(T), 3 / 2)

@inline function intensity_point(::Ring{D}, p) where {D}
    x, y = _getxy(p)
    T = paramtype(D)
    r = hypot(x, y)
    dr = T(1.0e-2)
    return ifelse( (abs(r - 1) < dr / 2),
        one(T) / (2 * T(π) * dr),
        zero(T)
    )
end

@inline function visibility_point(::Ring{D}, p) where {D}
    u, v = _getuv(p)
    T = paramtype(D)
    k = 2 * T(π) * sqrt(u^2 + v^2) + eps(T)
    vis = besselj0(k) + zero(T) * im
    return vis
end

struct Butterworth{N, T} <: GeometricModel{T} end

"""
    Butterworth{N}()
    Butterworth{N, T}()

Construct a model that corresponds to the Butterworth filter of order `N`.
The type of the output is given by `T` and if not given defaults to `Float64`
"""
Butterworth{N}() where {N} = Butterworth{N, Float64}()

radialextent(::Butterworth{N, T}) where {N, T} = convert(T, 5)
flux(::Butterworth{N, T}) where {N, T} = one(T)

visanalytic(::Type{<:Butterworth}) = IsAnalytic()
imanalytic(::Type{<:Butterworth}) = NotAnalytic()

function visibility_point(::Butterworth{N, T}, p) where {N, T}
    u, v = _getuv(p)
    b = hypot(u, v) + eps(T)
    return complex(inv(sqrt(1 + b^(2 * N))))
end

"""
    $(TYPEDEF)
m-ring geometric model. This is a infinitely thin unit flux delta ring
whose angular structure is given by a Fourier expansion. That is,

    I(r,θ) = (2π)⁻¹δ(r-1)∑ₙ(αₙcos(nθ) - βₙsin(nθ))

The `N` in the type defines the order of the Fourier expansion.



# Fields
$(FIELDS)
"""
struct MRing{T, V <: Union{AbstractVector{T}, NTuple}} <: GeometricModel{T}
    """
    Real Fourier mode coefficients
    """
    α::V
    """
    Imaginary Fourier mode coefficients
    """
    β::V
    function MRing(α::V, β::V) where {V <: Union{AbstractVector, NTuple}}
        @argcheck length(α) == length(β) "Lengths of real/imag components must be equal in MRing"
        return new{eltype(α), V}(α, β)
    end
end

"""
    MRing(c::Union{NTuple{N, <:Complex}, AbstractVector{<:Complex}})

Construct an MRing geometric model from a complex vector `c`
that correspond to the real and imaginary (or cos and sin) coefficients
of the Fourier expansion. The `N` in the type defines the order of
the Fourier expansion.
"""
function MRing(c::Union{AbstractVector{<:Complex}, NTuple{N, <:Complex}}) where {N}
    α = real.(c)
    β = imag.(c)
    return MRing(α, β)
end

function MRing(a::Number, b::Number)
    aT, bT = promote(a, b)
    return MRing((aT,), (bT,))
end

# Depreciate this method since we are moving to vectors for simplificty
#@deprecate MRing(a::Tuple, b::Tuple) MRing(a::AbstractVector, b::AbstractVector)

radialextent(::MRing{T}) where {T} = convert(paramtype(T), 3 / 2)

@inline function intensity_point(m::MRing{D}, p) where {D}
    x, y = _getxy(p)
    T = paramtype(D)
    r = hypot(x, y)
    θ = atan(-x, y)
    dr = T(0.02)
    @unpack_params α, β = m(p)
    return @trace if (abs(r - 1) < dr / 2)
        acc = one(T)
        for n in eachindex(α, β)
            s, c = sincos(n * θ)
            acc += 2 * (α[n] * c - β[n] * s)
        end
        acc / (2 * T(π) * dr)
    else
        zero(T)
    end
end

@inline function visibility_point(m::MRing{D}, p) where {D}
    @unpack_params α, β = m(p)
    T = paramtype(D)
    u, v = _getuv(p)
    k = T(2π) * sqrt(u^2 + v^2) + eps(T)
    vis = besselj0(k) + zero(T) * im
    θ = atan(-u, v)
    @inbounds for n in eachindex(α, β)
        s, c = sincos(n * θ)
        vis += 2 * (α[n] * c - β[n] * s) * (one(T)im)^n * besselj(n, k)
    end
    return vis
end

# function _mring_adjoint(α, β, u, v)
#     T = eltype(α)
#     ρ = hypot(u,v)
#     k = 2π*ρ + eps(T)
#     θ = atan(u,v)
#     vis = complex(besselj0(k))

#     j0 = besselj0(k)
#     j1 = besselj1(k)
#     #bj = Base.Fix2(besselj, k)
#     #jn = ntuple(bj, length(α))

#     ∂u = -complex(j1*2π*u/ρ)
#     ∂v = -complex(j1*2π*v/ρ)
#     ∂α = zeros(complex(T), length(α))
#     ∂β = zeros(complex(T), length(α))

#     ∂ku = 2π*u/ρ
#     ∂kv = 2π*v/ρ
#     ∂θu = v/ρ^2
#     ∂θv = -u/ρ^2

#     for n in eachindex(α, β)
#         s,c = sincos(n*θ)
#         imn = (1im)^n
#         jn = besselj(n,k)
#         ∂α[n] =  2*c*jn*imn
#         ∂β[n] = -2*s*jn*imn
#         dJ = j0 - n/k*jn
#         j0 = jn

#         visargc = 2*imn*(-α[n]*s - β[n]*c)
#         visarg  = 2*imn*(α[n]*c - β[n]*s)
#         vis += visarg*jn

#         ∂u +=  n*visargc*jn*∂θu + visarg*dJ*∂ku
#         ∂v +=  n*visargc*jn*∂θv + visarg*dJ*∂kv

#     end
#     return vis, ∂α, ∂β, ∂u, ∂v
# end

# function ChainRulesCore.rrule(::typeof(_mring_vis), m::MRing, u, v)
#     (;α, β) = m
#     pda = ProjectTo(α)
#     pdb = ProjectTo(β)
#     vis, ∂α, ∂β, ∂u, ∂v = _mring_adjoint(α, β, u, v)

#     function _mring_pullback(Δv)
#         return (NoTangent(), Tangent{typeof(m)}(α=pda(real.(Δv'.*∂α)), β=pdb((real.(Δv'.*∂β)))), real.(Δv'.*∂u), real.(Δv'.*∂v))
#     end

#     return vis, _mring_pullback

# end

"""
    $(TYPEDEF)

Creates a [Kamruddin and Dexter](https://academic.oup.com/mnras/article/434/1/765/1005984)
crescent model. This works by composing two disk models together.

# Arguments
- `router`: The radius of the outer disk
- `rinner`: The radius of the inner disk
- `shift`: How much the inner disk radius is shifted (positive is to the right)
- `floor`: The floor of the inner disk 0 means the inner intensity is zero and 1 means it is a large disk.
"""
function Crescent(router::T, rinner::T, shift::T, floor::T) where {T}
    m = stretched(Disk{T}(), router, router) * (T(π) * router^2) +
        T(-1) *
        shifted(
        stretched(Disk{T}(), rinner, rinner) * ((1 - floor) * T(π) * rinner^2),
        shift, zero(typeof(shift))
    )
    return m / flux(m)
end

"""
    $(TYPEDEF)
Creates the ConcordanceCrescent model, i.e. a flat-top crescent
with a displacment and a slash and shadow depth.
Note this creates a crescent with unit flux.
If you want a different flux please use the `renomed`
modifier.

## Fields
$(FIELDS)

## Notes
Unlike the Gaussian and Disk models this does not create the
unit version. In fact, this model could have been created using
the `Disk` and primitives by using VLBISkyModels's model composition
functionality.
"""
struct ConcordanceCrescent{T} <: GeometricModel{T}
    """
    Outer radius of the crescent
    """
    router::T
    """
    Inner radius of the crescent
    (i.e. inside this radius there is a hole)
    """
    rinner::T
    """
    Displacment of the inner disk radius
    """
    shift::T
    """
    Strength of the linear slash. Note that
    s∈[0.0,1.0] to ensure positivity in the image.
    """
    slash::T
end

radialextent(m::ConcordanceCrescent{T}) where {T} = m.router * 3 / 2

# Crescent normalization to ensure the
function _crescentnorm(m::ConcordanceCrescent, p)
    @unpack_params router, rinner, shift, slash = m(p)
    f = (1 + slash) * (router^2 - rinner^2) -
        (1 - slash) * shift * rinner * rinner / router
    return 2 / (π * f)
end

function intensity_point(m::ConcordanceCrescent{D}, p) where {D}
    T = paramtype(D)
    x, y = _getxy(p)
    r2 = x^2 + y^2
    norm = _crescentnorm(m, p)
    @unpack_params router, rinner, shift, slash = m(p)
    ifelse(r2 < router^2 & (x - shift)^2 + y^2 > rinner^2,
        norm / 2 * ((1 + x / router) + slash * (1 - x / router)),
        zero(T)
    )
end

function visibility_point(m::ConcordanceCrescent{D}, p) where {D}
    u, v = _getuv(p)
    T = paramtype(D)
    k = 2 * T(π) * sqrt(u^2 + v^2) + eps(T)
    norm = T(π) * _crescentnorm(m, p) / k
    @unpack_params router, rinner, shift, slash = m(p)
    phaseshift = exp(2 * shift * u * T(π) * 1im)
    b0outer, b0inner = besselj0(k * router), besselj0(k * rinner)
    b1outer, b1inner = besselj1(k * router), besselj1(k * rinner)
    b2outer, b2inner = besselj(2, k * router), besselj(2, k * rinner)

    v1 = (1 + slash) * router * b1outer
    v2 = ((1 + slash) + (1 - slash) * shift / router) *
        phaseshift * rinner * b1inner
    v3 = -2im * T(π) * u * (1 - slash) *
        (
        router * b0outer -
            router * b2outer -
            2 * b1outer / k
    ) / (2 * k)
    v4 = -2im * T(π) * u * (1 - slash) *
        (
        rinner * b0inner -
            rinner * b2inner -
            2 * b1inner / k
    ) / (2 * k) * (rinner / router) * phaseshift
    return norm * (v1 - v2 + v3 - v4)
end

"""
    $(TYPEDEF)
A symmetric extended ring whose radial profile follows an inverse
gamma distributions.

The formula in the image domain is given by

    I(r,θ) = βᵅrᵅ⁻²exp(-β/r)/2πΓ(α)

where `α = shape` and `β = shape+1`

# Note
We mainly use this as an example of a non-analytic Fourier transform
(although it has a complicated expression)

# Fields
$(FIELDS)

Note that if `T` isn't specified at construction then it defaults to `Float64`.
"""
struct ExtendedRing{T} <: GeometricModel{T}
    """shape of the radial distribution"""
    shape::T
end
visanalytic(::Type{<:ExtendedRing}) = NotAnalytic()

radialextent(::ExtendedRing{T}) where {T} = convert(paramtype(T), 6)

@fastmath @inline function intensity_point(m::ExtendedRing, p)
    x, y = _getxy(p)
    @unpack_params shape = m(p)
    T = typeof(shape)
    r = hypot(x, y) + eps(T)
    β = (shape + 1)
    α = shape
    return β^α * r^(-α - 2) * exp(-β / r) / gamma(α) / (2 * T(π))
end

"""
    $(TYPEDEF)

A infinitely thin parabolic segment in the image domain.
The segment is centered at zero, with roots ±1 and a yintercept of 1.

Note that if `T` isn't specified at construction then it defaults to `Float64`.
"""
struct ParabolicSegment{T} <: GeometricModel{T} end
ParabolicSegment() = ParabolicSegment{Float64}()
radialextent(::ParabolicSegment{T}) where {T} = convert(T, sqrt(2) + 1)

"""
    ParabolicSegment(a::Number, h::Number)

A parabolic segment with x-intercepts `±a` and a yintercept of `h`.

# Note

This is just a convenience function for `stretched(ParabolicSegment(), a, h)`
"""
@inline function ParabolicSegment(a::Number, h::Number)
    # Define stretched model from unital model
    return stretched(ParabolicSegment(), a, h)
end

function intensity_point(::ParabolicSegment{D}, p) where {D}
    x, y = _getxy(p)
    yw = (1 - x^2)
    T = paramtype(D)
    return ifelse(abs(y - yw) < T(0.01 / 2) & abs(x) < 1,
        1 / T(2 * 0.01),
        zero(T)
    )
end

function visibility_point(::ParabolicSegment{D}, p) where {D}
    u, v = _getuv(p)
    T = paramtype(D)
    ϵ = sqrt(eps(T))
    vϵ = complex(v + ϵ)
    phase = cispi(T(3) / 4 + 2 * vϵ + u^2 / (2 * vϵ))
    Δ1 = erf(√(T(π) / (2 * vϵ)) * cispi(T(1) / 4) * (u - 2 * vϵ))
    Δ2 = erf(√(T(π) / (2 * vϵ)) * cispi(T(1) / 4) * (u + 2 * vϵ))
    return phase / (√(2 * vϵ)) * (Δ1 - Δ2) / 4
end
