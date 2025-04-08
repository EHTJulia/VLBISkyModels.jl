export PolarizedModel, coherencymatrix, PoincareSphere2Map, PolExp2Map, PolExp2Map2,
       stokes_intensitymap,
       SingleStokes

import ComradeBase: AbstractPolarizedModel, m̆, evpa, CoherencyMatrix, StokesParams

# simple check to ensure that the four grids are equal across stokes parameters
function _check_grid(I::IntensityMap, Q::IntensityMap, U::IntensityMap, V::IntensityMap)
    return axisdims(I) == axisdims(Q) == axisdims(U) == axisdims(V)
end

"""
    stokes_intensitymap(I, Q, U, V)

Constructs an `IntensityMap` from four maps for I, Q, U, V.
"""
@inline function stokes_intensitymap(I::IntensityMap, Q::IntensityMap,
                                     U::IntensityMap, V::IntensityMap)
    _check_grid(I, Q, U, V)

    pI = baseimage(I)
    pQ = baseimage(Q)
    pU = baseimage(U)
    pV = baseimage(V)

    simg = StructArray{StokesParams{eltype(pI)}}((I=pI, Q=pQ, U=pU, V=pV))
    return IntensityMap(simg, axisdims(I), refdims(I), name(I))
end

@inline function stokes_intensitymap(I::AbstractArray, Q::AbstractArray,
                                     U::AbstractArray, V::AbstractArray,
                                     grid::AbstractRectiGrid)
    simg = StructArray{StokesParams{eltype(I)}}((; I, Q, U, V))
    return IntensityMap(simg, grid)
end

"""
    $(TYPEDEF)

Wrapped model for a polarized model. This uses the stokes representation of the image.

# Fields
$(FIELDS)
"""
struct PolarizedModel{TI,TQ,TU,TV} <: AbstractPolarizedModel
    """
    Stokes I model
    """
    I::TI
    """
    Stokes Q Model
    """
    Q::TQ
    """
    Stokes U Model
    """
    U::TU
    """
    Stokes V Model
    """
    V::TV
end

@inline radialextent(m::PolarizedModel) = radialextent(stokes(m, :I))
@inline flux(m::PolarizedModel) = StokesParams(flux(m.I), flux(m.Q), flux(m.U), flux(m.V))

function Base.show(io::IO, model::PolarizedModel)
    println(io, "PolarizedModel")
    println(io, "\tI: $(model.I)")
    println(io, "\tQ: $(model.Q)")
    println(io, "\tU: $(model.U)")
    return print(io, "\tV: $(model.V)")
end

Base.@assume_effects :foldable @inline visanalytic(::Type{PolarizedModel{I,Q,U,V}}) where {I,Q,U,V} = visanalytic(I) *
                                                                                                      visanalytic(Q) *
                                                                                                      visanalytic(U) *
                                                                                                      visanalytic(V)
Base.@assume_effects :foldable @inline imanalytic(::Type{PolarizedModel{I,Q,U,V}}) where {I,Q,U,V} = imanalytic(I) *
                                                                                                     imanalytic(Q) *
                                                                                                     imanalytic(U) *
                                                                                                     imanalytic(V)

@inline function intensity_point(pmodel::PolarizedModel, p)
    I = intensity_point(stokes(pmodel, :I), p)
    Q = intensity_point(stokes(pmodel, :Q), p)
    U = intensity_point(stokes(pmodel, :U), p)
    V = intensity_point(stokes(pmodel, :V), p)
    return StokesParams(I, Q, U, V)
end

@inline function visibility_point(pimg::PolarizedModel, p)
    si = visibility_point(stokes(pimg, :I), p)
    sq = visibility_point(stokes(pimg, :Q), p)
    su = visibility_point(stokes(pimg, :U), p)
    sv = visibility_point(stokes(pimg, :V), p)
    return StokesParams(si, sq, su, sv)
end

function visibilitymap_analytic(pimg::PolarizedModel, p::AbstractSingleDomain)
    si = baseimage(visibilitymap_analytic(stokes(pimg, :I), p))
    sq = baseimage(visibilitymap_analytic(stokes(pimg, :Q), p))
    su = baseimage(visibilitymap_analytic(stokes(pimg, :U), p))
    sv = baseimage(visibilitymap_analytic(stokes(pimg, :V), p))
    return StructArray{StokesParams{eltype(si)}}((si, sq, su, sv))
end

# function __extract_tangent(m::PolarizedModel)
#     tmI, tmQ, tmU, tmV = __extract_tangent.(split_stokes(m))
#     return Tangent{typeof(m)}(; I=tmI, Q=tmQ, U=tmU, V=tmV)
# end

function split_stokes(pimg::PolarizedModel)
    return (stokes(pimg, :I), stokes(pimg, :Q), stokes(pimg, :U), stokes(pimg, :V))
end

# If the model is numeric we don't know whether just a component is numeric or all of them are so
# we need to re-dispatch
function visibilitymap_numeric(pimg::PolarizedModel, p::FourierDualDomain)
    mI, mQ, mU, mV = split_stokes(pimg)
    si = _visibilitymap(visanalytic(typeof(mI)), mI, p)
    sq = _visibilitymap(visanalytic(typeof(mQ)), mQ, p)
    su = _visibilitymap(visanalytic(typeof(mU)), mU, p)
    sv = _visibilitymap(visanalytic(typeof(mV)), mV, p)
    return StructArray{StokesParams{eltype(si)}}((si, sq, su, sv))
end

function intensitymap!(pimg::IntensityMap{<:StokesParams},
                       pmodel::PolarizedModel)
    intensitymap!(stokes(pimg, :I), pmodel.I)
    intensitymap!(stokes(pimg, :Q), pmodel.Q)
    intensitymap!(stokes(pimg, :U), pmodel.U)
    intensitymap!(stokes(pimg, :V), pmodel.V)
    return pimg
end

function intensitymap(pmodel::PolarizedModel, dims::AbstractSingleDomain)
    imgI = baseimage(intensitymap(stokes(pmodel, :I), dims))
    imgQ = baseimage(intensitymap(stokes(pmodel, :Q), dims))
    imgU = baseimage(intensitymap(stokes(pmodel, :U), dims))
    imgV = baseimage(intensitymap(stokes(pmodel, :V), dims))
    return create_imgmap(StructArray{StokesParams{eltype(imgI)}}((imgI, imgQ, imgU, imgV)),
                         dims)
end

@inline function convolved(m::PolarizedModel, p::AbstractModel)
    return _convolved(ispolarized(typeof(p)), m::PolarizedModel, p)
end

@inline function _convolved(::IsPolarized, m::PolarizedModel, p)
    return ConvolvedModel(m, p)
end

@inline function _convolved(::NotPolarized, m::PolarizedModel, p)
    return PolarizedModel(convolved(stokes(m, :I), p),
                          convolved(stokes(m, :Q), p),
                          convolved(stokes(m, :U), p),
                          convolved(stokes(m, :V), p))
end

@inline convolved(p::AbstractModel, m::PolarizedModel) = convolved(m, p)
@inline function convolved(p::PolarizedModel, m::PolarizedModel)
    return PolarizedModel(convolved(stokes(p, :I), stokes(m, :I)),
                          convolved(stokes(p, :Q), stokes(m, :Q)),
                          convolved(stokes(p, :U), stokes(m, :U)),
                          convolved(stokes(p, :V), stokes(m, :V)))
end

# @inline function added(m::PolarizedModel, p::AbstractModel)
#     return PolarizedModel(
#                 added(stokes(m, :I), p),
#                 added(stokes(m, :Q), p),
#                 added(stokes(m, :U), p),
#                 added(stokes(m, :V), p),
#                 )
# end

@inline function added(p::PolarizedModel, m::PolarizedModel)
    return PolarizedModel(added(stokes(p, :I), stokes(m, :I)),
                          added(stokes(p, :Q), stokes(m, :Q)),
                          added(stokes(p, :U), stokes(m, :U)),
                          added(stokes(p, :V), stokes(m, :V)))
end

# for m in (:renormed, :rotated, :shifted, :stretched)
#     @eval begin
#       @inline function $m(z::PolarizedModel, arg::Vararg{X,N}) where {X,N}
#             return PolarizedModel(
#                     $m(stokes(z, :I), arg...),
#                     $m(stokes(z, :Q), arg...),
#                     $m(stokes(z, :U), arg...),
#                     $m(stokes(z, :V), arg...),
#             )
#       end
#     end
# end

struct SingleStokes{M,S} <: ComradeBase.AbstractModel
    model::M
    """
       SingleStokes(m::AbstractModel, s::Symbol)
    Takes a model whose trait `ispolarized` returns `IsPolarized` and extracts a SingleStokes
    parameter from it. The `s` parameter is the symbol that represents the Stokes parameter
    e.g., `:I`, `:Q`, `:U`, `:V` for stokes I, Q, U, V respectively.
    """
    function SingleStokes(m, S::Symbol)
        !(S ∈ (:I, :Q, :U, :V)) && throw(ArgumentError("Invalid Stokes parameter $S"))
        M = typeof(m)
        return new{M,S}(m)
    end
end

visanalytic(::Type{<:SingleStokes{M}}) where {M} = visanalytic((M))
imanalytic(::Type{<:SingleStokes{M}}) where {M} = imanalytic((M))
ispolarized(::Type{<:SingleStokes{M}}) where {M} = NotPolarized()

function ComradeBase.intensity_point(m::SingleStokes{M,S}, p) where {M,S}
    return getproperty(intensity_point(m.model, p), S)
end

function ComradeBase.visibility_point(m::SingleStokes{M,S}, p) where {M,S}
    return getproperty(visibility_point(m.model, p), S)
end

radialextent(m::SingleStokes) = radialextent(m.model)
flux(m::SingleStokes{M,S}) where {M,S} = getproperty(flux(m.model), S)

# Need this since rotations can be funky to we should rotate in polarization
function ModifiedModel(m::SingleStokes{M,:Q}, mods::NTuple{N,<:ModelModifier}) where {M,N}
    return SingleStokes(ModifiedModel(m.model, mods), :Q)
end

function ModifiedModel(m::SingleStokes{M,:U}, mods::NTuple{N,<:ModelModifier}) where {M,N}
    return SingleStokes(ModifiedModel(m.model, mods), :U)
end

"""
    PoincareSphere2Map(I, p, X, grid)
    PoincareSphere2Map(I::IntensityMap, p, X)
    PoincareSphere2Map(I, p, X, cache::AbstractCache)

Constructs an polarized intensity map model using the Poincare parameterization.
The arguments are:
  - `I` is a grid of fluxes for each pixel.
  - `p` is a grid of numbers between 0, 1 and the represent the total fractional polarization
  - `X` is a grid, where each element is 3 numbers that represents the point on the Poincare sphere
    that is, X[1,1] is a NTuple{3} such that `||X[1,1]|| == 1`.
  - `grid` is the dimensional grid that gives the pixels locations of the intensity map.

!!! note
    If `I` is an `IntensityMap` then grid is not required since the same grid that was use
    for `I` will be used to construct the polarized intensity map. If a cache is passed instead
    this will return a [`ContinuousImage`](@ref) object.



"""
function PoincareSphere2Map(I, p, X, grid)
    Q = similar(I)
    U = similar(I)
    V = similar(I)

    t1 = X[1]
    t2 = X[2]
    t3 = X[3]

    @inbounds @simd for i in eachindex(I, Q, U, V)
        pI = I[i] * p[i]
        Q[i] = pI * t1[i]
        U[i] = pI * t2[i]
        V[i] = pI * t3[i]
    end

    return stokes_intensitymap(I, Q, U, V, grid)
end

# function PoincareSphere2Map(I, p, X, grid)
#     pimgI = I .* p
#     stokesI = I
#     stokesQ = pimgI .* X[1]
#     stokesU = pimgI .* X[2]
#     stokesV = pimgI .* X[3]
#     return IntensityMap(StructArray{StokesParamsstokesI, stokesQ, stokesU, stokesV, grid)
# end

function PoincareSphere2Map(I::IntensityMap, p, X)
    return PoincareSphere2Map(baseimage(I), p, X, axisdims(I))
end

"""
    PolExp2Map(a, b, c, d, grid::AbstractRectiGrid)

Constructs an polarized intensity map model using the matrix exponential representation from
[Arras 2021 (Thesis)](https://www.philipp-arras.de/assets/dissertation.pdf).

Each Stokes parameter is parameterized as

    I = exp(a)cosh(p)
    Q = exp(a)sinh(p)b/p
    U = exp(a)sinh(p)c/p
    V = exp(a)sinh(p)d/p

where `a,b,c,d` are real numbers with no conditions, and `p=√(a² + b² + c²)`.
"""
@fastmath function PolExp2Map(a::AbstractArray,
                              b::AbstractArray,
                              c::AbstractArray,
                              d::AbstractArray,
                              grid::AbstractRectiGrid)
    pimgI = similar(a)
    pimgQ = similar(b)
    pimgU = similar(c)
    pimgV = similar(d)

    # This is just faster because it is a 1 pass algorithm
    @inbounds for i in eachindex(pimgI, pimgQ, pimgU, pimgV)
        p = sqrt(b[i]^2 + c[i]^2 + d[i]^2)
        ea = exp(a[i])
        tmp = ea * sinh(p) / p
        pimgI[i] = ea * cosh(p)
        pimgQ[i] = tmp * b[i]
        pimgU[i] = tmp * c[i]
        pimgV[i] = tmp * d[i]
    end

    return stokes_intensitymap(pimgI, pimgQ, pimgU, pimgV, grid)
end

@fastmath function PolExp2Map!(a::AbstractArray,
                               b::AbstractArray,
                               c::AbstractArray,
                               d::AbstractArray,
                               grid::AbstractRectiGrid)

    # This is just faster because it is a 1 pass algorithm
    @inbounds for i in eachindex(a, b, c, d)
        p = sqrt(b[i]^2 + c[i]^2 + d[i]^2)
        ip = inv(p)
        ea = exp(a[i])
        ep = exp(p)
        iep = inv(ep)
        sh = (ep - iep) / 2
        ch = (ep + iep) / 2
        tmp = ea * sh * ip
        a[i] = ea * ch
        b[i] = tmp * b[i]
        c[i] = tmp * c[i]
        d[i] = tmp * d[i]
    end

    return stokes_intensitymap(a, b, c, d, grid)
end

"""
    linearpol(pimg::AbstractPolarizedModel, p)

Return the complex linear polarization of the model `m` at point `p`.
"""
function PolarizedTypes.linearpol(pimg::AbstractPolarizedModel, p)
    return linearpol(intensity_point(pimg, p))
end

"""
    mpol(pimg::AbstractPolarizedModel, p)

Return the fractional linear polarization of the model `m` at point `p`.
"""
function PolarizedTypes.mpol(pimg::AbstractPolarizedModel, p)
    return mpol(intensity_point(pimg, p))
end

"""
    polarization(pimg::AbstractPolarizedModel, p)

Return the polarization vector (Q, U, V) of the model `m` at point `p`.
"""
function PolarizedTypes.polarization(pimg::AbstractPolarizedModel, p)
    return polarization(intensity_point(pimg, p))
end

"""
    fracpolarization(pimg::AbstractPolarizedModel, p)

Return the fractional polarization vector (Q/I, U/I, V/I) of the model `m` at point `p`.
"""
function PolarizedTypes.fracpolarization(pimg::AbstractPolarizedModel, p)
    return fracpolarization(intensity_point(pimg, p))
end

"""
    evpa(pimg::AbstractPolarizedModel, p)

electric vector position angle or EVPA of the polarized model `pimg` at `p`
"""
@inline function PolarizedTypes.evpa(pimg::AbstractPolarizedModel, p)
    return evpa(intensity_point(pimg, p))
end

"""
    polellipse(pimg::AbstractPolarizedModel, p)

Compute the polarization of the polarized model.
"""
@inline function PolarizedTypes.polellipse(pimg::AbstractPolarizedModel, p)
    return polellipse(intensity_point(pimg, p))
end

"""
    m̆(pimg::AbstractPolarizedModel, p)
    mbreve(pimg::AbstractPolarizedModel, p)

Computes the fractional linear polarization in the visibility domain

    m̆ = (Q̃ + iŨ)/Ĩ

To create the symbol type `m\\breve` in the REPL or use the
[`mbreve`](@ref) function.
"""
@inline function PolarizedTypes.m̆(pimg::AbstractPolarizedModel, p)
    Q = visibility(stokes(pimg, :Q), p)
    U = visibility(stokes(pimg, :U), p)
    I = visibility(stokes(pimg, :I), p)
    return (Q + 1im * U) / I
end

"""
    $(SIGNATURES)

Explicit m̆ function used for convenience.
"""
PolarizedTypes.mbreve(pimg::AbstractPolarizedModel, p) = m̆(pimg, p)
