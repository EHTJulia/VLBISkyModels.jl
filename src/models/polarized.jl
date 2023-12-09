export PolarizedModel, coherencymatrix, PoincareSphere2Map

import ComradeBase: AbstractPolarizedModel, m̆, evpa, CoherencyMatrix, StokesParams

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

function Base.show(io::IO, model::PolarizedModel)
    println(io, "PolarizedModel")
    println(io, "\tI: $(summary(model.I))")
    println(io, "\tQ: $(summary(model.Q))")
    println(io, "\tU: $(summary(model.U))")
    print(io, "\tV: $(summary(model.V))")
end

Base.@constprop :aggressive @inline visanalytic(::Type{PolarizedModel{I,Q,U,V}}) where {I,Q,U,V} = visanalytic(I)*visanalytic(Q)*visanalytic(U)*visanalytic(V)
Base.@constprop :aggressive @inline imanalytic(::Type{PolarizedModel{I,Q,U,V}}) where {I,Q,U,V} = imanalytic(I)*imanalytic(Q)*imanalytic(U)*imanalytic(V)

@inline function intensity_point(pmodel::PolarizedModel, p)
    I = intensity_point(stokes(pmodel, :I), p)
    Q = intensity_point(stokes(pmodel, :Q), p)
    U = intensity_point(stokes(pmodel, :U), p)
    V = intensity_point(stokes(pmodel, :V), p)
    return StokesParams(I,Q,U,V)
end


@inline function visibility_point(pimg::PolarizedModel, u, v, time, freq)
    si = visibility_point(stokes(pimg, :I), u, v, time, freq)
    sq = visibility_point(stokes(pimg, :Q), u, v, time, freq)
    su = visibility_point(stokes(pimg, :U), u, v, time, freq)
    sv = visibility_point(stokes(pimg, :V), u, v, time, freq)
    return StokesParams(si, sq, su, sv)
end

function visibilities_analytic(pimg::PolarizedModel, u, v, t, f)
    si = visibilities_analytic(stokes(pimg, :I), u, v, t, f)
    sq = visibilities_analytic(stokes(pimg, :Q), u, v, t, f)
    su = visibilities_analytic(stokes(pimg, :U), u, v, t, f)
    sv = visibilities_analytic(stokes(pimg, :V), u, v, t, f)
    return StructArray{StokesParams{eltype(si)}}((si, sq, su, sv))
end

function __extract_tangent(m::PolarizedModel)
    tmI, tmQ, tmU, tmV = __extract_tangent.(split_stokes(m))
    return Tangent{typeof(m)}(I=tmI, Q=tmQ, U=tmU, V=tmV)
end

split_stokes(pimg::PolarizedModel) = (stokes(pimg, :I), stokes(pimg, :Q), stokes(pimg, :U), stokes(pimg, :V))

# If the model is numeric we don't know whether just a component is numeric or all of them are so
# we need to re-dispatch
function visibilities_numeric(pimg::PolarizedModel, u, v, t, f)
    mI, mQ, mU, mV = split_stokes(pimg)
    si = _visibilities(visanalytic(typeof(mI)), mI, u, v, t, f)
    sq = _visibilities(visanalytic(typeof(mQ)), mQ, u, v, t, f)
    su = _visibilities(visanalytic(typeof(mU)), mU, u, v, t, f)
    sv = _visibilities(visanalytic(typeof(mV)), mV, u, v, t, f)
    return StructArray{StokesParams{eltype(si)}}((si, sq, su, sv))
end

function intensitymap!(pimg::Union{StokesIntensityMap, IntensityMap{<:StokesParams}}, pmodel::PolarizedModel)
    intensitymap!(stokes(pimg, :I), pmodel.I)
    intensitymap!(stokes(pimg, :Q), pmodel.Q)
    intensitymap!(stokes(pimg, :U), pmodel.U)
    intensitymap!(stokes(pimg, :V), pmodel.V)
    return pimg
end

function intensitymap(pmodel::PolarizedModel, dims::AbstractDims)
    imgI = baseimage(intensitymap(stokes(pmodel, :I), dims))
    imgQ = baseimage(intensitymap(stokes(pmodel, :Q), dims))
    imgU = baseimage(intensitymap(stokes(pmodel, :U), dims))
    imgV = baseimage(intensitymap(stokes(pmodel, :V), dims))
    return IntensityMap(StructArray{StokesParams{eltype(imgI)}}((imgI, imgQ, imgU, imgV)), dims)
end

@inline function convolved(m::PolarizedModel, p::AbstractModel)
    return _convolved(ispolarized(typeof(p)), m::PolarizedModel, p)
end

@inline function _convolved(::IsPolarized, m::PolarizedModel, p)
    return ConvolvedModel(m, p)
end

@inline function _convolved(::NotPolarized, m::PolarizedModel, p)
    return PolarizedModel(
            convolved(stokes(m, :I), p),
            convolved(stokes(m, :Q), p),
            convolved(stokes(m, :U), p),
            convolved(stokes(m, :V), p),
        )
end

@inline convolved(p::AbstractModel, m::PolarizedModel) = convolved(m, p)
@inline function convolved(p::PolarizedModel, m::PolarizedModel)
    return PolarizedModel(
            convolved(stokes(p, :I), stokes(m, :I)),
            convolved(stokes(p, :Q), stokes(m, :Q)),
            convolved(stokes(p, :U), stokes(m, :U)),
            convolved(stokes(p, :V), stokes(m, :V)),
        )
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
    return PolarizedModel(
            added(stokes(p, :I), stokes(m, :I)),
            added(stokes(p, :Q), stokes(m, :Q)),
            added(stokes(p, :U), stokes(m, :U)),
            added(stokes(p, :V), stokes(m, :V)),
        )
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

function modelimage(model::PolarizedModel, grid::AbstractDims, alg::FourierTransform=FFTAlg(), pulse=DeltaPulse(), thread::Bool=false)
    return PolarizedModel(
        modelimage(stokes(model, :I), grid, alg, pulse, thread),
        modelimage(stokes(model, :Q), grid, alg, pulse, thread),
        modelimage(stokes(model, :U), grid, alg, pulse, thread),
        modelimage(stokes(model, :V), grid, alg, pulse, thread)
        )
end



"""
    PoincareSphere2Map(I, p, X, grid)
    PoincareSphere2Map(I::IntensityMap, p, X)

Constructs an polarized intensity map model using the Poincare parameterization.
The arguments are:
  - `I` is a grid of fluxes for each pixel.
  - `p` is a grid of numbers between 0, 1 and the represent the total fractional polarization
  - `X` is a grid, where each element is 3 numbers that represents the point on the Poincare sphere
    that is, X[1,1] is a NTuple{3} such that `||X[1,1]|| == 1`.
  - `grid` is the dimensional grid that gives the pixels locations of the intensity map.

!!! note
    If `I` is an `IntensityMap` then grid is not required since the same grid that was use
    for `I` will be used to construct the polarized intensity map

!!! warning
    The return type for this function is a polarized image object, however what we return
    is not considered to be part of the stable API so it may change suddenly.
"""
function PoincareSphere2Map(I, p, X, grid)
    pimgI = I.*p
    stokesI = IntensityMap(I, grid)
    stokesQ = IntensityMap(pimgI .* X[1], grid)
    stokesU = IntensityMap(pimgI .* X[2], grid)
    stokesV = IntensityMap(pimgI .* X[3], grid)
    return StokesIntensityMap(stokesI, stokesQ, stokesU, stokesV)
end
PoincareSphere2Map(I::IntensityMap, p, X) = PoincareSphere2Map(baseimage(I), p, X, axiskeys(I))

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
    return (Q+1im*U)/I
end

"""
    $(SIGNATURES)

Explicit m̆ function used for convenience.
"""
PolarizedTypes.mbreve(pimg::AbstractPolarizedModel, p) = m̆(pimg, p)
