export StokesIntensityMap


"""
    $(TYPEDEF)

General struct that holds intensity maps for each stokes parameter. Each image `I, Q, U, V`
must share the same axis dimensions. This type also obeys much of the usual array interface
in Julia. The following methods have been implemented:

  - size
  - eltype (returns StokesParams)
  - ndims
  - getindex
  - setindex!
  - pixelsizes
  - fieldofview
  - imagepixels
  - domainpoints
  - stokes


!!! warning
    This may eventually be phased out for `IntensityMaps` whose base types are `StokesParams`,
    but currently we use this for speed reasons with Zygote.
"""
struct StokesIntensityMap{T, N, SI<:AbstractArray{T,N}, SQ<:AbstractArray{T,N}, SU<:AbstractArray{T, N}, SV<:AbstractArray{T, N}, G<:AbstractRectiGrid}
    """
    Stokes I image
    """
    I::SI
    """
    Stokes Q image
    """
    Q::SQ
    """
    Stokes U image
    """
    U::SU
    """
    Stokes V image
    """
    V::SV
    """
    grid
    """
    grid::G
end

function StokesIntensityMap(
    I::IntensityMap{T,N},
    Q::IntensityMap{T,N},
    U::IntensityMap{T,N},
    V::IntensityMap{T,N}) where {T<:Real, N}

    check_grid(I, Q, U, V)
    g = axisdims(I)
    return StokesIntensityMap(baseimage(I), baseimage(Q), baseimage(U), baseimage(V), g)
end


function StokesIntensityMap(img::IntensityMap{<:StokesParams})
    return StokesIntensityMap(stokes(img, :I), stokes(img, :Q), stokes(img, :U), stokes(img, :V))
end

Base.size(im::StokesIntensityMap) = size(im.I)
Base.eltype(::StokesIntensityMap{T}) where {T} = StokesParams{T}
Base.ndims(::StokesIntensityMap{T,N}) where {T,N} = N
Base.ndims(::Type{<:StokesIntensityMap{T,N}}) where {T,N} = N
Base.@propagate_inbounds Base.getindex(im::StokesIntensityMap, i::Int) = StokesParams(getindex(im.I, i),getindex(im.Q, i),getindex(im.U, i),getindex(im.V, i))
Base.@propagate_inbounds Base.getindex(im::StokesIntensityMap, I...) = StokesParams.(getindex(im.I, I...), getindex(im.Q, I...), getindex(im.U, I...), getindex(im.V, I...))

function Base.setindex!(im::StokesIntensityMap, x::StokesParams, inds...)
    setindex!(im.I, x.I, inds...)
    setindex!(im.Q, x.Q, inds...)
    setindex!(im.U, x.U, inds...)
    setindex!(im.V, x.V, inds...)
end

Enzyme.EnzymeRules.inactive(::typeof(axisdims), ::StokesIntensityMap) = nothing
ComradeBase.axisdims(img::StokesIntensityMap) = img.grid
ChainRulesCore.@non_differentiable ComradeBase.axisdims(::StokesIntensityMap)
ComradeBase.pixelsizes(img::StokesIntensityMap)  = pixelsizes(axisdims(img))
ComradeBase.fieldofview(img::StokesIntensityMap) = fieldofview(axisdims(img))
ComradeBase.domainpoints(img::StokesIntensityMap)   = domainpoints(axisdims(img))
ComradeBase.flux(img::StokesIntensityMap{T}) where {T} = StokesParams(flux(stokes(img, :I)),
                                             flux(stokes(img, :Q)),
                                             flux(stokes(img, :U)),
                                             flux(stokes(img, :V)),
                                             )
VLBISkyModels.centroid(m::StokesIntensityMap) = centroid(stokes(m, :I))


# simple check to ensure that the four grids are equal across stokes parameters
function check_grid(I::IntensityMap, Q::IntensityMap,U::IntensityMap ,V::IntensityMap)
    axisdims(I) == axisdims(Q) == axisdims(U) == axisdims(V)
end

ChainRulesCore.@non_differentiable check_grid(IntensityMap...)

@inline function ComradeBase.stokes(pimg::StokesIntensityMap, v::Symbol)
    return IntensityMap(getfield(pimg, v), axisdims(pimg))
end


function Base.summary(io::IO, x::StokesIntensityMap)
    return _summary(io, x)
end

function _summary(io, x::StokesIntensityMap{T,N}) where {T,N}
    println(io, ndims(x), "-dimensional")
    println(io, "StokesIntensityMap{$T, $N}")
    print(io, "   Stokes I: ")
    summary(io, x.I)
    print(io, "\n   Stokes Q: ")
    summary(io, x.Q)
    print(io, "\n   Stokes U: ")
    summary(io, x.U)
    print(io, "\n   Stokes V: ")
    summary(io, x.V)
end

Base.show(io::IO, img::StokesIntensityMap) = summary(io, img)

function IntensityMap(img::StokesIntensityMap)
    I = stokes(img, :I) |> baseimage
    Q = stokes(img, :Q) |> baseimage
    U = stokes(img, :U) |> baseimage
    V = stokes(img, :V) |> baseimage

    simg = StructArray{StokesParams{eltype(I)}}(;I, Q, U, V)
    return IntensityMap(simg, axisdims(img), refdims(I), name(stokes(img, :I)))
end
