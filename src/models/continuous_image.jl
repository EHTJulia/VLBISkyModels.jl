export ContinuousImage, spatialdims

"""
    ContinuousImage{A<:IntensityMap, P} <: AbstractModel
    ContinuousImage(img::Intensitymap, kernel)

The basic continuous image model for VLBISkyModels. This expects a IntensityMap style object as its imag
as well as a image kernel or pulse that allows you to evaluate the image at any image
and visibility location. The image model is

    I(x,y) = ∑ᵢ Iᵢⱼ κ(x-xᵢ, y-yᵢ)

where `Iᵢⱼ` are the flux densities of the image `img` and κ is the intensity function for the
`kernel`.

Note that if the image grid is the same as the grid passed to `intensitymap` then the discrete image 
is used directly for efficiency.
"""
struct ContinuousImage{P, G, K} <: AbstractModel
    """
    Discrete representation of the image.
    """
    params::P
    """
    The image grid
    """
    grid::G
    """
    Image kernel or pulse that transforms from the discrete image to a continuous one.
    """
    kernel::K
end

make_map(cimg::ContinuousImage) = IntensityMap(cimg.params, cimg.grid)

function Base.show(io::IO, img::ContinuousImage{A, P}) where {A, P}
    sA = split("$A", ",")[1]
    sA = sA * "}"
    return print(io, "ContinuousImage{$sA, $P}($(size(img)))")
end

function ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A <: IntensityMap{<:StokesParams}}
    return IsPolarized()
end
function ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A <: IntensityMap{<:Number}}
    return NotPolarized()
end
# An image whose pixels are stored directly as an array
function ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A <: AbstractArray{<:StokesParams}}
    return IsPolarized()
end
function ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A <: AbstractArray{<:Number}}
    return NotPolarized()
end
# A multidomain image whose pixels are described by a `DomainParams` (e.g. a frequency
# or time model). The polarization is determined by the base parameter element type.
function ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A <: DomainParams}
    return _ispol_paramtype(eltype(paramtype(A)))
end
@inline _ispol_paramtype(::Type{<:StokesParams}) = IsPolarized()
@inline _ispol_paramtype(::Type{<:Number}) = NotPolarized()
ComradeBase.flux(m::ContinuousImage) = flux(make_map(m)) * flux(m.kernel)

function ComradeBase.stokes(cimg::ContinuousImage, v)
    return ContinuousImage(stokes(make_map(cimg), v), cimg.kernel)
end
ComradeBase.centroid(m::ContinuousImage) = centroid(make_map(m))
Base.parent(cimg::ContinuousImage) = make_map(cimg)
Base.length(m::ContinuousImage) = length(make_map(m))
Base.size(m::ContinuousImage) = size(make_map(m))
Base.size(m::ContinuousImage, i::Int) = size(make_map(m), i::Int)
Base.firstindex(m::ContinuousImage) = firstindex(make_map(m))
Base.lastindex(m::ContinuousImage) = lastindex(make_map(m))
Base.eltype(::ContinuousImage{A, P}) where {A, P} = eltype(A)
Base.eltype(::ContinuousImage{A}) where {A <: DomainParams} = eltype(paramtype(A))

Base.getindex(img::ContinuousImage, args...) = getindex(make_map(img), args...)
Base.axes(m::ContinuousImage) = axes(make_map(m))
ComradeBase.domainpoints(m::ContinuousImage) = domainpoints(m.grid)
ComradeBase.axisdims(m::ContinuousImage) = m.grid

function ContinuousImage(img::IntensityMap, pulse::Pulse)
    arr = baseimage(img)
    g = axisdims(img)
    return ContinuousImage{typeof(arr), typeof(g), typeof(pulse)}(arr, g, pulse)
end

function ContinuousImage(im::AbstractMatrix, g::AbstractRectiGrid, pulse)
    return ContinuousImage{typeof(im), typeof(g), typeof(pulse)}(im, g, pulse)
end

function InterpolatedModel(
        model::ContinuousImage,
        d::FourierDualDomain{
            <:AbstractRectiGrid, <:AbstractSingleDomain,
            <:FFTAlg,
        }
    )
    img = make_map(model) # intensity map
    sitp = build_intermodel(img, forward_plan(d), algorithm(d), model.kernel)
    return InterpolatedModel{typeof(model), typeof(sitp)}(model, sitp)
end

# IntensityMap will obey the Comrade interface. This is so I can make easy models
visanalytic(::Type{<:ContinuousImage}) = NotAnalytic() # not analytic b/c we want to hook into FFT stuff
imanalytic(::Type{<:ContinuousImage}) = IsAnalytic()

radialextent(c::ContinuousImage) = maximum(values(fieldofview(spatialdims(c.grid)))) / 2

"""
    spatialdims(g::AbstractRectiGrid)
    spatialdims(img::IntensityMap)

Return the spatial (`X`, `Y`) sub-grid of a (possibly multidomain) grid or image,
dropping any extra dimensions such as frequency (`Fr`) or time (`Ti`).
"""
spatialdims(g::AbstractRectiGrid) = rebuild(g; dims = dims(g)[1:2])
spatialdims(img::IntensityMap) = spatialdims(axisdims(img))

function support_ranges(img, p, rx, ry)
    dx, dy = pixelsizes(img)
    g = axisdims(img)
    x0 = first(g.X)
    y0 = first(g.Y)

    cs = round(Int, (p.X - x0) / dx) + 1
    rs = round(Int, (p.Y - y0) / dy) + 1

    # Units in pixels
    wx = ceil(Int, rx / dx)
    wy = ceil(Int, ry / dy)

    ix = max(firstindex(img, 1), cs - wx):min(lastindex(img, 1), cs + wx)
    iy = max(firstindex(img, 2), rs - wy):min(lastindex(img, 2), rs + wy)

    return ix, iy
end

function ComradeBase.build_param(param::AbstractArray{<:Number}, p)
    return param
end


@inline function intensity_point(m::ContinuousImage, p)
    img = make_map(m)
    dx, dy = pixelsizes(axisdims(img))
    ms = stretched(m.kernel, dx, dy)

    rx, ry = kernel_extent(m.kernel)
    rx *= dx
    ry *= dy

    dp = domainpoints(img)
    sum = zero(eltype(img))

    ix, iy = support_ranges(img, p, rx, ry)

    @trace for j in iy
        @trace for i in ix
            dpi = (X = p.X - dp[i, j].X, Y = p.Y - dp[i, j].Y)
            k = intensity_point(ms, dpi)
            sum += rgetindex(img, i, j) * k
        end
    end
    return sum
end


# function intensity_point(m::ContinuousImage, p)
#     @unpack_params img = m(p)
#     dx, dy = pixelsizes(m.img)
#     sum = zero(eltype(m.img))
#     ms = stretched(m.kernel, dx, dy)
#     @inbounds for (I, p0) in pairs(domainpoints(m.img))
#         dp = (X = (p.X - p0.X), Y = (p.Y - p0.Y))
#         k = intensity_point(ms, dp)
#         sum += m.img[I] * k
#     end
#     return sum
# end

function convolved(cimg::ContinuousImage, m::AbstractModel)
    return ContinuousImage(cimg.params, cimg.grid, convolved(cimg.kernel, m))
end
convolved(cimg::AbstractModel, m::ContinuousImage) = convolved(m, cimg)

# @inline function ModifiedModel(m::ContinuousImage, t::Tuple)
#     doesnot_uv_modify(t) === Static.False() && throw(
#                           ArgumentError(
#                             "ContinuousImage does not support modifying the uv plane."*
#                             "This would require a dynamic grid which is not currently implemented"*
#                             "Transformations like rotations just introduce additional degeneracies,
#                              making imaging more difficult"
#                             ))
#     return ModifiedModel{typeof(m), typeof(t)}(m, t)
# end


@inline function visibilitymap_numeric(m::ContinuousImage, grid::FourierDualDomain)
    # We need to make sure that the grid is the same size as the image
    checkgrid(axisdims(m), imgdomain(grid))
    img = make_map(m)
    vis = applyft(forward_plan(grid), img)
    return applypulse!(vis, m.kernel, grid)
end

# Multidomain (e.g. multifrequency/multitime) images. The `params` field is a
# `DomainParams` (such as a `MultiDomainParams` or `PolySpectral`) that knows how to
# materialize the spatial image at each point of the extra (`Fr`/`Ti`) dimensions.
# The stored `grid` is the spatial (X, Y) grid; the full cube grid comes from the
# image domain of the visibility grid passed to `visibilitymap`.

# Shape with `len` on axis `i` and `1` elsewhere, e.g. `_axisshape(nfr, 3, Val(3)) = (1,1,nfr)`.
@inline _axisshape(len::Int, i::Int, ::Val{Nd}) where {Nd} = ntuple(j -> j == i ? len : 1, Val(Nd))

# Factor a grid's coordinates for broadcasting: a NamedTuple mapping each dimension name
# to its coordinate vector reshaped onto that dimension's axis (e.g. `Fr -> (1, 1, nfr)`).
# Passing this to `build_param` lets the spectral model broadcast the base image across the
# extra `Fr`/`Ti` axes, materializing the whole cube in one pass. Written type-stably
# (per-dimension `map`, `Val`-sized shapes, `basedim` for values) so it is allocation-light
# and differentiable by Enzyme — the previous runtime-`getproperty`/`ntuple` form produced
# a `Union` that broke Enzyme's type analysis.
function _cubepoint(g::AbstractRectiGrid)
    ds = dims(g)
    nd = Val(length(ds))
    nms = map(name, ds)
    coords = map(ds, ntuple(identity, nd)) do d, i
        v = collect(ComradeBase.basedim(d))
        reshape(v, _axisshape(length(v), i, nd))
    end
    return NamedTuple{nms}(coords)
end

# Materialize the spatial image cube by evaluating the domain model `params` over the
# whole cube grid `g` in a single broadcast (via `_cubepoint`). This is generic (any
# `DomainParams` whose `build_param` is written with broadcasts works), allocation-light,
# type stable, and traceable by Reactant — unlike a per-slice `mapslices`/scalar loop.
function _paramcube(params::DomainParams, g::AbstractRectiGrid)
    raw = build_param(params, _cubepoint(g))
    sz = size(g)
    size(raw) == sz && return raw
    # A model need not depend on every extra dimension (e.g. a frequency-only model on a
    # time+frequency grid); broadcast it up to the full cube shape.
    cube = similar(raw, sz)
    cube .= raw
    return cube
end

function intensitymap_analytic(m::ContinuousImage{<:DomainParams}, dims::AbstractRectiGrid)
    gspat = spatialdims(dims)
    datacube = _paramcube(m.params, dims)
    out = allocate_imgmap(m, dims)
    # Per-frequency/time kernel convolution. This path is not part of the Reactant forward
    # model (which uses `visibilitymap`), so a simple slice loop over preallocated `out`
    # is fine and keeps it type stable.
    @inbounds for k in CartesianIndices(axes(datacube)[3:end])
        sl = ContinuousImage(view(datacube, :, :, k), gspat, m.kernel)
        intensitymap_analytic!(IntensityMap(view(out, :, :, k), gspat), sl)
    end
    return out
end

function intensitymap_analytic!(img::IntensityMap, m::ContinuousImage{<:DomainParams})
    copyto!(parent(img), intensitymap_analytic(m, axisdims(img)))
    return nothing
end

function visibilitymap_numeric(m::ContinuousImage{<:DomainParams}, grid::FourierDualDomain)
    gimg = imgdomain(grid)
    checkgrid(axisdims(m), spatialdims(gimg))
    mfimg = IntensityMap(_paramcube(m.params, gimg), gimg)
    vis = applyft(forward_plan(grid), mfimg)
    return applypulse!(vis, m.kernel, grid)
end

@inline function visibilitymap_numeric(
        m::ContinuousImage,
        grid::FourierDualDomain{GI, GV, <:FFTAlg}
    ) where {
        GI <: AbstractSingleDomain,
        GV <: AbstractSingleDomain,
    }
    minterp = InterpolatedModel(m, grid)
    return visibilitymap(minterp, visdomain(grid))
end

function applypulse!(vis, pulse, gfour::AbstractFourierDualDomain)
    grid = imgdomain(gfour)
    guv = visdomain(gfour)
    dx, dy = pixelsizes(grid)
    mp = stretched(pulse, dx, dy)
    # we grab the parent array since for some reason Enzyme struggles to see
    # through the broadcast
    pvis = parent(vis)
    dp = domainpoints(guv)
    vp = Base.Fix1(visibility_point, mp)
    pvis .*= vp.(dp)
    # for i in eachindex(pvis, dp)
    #     pvis[i] *= visibility_point(mp, dp[i])
    # end
    # pvis .*= visibility_point.(Ref(mp), dp)
    return vis
end

# function intensitymap_analytic!(img::IntensityMap, m::Union{ContinuousImage, ModifiedModel{<:ContinuousImage}})
#     intensitymap_numeric!(img, m)
#     # guv = uvgrid(axisdims(img))
#     # U = guv.U.*ones(length(guv.V))' |> vec
#     # V = ones(length(guv.U)).*guv.V' |> vec
#     # gfour = FourierDualDomain(g, UnstructuredDomain((;U, V)), FFTAlg())
#     # vis = reshape(parent(visibilitymap_numeric(m, gfour)), size(img))
#     # img .= real.(ifftshift(ifft!(fftshift(conj.(vis)))))
#     return nothing
# end

# function visibilitymap_numeric!(img::IntensityMap, m::ContinuousImage)
#     gfour = FourierDualDomain(axisdims(parent(m)), axisdims(img), FFTAlg())
#     img .= visibilitymap_numeric(m, gfour)
#     return nothing
# end

# Make a special pass through for this as well
function visibilitymap_numeric(
        m::ContinuousImage,
        grid::FourierDualDomain{GI, GV, <:FFTAlg}
    ) where {GI, GV}
    minterp = InterpolatedModel(m, grid)
    return visibilitymap(minterp, visdomain(grid))
end

function checkgrid(imgdims, grid)
    truth = (dims(imgdims) == dims(grid))
    return truth
end
ChainRulesCore.@non_differentiable checkgrid(::Any, ::Any)
EnzymeRules.inactive(::typeof(checkgrid), args...) = nothing

# A special pass through for Modified ContinuousImage
const ScalingTransform = Union{Shift, Renormalize}
function visibilitymap_numeric(
        m::ModifiedModel{M, T},
        p::FourierDualDomain
    ) where {
        M <: ContinuousImage, N,
        T <: NTuple{
            N,
            ScalingTransform,
        },
    }
    ispol = ispolarized(M)
    vbase = visibilitymap_numeric(m.model, p)
    puv = visdomain(p)
    _apply_scaling!(ispol, m.transform, vbase, puv)
    return vbase
end


@inline function _apply_scaling!(mbase, t::Tuple, vbase, p)
    # out = similar(vbase)
    pvbase = baseimage(vbase)
    uc = unitscale(complex(eltype(p.U)), mbase)
    dp = domainpoints(p)
    pvbase .*= last.(modify_uv.(Ref(mbase), Ref(t), dp, Ref(uc)))
    return nothing
end
