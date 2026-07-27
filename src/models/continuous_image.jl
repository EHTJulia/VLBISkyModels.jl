export ContinuousImage, spatialdims

"""
    ContinuousImage{P, G, K} <: AbstractModel
    ContinuousImage(img::IntensityMap, kernel)
    ContinuousImage(im::AbstractArray, grid::AbstractRectiGrid, kernel)

The basic continuous image model for VLBISkyModels. This expects an IntensityMap style object
(or an array of pixel parameters together with its grid)
as well as a image kernel or pulse that allows you to evaluate the image at any image
and visibility location. The image model is

    I(x,y) = ∑ᵢ Iᵢⱼ κ(x-xᵢ, y-yᵢ)

where `Iᵢⱼ` are the flux densities of the image `img` and κ is the intensity function for the
`kernel`.

The pixel parameters may also be a `DomainParams` (e.g. a [`MultiDomainParams`](@ref)
built by [`MultiDomainImage`](@ref)) describing how the image varies across frequency
and/or time.

Note that if the image grid is the same as the grid passed to `intensitymap` then the discrete image
is used directly for efficiency.
"""
struct ContinuousImage{
        P <: Union{AbstractArray, DomainParams}, G <: AbstractRectiGrid,
        K <: AbstractModel,
    } <: AbstractModel
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
# A bare spectral/temporal model has no static spatial map: there is no base image to
# show until the model is evaluated at a frequency/time.
function make_map(cimg::ContinuousImage{<:DomainParams})
    throw(
        ArgumentError(
            "A `$(nameof(typeof(cimg.params)))`-parameterized image has no static " *
                "spatial map; evaluate it with `intensitymap` over a grid with the " *
                "extra `Fr`/`Ti` dimensions."
        )
    )
end
# For a multidomain `ContinuousImage` the `params` field is a `MultiDomainParams` whose
# base spatial image lives at the root of a (possibly nested) chain, and `grid` is the
# spatial (X, Y) grid. Build the spatial map from that base so `size`/`show`/`flux`/etc.
# work on these images.
make_map(cimg::ContinuousImage{<:MultiDomainParams}) = IntensityMap(baseparams(cimg.params), cimg.grid)

function Base.show(io::IO, img::ContinuousImage)
    pname = nameof(typeof(img.params))
    kname = nameof(typeof(img.kernel))
    # Use the grid for the size: it is defined even for bare `DomainParams` params,
    # which have no static spatial map.
    return print(io, "ContinuousImage{$pname{$(eltype(img))}, $kname}($(size(img.grid)))")
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
# For a multidomain image these reduce to the base (reference-domain) spatial image via
# `make_map`, i.e. `flux`/`centroid` describe the reference-frequency/-time image, not the
# whole cube. Evaluate `intensitymap` over an `Fr`/`Ti` grid for per-domain quantities.
ComradeBase.flux(m::ContinuousImage) = flux(make_map(m)) * flux(m.kernel)

function ComradeBase.stokes(cimg::ContinuousImage, v)
    return ContinuousImage(stokes(make_map(cimg), v), cimg.kernel)
end
# Project a multidomain image onto a Stokes component by projecting the base spatial
# `params` array and keeping the domain models, so the multidomain structure (and the
# spatial `grid`/`kernel`) is preserved. The generic `make_map`-based method would instead
# collapse to a plain base image. Like plain images, `stokes` is polarized-only (the
# root base must be a `StokesParams` array).
function ComradeBase.stokes(cimg::ContinuousImage{<:MultiDomainParams}, v)
    return ContinuousImage(stokes(cimg.params, v), cimg.grid, cimg.kernel)
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

function ContinuousImage(img::IntensityMap, kernel)
    arr = baseimage(img)
    g = axisdims(img)
    return ContinuousImage{typeof(arr), typeof(g), typeof(kernel)}(arr, g, kernel)
end

function ContinuousImage(im::AbstractArray, g::AbstractRectiGrid, kernel::AbstractModel)
    size(im) == size(g) || throw(
        DimensionMismatch(
            "The image array size $(size(im)) does not match the grid size $(size(g))."
        )
    )
    arr = baseimage(im)
    return ContinuousImage{typeof(arr), typeof(g), typeof(kernel)}(arr, g, kernel)
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
function spatialdims(g::AbstractRectiGrid)
    ds = dims(g)
    return rebuild(g; dims = ds[1:2])
end
spatialdims(img::IntensityMap) = spatialdims(axisdims(img))

@doc """
    MultiDomainImage(img::IntensityMap, kernel, domains...)
    MultiDomainImage(cimg::ContinuousImage, domains...)

Convenience constructor for a multidomain (e.g. multifrequency or multitime)
[`ContinuousImage`](@ref).

The spatial image `img` (a 2D `IntensityMap`) provides the base parameters and the
spatial `(X, Y)` grid, `kernel` is the image pulse, and `domains...` are one or more
`DomainParams` models (such as [`PolySpectral`](@ref)) describing how the image varies
across the extra domains.

The result is a `ContinuousImage` whose `params` field is a [`MultiDomainParams`](@ref).
When passed to `intensitymap` or `visibilitymap` over a grid with extra `Fr`/`Ti`
dimensions the spatial image cube is materialized by evaluating the domain models at
each frequency/time.

# Example
```julia
base = IntensityMap(rand(64, 64), imagepixels(10.0, 10.0, 64, 64))
dom  = PolySpectral((1.0,), 230.0e9)          # spectral index = 1
cimg = MultiDomainImage(base, BSplinePulse{3}(), dom)
```
"""
function MultiDomainImage(img::SpatialIntensityMap, kernel, domains...)
    mdp = MultiDomainParams(parent(img), domains)
    return ContinuousImage(mdp, spatialdims(img), kernel)
end

function MultiDomainImage(cimg::ContinuousImage, domains...)
    length(dims(cimg.grid)) == 2 || throw(
        ArgumentError(
            "MultiDomainImage expects an image on a 2D spatial grid, got " *
                "$(length(dims(cimg.grid))) dimensions; the extra Fr/Ti structure " *
                "comes from `domains`."
        )
    )
    mdp = MultiDomainParams(cimg.params, domains)
    return ContinuousImage(mdp, cimg.grid, cimg.kernel)
end

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
    # Evaluate the stored params at the domain point so multidomain images keep their
    # frequency/time dependence here (this is the path composite models sum through).
    # For a plain array `params` this is a no-op passthrough.
    img = IntensityMap(build_param(m.params, p), m.grid)
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
# extra `Fr`/`Ti` axes, materializing the whole cube in one pass. Must stay type-stable
# (per-dimension `map`, `Val`-sized shapes, `basedim` for values): a `Union`-typed
# coordinate tuple here breaks Enzyme's type analysis.
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
    out = allocate_imgmap(m, dims)
    intensitymap_analytic!(out, m)
    return out
end

function intensitymap_analytic!(img::IntensityMap, m::ContinuousImage{<:DomainParams})
    dims = axisdims(img)
    gspat = spatialdims(dims)
    datacube = _paramcube(m.params, dims)
    # Per-frequency/time kernel convolution, written directly into the caller's buffer.
    # This path is not part of the Reactant forward model (which uses `visibilitymap`),
    # so a simple slice loop is fine and keeps it type stable.
    pimg = parent(img)
    for k in CartesianIndices(axes(datacube)[3:end])
        sl = ContinuousImage(view(datacube, :, :, k), gspat, m.kernel)
        intensitymap_analytic!(IntensityMap(view(pimg, :, :, k), gspat), sl)
    end
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

# Disambiguates the multidomain method from the FFTAlg fast path above. FFTAlg evaluates
# visibilities by interpolating a single 2D FFT (`InterpolatedModel`), which has no
# notion of the extra `Fr`/`Ti` axes, so multidomain images cannot use it.
function visibilitymap_numeric(
        ::ContinuousImage{<:DomainParams},
        ::FourierDualDomain{GI, GV, <:FFTAlg}
    ) where {
        GI <: AbstractSingleDomain,
        GV <: AbstractSingleDomain,
    }
    throw(
        ArgumentError(
            "FFTAlg does not support multidomain (frequency/time dependent) images. " *
                "Use a nonuniform transform such as NFFTAlg or DFTAlg instead."
        )
    )
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
