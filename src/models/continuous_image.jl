export ContinuousImage, MultiDomainImage

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

The pixel parameters may also be a [`MultiDomainParams`](@ref) describing how the image
varies across frequency and/or time, in which case the image is a
[`MultiDomainImage`](@ref). A bare `DomainParams` is not accepted: an image needs a base,
which a chain supplies and a lone spectral or temporal model does not.

Note that if the image grid is the same as the grid passed to `intensitymap` then the discrete image
is used directly for efficiency.

!!! note
    `intensity_point` on a [`MultiDomainImage`](@ref) evaluates the domain models over the
    kernel's support window around the requested point, not over the whole image. Modifiers
    that displace the image (`ModifiedModel`) still evaluate point by point.
"""
struct ContinuousImage{
        P <: Union{AbstractArray, MultiDomainParams}, G <: AbstractRectiGrid,
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

# The bounds on `G` and `K` must repeat those of `ContinuousImage`. Leaving them off widens
# those slots, which makes the alias and `ContinuousImage` incomparable rather than nested,
# and methods signed on the alias then lose dispatch to the general ones.
const MultiDomainImage{
    M <: MultiDomainParams, G <: AbstractRectiGrid, K <: AbstractModel,
} = ContinuousImage{M, G, K}

@doc """
    MultiDomainImage{M<:MultiDomainParams, G, K}
    MultiDomainImage(img::IntensityMap, kernel, domains...)
    MultiDomainImage(cimg::ContinuousImage, domains...)

A [`ContinuousImage`](@ref) whose pixel parameters vary across the extra (frequency and/or
time) domains, i.e. one whose `params` field is a [`MultiDomainParams`](@ref). This is a
type alias, so `img isa MultiDomainImage` and dispatch on it both work.

The spatial image `img` (a 2D `IntensityMap`) provides the base parameters and the
spatial `(X, Y)` grid, `kernel` is the image pulse, and `domains...` are one or more
`DomainParams` models (such as [`PolySpectral`](@ref)) describing how the image varies
across the extra domains.

When passed to `intensitymap` or `visibilitymap` over a grid with extra `Fr`/`Ti`
dimensions the spatial image cube is materialized by evaluating the domain models at
each frequency/time. `FFTAlg` cannot transform these images, since it interpolates a
single 2D FFT; use a nonuniform transform such as `NFFTAlg` or `DFTAlg`.

Chaining extends the model tuple rather than nesting, so
`MultiDomainImage(MultiDomainImage(img, kernel, m1), m2)` and
`MultiDomainImage(img, kernel, m1, m2)` are the same model.

# Example
```julia
base = IntensityMap(rand(64, 64), imagepixels(10.0, 10.0, 64, 64))
dom  = PolySpectral((1.0,), 230.0e9)          # spectral index = 1
cimg = MultiDomainImage(base, BSplinePulse{3}(), dom)
```
""" MultiDomainImage

make_map(cimg::ContinuousImage) = IntensityMap(cimg.params, cimg.grid)
# For a multidomain image the `params` field is a chain whose base is the spatial image at
# the reference domain point, and `grid` is the spatial (X, Y) grid. Build the map from that
# base so `size`/`show`/`flux`/etc. describe the reference image.
make_map(cimg::MultiDomainImage) = IntensityMap(cimg.params.base, cimg.grid)

function Base.show(io::IO, img::ContinuousImage)
    pname = nameof(typeof(img.params))
    kname = nameof(typeof(img.kernel))
    # The grid gives the spatial size for both plain and multidomain images; the extra
    # `Fr`/`Ti` extent belongs to the grid an image is evaluated over, not to the image.
    return print(io, "ContinuousImage{$pname{$(eltype(img))}, $kname}($(size(img.grid)))")
end

# `paramtype` reports the element type of a single pixel whether the pixels are stored
# directly as an array or described by a chain of domain models, so both cases read the
# same way here.
ComradeBase.ispolarized(::Type{<:ContinuousImage{P}}) where {P} = _ispol(paramtype(P))
@inline _ispol(::Type{<:StokesParams}) = IsPolarized()
@inline _ispol(::Type{<:Number}) = NotPolarized()
# For a multidomain image these reduce to the base (reference-domain) spatial image via
# `make_map`, i.e. `flux`/`centroid` describe the reference-frequency/-time image, not the
# whole cube. Evaluate `intensitymap` over an `Fr`/`Ti` grid for per-domain quantities.
ComradeBase.flux(m::ContinuousImage) = flux(make_map(m)) * flux(m.kernel)

function ComradeBase.stokes(cimg::ContinuousImage, v)
    return ContinuousImage(stokes(make_map(cimg), v), cimg.kernel)
end
# Project a multidomain image onto a Stokes component by projecting the whole chain, so the
# multidomain structure (and the spatial `grid`/`kernel`) is preserved. The generic
# `make_map`-based method would instead collapse to a plain base image.
function ComradeBase.stokes(cimg::MultiDomainImage, v)
    return ContinuousImage(stokes(cimg.params, v), cimg.grid, cimg.kernel)
end
ComradeBase.centroid(m::ContinuousImage) = centroid(make_map(m))
Base.parent(cimg::ContinuousImage) = make_map(cimg)
Base.length(m::ContinuousImage) = length(make_map(m))
Base.size(m::ContinuousImage) = size(make_map(m))
Base.size(m::ContinuousImage, i::Int) = size(make_map(m), i::Int)
Base.firstindex(m::ContinuousImage) = firstindex(make_map(m))
Base.lastindex(m::ContinuousImage) = lastindex(make_map(m))
Base.eltype(::ContinuousImage{P}) where {P} = paramtype(P)

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

function ContinuousImage(
        params::MultiDomainParams, g::AbstractRectiGrid, kernel::AbstractModel
    )
    base = params.base
    base isa AbstractArray || throw(
        ArgumentError(
            "The parameters of an image must be a chain whose base is the spatial image " *
                "array; got a base of type $(nameof(typeof(base)))."
        )
    )
    size(base) == size(g) || throw(
        DimensionMismatch(
            "The base image size $(size(base)) does not match the grid size $(size(g))."
        )
    )
    return ContinuousImage{typeof(params), typeof(g), typeof(kernel)}(params, g, kernel)
end

# A lone spectral or temporal model carries no base image, so it cannot describe one.
function ContinuousImage(params::DomainParams, ::AbstractRectiGrid, ::AbstractModel)
    throw(
        ArgumentError(
            "A `$(nameof(typeof(params)))` has no base image. Pair it with one using " *
                "`MultiDomainParams(base, model)`, or build the image with `MultiDomainImage`."
        )
    )
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

# A point expressed on the grid's own axes. `domainpoints` places pixel `(i, j)` at
# `rotmat(g) * (X[i], Y[j])`, so undoing that rotation once turns the grid back into a plain
# `X`/`Y` product for everything downstream. The identity for an unrotated grid.
@inline function togrid(g::AbstractRectiGrid, p)
    v = ComradeBase.rotmat(g)' * SVector(p.X, p.Y)
    return (X = v[1], Y = v[2])
end

# The center pixel of the kernel's support around `p`, which must already be on the grid's
# axes (see `togrid`). Only this position depends on the point; the support half-widths
# below are fixed by the grid and pulse alone.
function support_center(g::AbstractRectiGrid, p)
    dx, dy = pixelsizes(g)
    cs = round(Int, (p.X - first(g.X)) / dx) + firstindex(g.X)
    rs = round(Int, (p.Y - first(g.Y)) / dy) + firstindex(g.Y)
    return cs, rs
end

# The support half-widths in pixels: static once the grid and pulse are fixed.
function support_halfwidths(g::AbstractRectiGrid, rx, ry)
    dx, dy = pixelsizes(g)
    return ceil(Int, rx / dx), ceil(Int, ry / dy)
end

# The pixel index ranges covering the kernel's support around `p`, clamped to the grid.
# The center itself is clamped first, so the ranges are never empty even for a point whose
# whole window lies off the grid: the masked accumulation below still needs somewhere
# valid to (harmlessly) read, and every off-grid tap contributes zero regardless.
function support_ranges(g::AbstractRectiGrid, p, rx, ry)
    cs, rs = support_center(g, p)
    wx, wy = support_halfwidths(g, rx, ry)
    X = g.X
    Y = g.Y
    csf = clamp(cs, firstindex(X), lastindex(X))
    rsf = clamp(rs, firstindex(Y), lastindex(Y))
    ix = max(firstindex(X), csf - wx):min(lastindex(X), csf + wx)
    iy = max(firstindex(Y), rsf - wy):min(lastindex(Y), rsf + wy)
    return ix, iy
end

# Pixel values covering the support window `(ix, iy)`, together with the offsets that turn a
# grid index into an index into them. Stored pixels are used as they are, so the offsets are
# zero; a chain of domain models is restricted to the window and evaluated there, which is
# what keeps a single point from costing the whole image.
@inline window_values(params::AbstractArray, ix, iy, p) = (params, 0, 0)
@inline function window_values(params::MultiDomainParams, ix, iy, p)
    sub = build_param(restrict_params(params, ix, iy), p)
    return sub, first(ix) - first(axes(sub, 1)), first(iy) - first(axes(sub, 2))
end

@inline function intensity_point(m::ContinuousImage, p)
    g = m.grid
    dx, dy = pixelsizes(g)
    ms = stretched(m.kernel, dx, dy)

    rx, ry = kernel_extent(m.kernel)
    rx *= dx
    ry *= dy

    # Both the support window and the pulse offset are computed on the grid's axes, so they
    # stay consistent with each other and with the pixel footprints when the grid is rotated.
    # The pulse is the pixel response, which is aligned with the pixels rather than the sky.
    pg = togrid(g, p)
    ix, iy = support_ranges(g, pg, rx, ry)
    vals, i0, j0 = window_values(m.params, ix, iy, p)

    X = g.X
    Y = g.Y
    sum = zero(eltype(vals))

    cs, rs = support_center(g, pg)
    wx, wy = support_halfwidths(g, rx, ry)
    ilo, ihi = firstindex(X), lastindex(X)
    jlo, jhi = firstindex(Y), lastindex(Y)

    # The loops run over the static support offsets — only the window center depends on the
    # point — so the trip count is fixed by the grid and pulse, and a tracing compiler may
    # unroll or keep the loop as it sees fit. A tap falling off the grid is masked to zero
    # rather than truncating the ranges, which is what the clamped `support_ranges` did;
    # its clamped index always lands inside the fetched window, so the masked read is
    # in bounds. `track_numbers = false` for the same reason as the slice loop of a
    # multidomain image: promoting the numbers captured by the loop would put a traced
    # value into the `LinRange` of grid coordinates, whose length parameter cannot hold
    # one. The coordinate lookups go through `rgetindex` like the pixel values, so they
    # too accept a traced index.
    @trace track_numbers = false for dj in -wy:wy
        @trace track_numbers = false for di in -wx:wx
            i = cs + di
            j = rs + dj
            inb = (ilo <= i) & (i <= ihi) & (jlo <= j) & (j <= jhi)
            ic = clamp(i, ilo, ihi)
            jc = clamp(j, jlo, jhi)
            dpi = (X = pg.X - rgetindex(X, ic), Y = pg.Y - rgetindex(Y, jc))
            k = intensity_point(ms, dpi)
            v = rgetindex(vals, ic - i0, jc - j0)
            sum += ifelse(inb, v * k, zero(sum))
            nothing
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

function intensitymap_analytic(m::MultiDomainImage, dims::AbstractRectiGrid)
    out = allocate_imgmap(m, dims)
    intensitymap_analytic!(out, m)
    return out
end

# The grid the domain models are materialized on: the image's own spatial grid crossed with
# the extra (`Fr`/`Ti`) dimensions of the grid being evaluated over. The pixel parameters
# describe the image on `m.grid`, so that is where the cube has to be built.
function _cubegrid(gspat::AbstractRectiGrid, gout::AbstractRectiGrid)
    return rebuild(gspat; dims = (dims(gspat)..., dims(gout)[3:end]...))
end

# Grid evaluation of any `ContinuousImage` goes through the slice resampler: a plain image
# is the one-slice case, a stored cube resamples slice by slice, and a multidomain image
# (below) materializes its cube first. Composite and modified models still evaluate per
# point through the generic path.
function intensitymap_analytic!(img::IntensityMap, m::ContinuousImage)
    gout = axisdims(img)
    gspat = spatialdims(gout)
    _resample_slices!(parent(img), m.params, m, gspat)
    return nothing
end

function intensitymap_analytic!(img::IntensityMap, m::MultiDomainImage)
    gout = axisdims(img)
    gspat = spatialdims(gout)
    datacube = _paramcube(m.params, _cubegrid(m.grid, gout))
    _resample_slices!(parent(img), datacube, m, gspat)
    return nothing
end

# Per-slice kernel resampling, written directly into the caller's buffer `pimg`. Each
# spatial slice is a `ContinuousImage` on the image's own grid, resampled onto `gspat`
# through `intensity_point` by the executor of `gspat` — called directly, since the entry
# points above own the `intensitymap_analytic!` dispatch for images.
#
# Collapsing the dimensions past `X`/`Y` into one axis covers any `Fr`/`Ti` structure with
# a single loop: the cube carries those dimensions in the order the output grid does, so
# both arrays flatten to the same points in the same order. A single stored slice
# evaluated over a larger grid is replicated, matching per-point evaluation of a spatial
# image at any `Fr`/`Ti`.
#
# A traced array can be neither reshaped without a wrapper nor viewed at a traced index, so
# the Reactant extension overrides this method with a `dynamic_slice`/`dynamic_update_slice`
# version, leaving this one as the allocation-free CPU path.
function _resample_slices!(pimg::AbstractArray, datacube, m::ContinuousImage, gspat)
    # The dominant case — a spatial image evaluated over a spatial grid — pays no slice
    # machinery: the executor reads the model's raw pixel array directly. (`datacube` is
    # `m.params` itself exactly when no multidomain cube was materialized.)
    if ndims(pimg) == 2 && ndims(datacube) == 2 && datacube === m.params
        return ComradeBase.intensitymap_analytic_executor!(
            IntensityMap(pimg, gspat), m, ComradeBase.executor(gspat)
        )
    end
    gsrc = spatialdims(m.grid)
    cube = reshape(datacube, size(gsrc)..., :)
    rimg = reshape(pimg, size(gspat)..., :)
    nsl = size(cube, 3)
    (nsl == size(rimg, 3) || nsl == 1) || throw(
        DimensionMismatch(
            "The image carries $(nsl) spatial slices but the output grid carries " *
                "$(size(rimg, 3))."
        )
    )
    ex = ComradeBase.executor(gspat)
    for k in axes(rimg, 3)
        sl = ContinuousImage(view(cube, :, :, min(k, nsl)), gsrc, m.kernel)
        ComradeBase.intensitymap_analytic_executor!(
            IntensityMap(view(rimg, :, :, k), gspat), sl, ex
        )
    end
    return nothing
end

function visibilitymap_numeric(m::MultiDomainImage, grid::FourierDualDomain)
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

# FFTAlg evaluates visibilities by interpolating a single 2D FFT (`InterpolatedModel`),
# which has no notion of the extra `Fr`/`Ti` axes, so multidomain images cannot use it. This
# method is load-bearing for dispatch as well as for the message: without it the multidomain
# method and the FFTAlg fast path above are ambiguous.
function visibilitymap_numeric(
        ::MultiDomainImage,
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

# The Fourier plans are built from the grid the visibilities are computed on, so the image
# must live on exactly that grid: pixels spanning a different field of view or lying at a
# different position angle would be transformed as if they sat at the wrong sky positions.
function checkgrid(imgdims, grid)
    (dims(imgdims) == dims(grid) && posang(imgdims) == posang(grid)) && return nothing
    throw(
        DimensionMismatch(
            "The image grid does not match the grid the visibilities are computed on.\n" *
                "  image: $(dims(imgdims)), posang = $(posang(imgdims))\n" *
                "  grid:  $(dims(grid)), posang = $(posang(grid))"
        )
    )
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
