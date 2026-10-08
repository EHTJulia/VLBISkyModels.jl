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
    `intensity_point` on a [`MultiDomainImage`](@ref) evaluates the domain models only at the
    pixels under the kernel around the requested point, provided every family's
    `apply_param` returns a lazy `Base.Broadcasted`. Modifiers
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
base = IntensityMap(rand(64, 64), spatialgrid(10.0, 10.0, 64, 64))
dom  = PolySpectral((1.0,), 230.0e9)          # spectral index = 1
cimg = MultiDomainImage(base, BSplinePulse{3}(), dom)
```
""" MultiDomainImage

make_map(cimg::ContinuousImage) = IntensityMap(cimg.params, cimg.grid)
# The base image at the reference domain point; `size`, `flux`, etc. describe it.
make_map(cimg::MultiDomainImage) = IntensityMap(cimg.params.base, cimg.grid)

function Base.show(io::IO, img::ContinuousImage)
    pname = nameof(typeof(img.params))
    kname = nameof(typeof(img.kernel))
    return print(io, "ContinuousImage{$pname{$(eltype(img))}, $kname}($(size(img.grid)))")
end

ComradeBase.ispolarized(::Type{<:ContinuousImage{P}}) where {P} = _ispol(paramtype(P))
@inline _ispol(::Type{<:StokesParams}) = IsPolarized()
@inline _ispol(::Type{<:Number}) = NotPolarized()
ComradeBase.flux(m::ContinuousImage) = flux(make_map(m)) * flux(m.kernel)

function ComradeBase.stokes(cimg::ContinuousImage, v)
    return ContinuousImage(stokes(make_map(cimg), v), cimg.kernel)
end
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

ContinuousImage(img::IntensityMap, kernel) = ContinuousImage(baseimage(img), axisdims(img), kernel)

function ContinuousImage(im::AbstractArray, g::AbstractRectiGrid, kernel::AbstractModel)
    size(im) == size(g) || throw(
        DimensionMismatch(
            "The image array size $(size(im)) does not match the grid size $(size(g))."
        )
    )
    return ContinuousImage{typeof(im), typeof(g), typeof(kernel)}(im, g, kernel)
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


function MultiDomainImage(img::IntensityMap, kernel, domains...)
    ndims(axisdims(img)) == 2 || throw(
        ArgumentError(
            "MultiDomainImage expects an image on a 2D spatial grid, got " *
                "$(ndims(axisdims(img))) dimensions; the extra Fr/Ti structure " *
                "comes from `domains`."
        )
    )
    mdp = MultiDomainParams(baseimage(img), domains)
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

# The pixel values at `p`. A chain stays lazy, so a point evaluates only the pixels under
# the kernel.
@inline pointvalues(params::AbstractArray, p) = params
@inline function pointvalues(params::MultiDomainParams, p)
    chain = ComradeBase._applymodels(build_param(params.base, p), params.models, p)
    return Base.Broadcast.instantiate(chain)
end

pointeltype(vals) = eltype(vals)
pointeltype(bc::Base.Broadcast.Broadcasted) = Base.Broadcast.combine_eltypes(bc.f, bc.args)

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
    vals = pointvalues(m.params, p)

    X = g.X
    Y = g.Y
    sum = zero(pointeltype(vals))

    cs, rs = support_center(g, pg)
    wx, wy = support_halfwidths(g, rx, ry)
    ilo, ihi = firstindex(X), lastindex(X)
    jlo, jhi = firstindex(Y), lastindex(Y)

    # Fixed trip count: only the window center depends on the point. Off-grid taps are
    # masked, and their clamped index stays on the grid. `track_numbers = false`
    # keeps traced values out of the grid's `LinRange` length.
    @trace track_numbers = false for dj in -wy:wy
        @trace track_numbers = false for di in -wx:wx
            i = cs + di
            j = rs + dj
            inb = (ilo <= i) & (i <= ihi) & (jlo <= j) & (j <= jhi)
            ic = clamp(i, ilo, ihi)
            jc = clamp(j, jlo, jhi)
            dpi = (X = pg.X - rgetindex(X, ic), Y = pg.Y - rgetindex(Y, jc))
            k = intensity_point(ms, dpi)
            v = rgetindex(vals, ic, jc)
            sum += ifelse(inb, v * k, zero(sum))
            nothing
        end
    end
    return sum
end


function convolved(cimg::ContinuousImage, m::AbstractModel)
    return ContinuousImage(cimg.params, cimg.grid, convolved(cimg.kernel, m))
end
convolved(cimg::AbstractModel, m::ContinuousImage) = convolved(m, cimg)

@inline function visibilitymap_numeric(m::ContinuousImage, grid::FourierDualDomain)
    # We need to make sure that the grid is the same size as the image
    checkgrid(axisdims(m), imgdomain(grid))
    img = make_map(m)
    vis = applyft(forward_plan(grid), img)
    return IntensityMap(applypulse!(vis, m.kernel, grid), visdomain(grid))
end

# Shape with `len` on axis `i` and `1` elsewhere, e.g. `_axisshape(nfr, 3, Val(3)) = (1,1,nfr)`.
@inline _axisshape(len::Int, i::Int, ::Val{Nd}) where {Nd} = ntuple(j -> j == i ? len : 1, Val(Nd))

struct OnAxis{N}
    nd::Val{N}
end
function (a::OnAxis)(d, i)
    v = ComradeBase.basedim(d)
    return reshape(v, _axisshape(length(v), i, a.nd))
end

# Each dim's coordinates reshaped onto its own axis, e.g. `Fr => (1, 1, nfr)`, so that
# `build_param` broadcasts the whole cube at once. Must stay type stable: a `Union`-typed
# coordinate tuple breaks Enzyme's type analysis.
function _cubepoint(g::AbstractRectiGrid)
    ds = dims(g)
    nd = Val(length(ds))
    return NamedTuple{map(name, ds)}(map(OnAxis(nd), ds, ntuple(identity, nd)))
end

# The chain evaluated over the whole cube grid `g` in one broadcast.
function _paramcube(params::DomainParams, g::AbstractRectiGrid)
    raw = build_param(params, _cubepoint(g))
    sz = size(g)
    size(raw) == sz && return raw
    # A chain that does not read a dim of `g` is constant along it.
    cube = similar(raw, sz)
    cube .= raw
    return cube
end

function intensitymap_analytic(m::MultiDomainImage, dims::AbstractRectiGrid)
    out = allocate_imgmap(m, dims)
    intensitymap_analytic!(out, m)
    return out
end

# The image's own spatial grid crossed with the non-spatial dims of `gout`.
function _cubegrid(gspat::AbstractRectiGrid, gout::AbstractRectiGrid)
    return gridproduct(gspat, DD.otherdims(gout, (X, Y))...)
end

# Grid evaluation of any `ContinuousImage` goes through the slice resampler: a plain image
# is the one-slice case, a stored cube resamples slice by slice, and a multidomain image
# (below) materializes its cube first. Composite and modified models still evaluate per
# point through the generic path.
function intensitymap_analytic!(img::IntensityMap, m::ContinuousImage)
    _resample!(img, make_map(m), m.kernel)
    return nothing
end

function intensitymap_analytic!(img::IntensityMap, m::MultiDomainImage)
    gcube = _cubegrid(m.grid, axisdims(img))
    _resample!(img, IntensityMap(_paramcube(m.params, gcube), gcube), m.kernel)
    return nothing
end

"""
    _resample!(img::IntensityMap, src::IntensityMap, kernel)

Writes into `img` the kernel resampling of the pixel map `src` onto the spatial grid of `img`,
for every index of the dims beyond `X` and `Y` (`Fr`, `Ti`); a dim of `img` that `src` lacks
is replicated, matched by name. A `StokesMap` is resampled one Stokes component at a time.

A `Pulse` factors as `κ(ΔX)κ(ΔY)`, so on grids that share a position angle the resampling is
two 1D passes, along `X` and then `Y`. Each output pixel along an axis reads a fixed number
of source pixels (the kernel's taps) with precomputed weights (`AxisTaps`), and each
pass is one broadcast that sums the taps' weighted gathers, fused so that only the pass's
result is allocated. The same code runs on CPU arrays and under Reactant, where it lowers to
gathers and elementwise operations. Other kernels and rotated pairs of grids evaluate each
output pixel over its support window instead (CPU only).
"""
function _resample!(img::IntensityMap, src::IntensityMap, kernel)
    extra = DD.otherdims(img, (X, Y))
    _check_resample(img, src, extra)
    gsrc = spatialdims(axisdims(src))
    gspat = spatialdims(axisdims(img))
    (kernel isa Pulse && posang(gsrc) == posang(gspat)) || return _resample_window!(img, src, kernel)
    tx = AxisTaps(kernel, gsrc.X, gspat.X)
    ty = AxisTaps(kernel, gsrc.Y, gspat.Y)
    R = _pass(_pass(baseimage(src), tx, Val(1)), ty, Val(2))
    DD.broadcast_dims!(identity, img, DD.DimArray(R, (dims(gspat)..., DD.otherdims(src, (X, Y))...)))
    return nothing
end

function _resample!(img::StokesMap, src::StokesMap, kernel)
    I, Q, U, V = _stokesviews(img)
    _resample!(I, stokes(src, :I), kernel)
    _resample!(Q, stokes(src, :Q), kernel)
    _resample!(U, stokes(src, :U), kernel)
    _resample!(V, stokes(src, :V), kernel)
    return nothing
end

"""
    AxisTaps(kernel::Pulse, xs, xo)

The taps of `kernel` for resampling an axis with pixel centers `xs` onto `xo`: output `a`
reads source pixels `index[:, a]` with weights `weight[:, a]`. A tap off the grid reads the
nearest pixel with weight 0.
"""
struct AxisTaps{W, I <: AbstractMatrix{Int}, M <: AbstractMatrix}
    index::I
    weight::M
end

function AxisTaps(kernel::Pulse, xs::AbstractVector, xo::AbstractVector)
    dx = step(xs)
    r = radialextent(kernel)
    w = 2 * ceil(Int, r)
    lo = floor.(Int, (xo .- first(xs)) ./ dx .+ 1 .- r) .+ 1
    raw = lo' .+ (0:(w - 1))
    index = clamp.(raw, firstindex(xs), lastindex(xs))
    k = κ.(Ref(kernel), (xo' .- xs[index]) ./ dx) .* (step(xo) / dx)
    weight = ifelse.(raw .== index, k, zero(eltype(k)))
    return AxisTaps{w, typeof(index), typeof(weight)}(index, weight)
end

struct Tap{D, A, T}
    S::A
    taps::T
end
Tap{D}(S, taps) where {D} = Tap{D, typeof(S), typeof(taps)}(S, taps)
(f::Tap{1})(t) = Broadcast.broadcasted(*, view(f.taps.weight, t, :), selectdim(f.S, 1, view(f.taps.index, t, :)))
(f::Tap{2})(t) = Broadcast.broadcasted(*, reshape(view(f.taps.weight, t, :), 1, :), selectdim(f.S, 2, view(f.taps.index, t, :)))

function _pass(S, taps::AxisTaps{W}, ::Val{D}) where {W, D}
    return Broadcast.materialize(Broadcast.broadcasted(+, ntuple(Tap{D}(S, taps), Val(W))...))
end

function _resample_window!(img::IntensityMap, src::IntensityMap, kernel)
    extra = DD.otherdims(img, (X, Y))
    shared = DD.commondims(src, extra)
    pos = map(Base.Fix1(DD.dimnum, extra), shared)
    gsrc = spatialdims(axisdims(src))
    gspat = spatialdims(axisdims(img))
    ex = ComradeBase.executor(gspat)
    for I in CartesianIndices(map(length, extra))
        i = Tuple(I)
        out = _sliceat(img, map(rebuild, extra, i))
        sl = _sliceat(src, map(rebuild, shared, getindex.(Ref(i), pos)))
        ComradeBase.intensitymap_analytic_executor!(
            out, ContinuousImage(baseimage(sl), gsrc, kernel), ex
        )
    end
    return nothing
end

_sliceat(A, ::Tuple{}) = A
_sliceat(A, ds::Tuple) = view(A, ds...)

function _check_resample(img, src, extra)
    srcextra = DD.otherdims(src, (X, Y))
    shared = DD.commondims(src, extra)
    length(shared) == length(srcextra) || throw(
        DimensionMismatch(
            "the image has dims $(map(name, srcextra)) but the output grid has only $(map(name, extra)) beyond X and Y"
        )
    )
    DD.comparedims(shared, dims(img, shared))
    return shared
end

function visibilitymap_numeric(m::MultiDomainImage, grid::FourierDualDomain)
    gimg = imgdomain(grid)
    checkgrid(axisdims(m), spatialdims(gimg))
    mfimg = IntensityMap(_paramcube(m.params, gimg), gimg)
    vis = applyft(forward_plan(grid), mfimg)
    return IntensityMap(applypulse!(vis, m.kernel, grid), visdomain(grid))
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
    pvis = _densevis(vis)
    pvis .*= ComradeBase.pointbroadcasted(Base.Fix1(visibility_point, mp), guv)
    return vis
end

# Polarized visibilities are scaled through their dense storage: Enzyme does not see through
# a broadcast into the `StokesParams` view. Other arrays are scaled directly, since under
# Reactant a reshaped visibility map is a lazy `ReshapedArray` whose parent is flat.
_densevis(vis::FieldDimArray) = parent(vis)
_densevis(vis) = vis

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
    pvbase .*= ComradeBase.pointbroadcasted(UVScale(mbase, t, uc), p)
    return nothing
end

struct UVScale{M, T, S}
    model::M
    transform::T
    scale::S
end
(f::UVScale)(p) = last(modify_uv(f.model, f.transform, p, f.scale))
