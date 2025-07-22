module VLBISkyModelsMakieExt

using VLBISkyModels
using Makie
using DimensionalData
using ComradeBase: basedim
using StaticArrays

const DD = DimensionalData

const DDM = Base.get_extension(DD, :DimensionalDataMakie)

import VLBISkyModels: polimage, polimage!, imageviz

Makie.convert_single_argument(m::ComradeBase.UnstructuredMap) = parent(m)


function Makie.convert_arguments(
        P::Type{Image}, img::IntensityMap{<:StokesParams};
        xdim = nothing, ydim = nothing
    )
    return Makie.convert_arguments(P, DimArray(stokes(img, :I)); xdim, ydim)
end

function Makie.convert_arguments(
        P::Type{Heatmap}, img::IntensityMap{<:StokesParams};
        xdim = nothing, ydim = nothing
    )
    return Makie.convert_arguments(P, DimArray(stokes(img, :I)); xdim, ydim)
end

function Makie.convert_arguments(
        P::Type{Contour}, img::IntensityMap{<:StokesParams};
        xdim = nothing, ydim = nothing
    )
    return Makie.convert_arguments(P, DimArray(stokes(img, :I)); xdim, ydim)
end

function Makie.convert_arguments(
        P::Type{Contourf}, img::IntensityMap{<:StokesParams};
        xdim = nothing, ydim = nothing
    )
    return Makie.convert_arguments(P, DimArray(stokes(img, :I)); xdim, ydim)
end

function Makie.convert_arguments(
        P::Type{Spy}, img::IntensityMap{<:StokesParams};
        xdim = nothing, ydim = nothing
    )
    return Makie.convert_arguments(P, DimArray(stokes(img, :I)); xdim, ydim)
end


function DDM.axis_attributes(::Type{P}, dd::IntensityMap; xdim, ydim) where {P <: Union{Heatmap, Image, Surface, Contour, Contourf, Contour3d, Spy}}
    dims_axes = DDM.obs_f(i -> DDM.get_dimensions_of_makie_axis(i, (xdim, ydim)), dd)
    lookup_attributes = DDM.get_axis_ticks(Makie.to_value(dims_axes))

    return merge(
        lookup_attributes,
        (;
            xlabel = DDM.obs_f(i -> DD.label(i[1]), dims_axes),
            ylabel = DDM.obs_f(i -> DD.label(i[2]), dims_axes),
            title = DDM.obs_f(DD.refdims_title, dd),
            xreversed = true,
        )
    )
end

const VectorDim = Union{AbstractVector, DD.Dimension}

function Makie.convert_arguments(
        t::ImageLike, g::VLBISkyModels.AbstractRectiGrid,
        m::VLBISkyModels.AbstractModel
    )
    img = intensitymap(m, g)
    return convert_arguments(t, img)
end

function Makie.convert_arguments(
        t::Union{ImageLike, CellGrid}, g::VLBISkyModels.AbstractRectiGrid,
        m::VLBISkyModels.AbstractModel
    )
    img = intensitymap(m, g)
    return convert_arguments(t, img)
end

function Makie.convert_arguments(
        t::Union{ImageLike, CellGrid}, X, Y,
        m::VLBISkyModels.AbstractModel
    )
    g = RectiGrid((; X, Y))
    img = intensitymap(m, g)
    return convert_arguments(t, img)
end

function Makie.convert_arguments(
        t::Type{T}, g::VLBISkyModels.AbstractRectiGrid,
        m::VLBISkyModels.AbstractModel
    ) where {T <: Union{Spy, Contour, Contourf}}
    img = intensitymap(m, g)
    return convert_arguments(t, img)
end

function Makie.convert_arguments(
        t::Type{T}, X, Y,
        m::VLBISkyModels.AbstractModel
    ) where {T <: Union{Spy, Contour, Contourf}}
    g = RectiGrid((; X, Y))
    img = intensitymap(m, g)
    return convert_arguments(t, img)
end


function polintensity(s::StokesParams)
    return sqrt(s.Q^2 + s.U^2 + s.V^2)
end

"""
    polimage(img::IntensityMap{<:StokesParams};
                colormap = :grayC,
                colorrange = Makie.automatic,
                pcolorrange=Makie.automatic,
                pcolormap=Makie.Reverse(:RdBu),
                nvec = 30,
                min_frac = 0.1,
                min_pol_frac=0.2,
                length_norm=1.0,
                plot_total=true)

Plot a polarized intensity map using the image `img`.

The plot follows the conventions from [EHTC M87 Paper VII](https://iopscience.iop.org/article/10.3847/2041-8213/abe71d).

  - `colorscale`

The polarized image will consist of a set. The attribute `plot_total` changes what polarized
quantities are considered.

**If `plot_total = true`**

 - The total polarization will be considered and the markers will be given by ellipses.
 - The orientation of the ellipse is equal to the EVPA.
 - The area of the ellipse is proportional to `|V|²`.
 - The semi-major axis is related to the total polarized intensity times the `length_norm`.
 - The color of the ellipse is given by the fractional total polarization times the
sign of Stokes `V`.

**If `plot_total = false`**

 - Only the linear polarization is considered and the markers will be ticks.
 - The orientation of the ticks is equal to the EVPA.
 - The length of the ticks is equal to the total linear polarized intensity,
   i.e. `√(Q² + U²)` times the `length_norm`.
 - The color of the tick is given by the fractional linear polarization.


!!! warning
    The polarized plotting is intrinsically defined using astronomer/EHT polarization conventions
    This means that in order to have the polarization ticks plotted in a way that makes sense
    you need to have `xreversed=true` when defining your axis.

"""
Makie.@recipe PolImage (img::IntensityMap{<:StokesParams},) begin
    "Sets the color range of the polarization"
    pcolorrange = Makie.automatic
    "Polarized color map"
    pcolormap = Makie.automatic
    "Number of polarization vectors to plot"
    nvec = 30
    "Minimal flux fraction to show ticks"
    min_frac = 0.1
    "Minimal polarization fraction to show ticks"
    min_pol_frac = 0.1
    "Renormalize the tick length"
    length_norm = 1.0
    "Adjust the length of the ticks"
    adjust_length = false
    "If true plot the total polarization, if false only plot the linear polarization"
    plot_total = true
    Makie.mixin_generic_plot_attributes()...
    Makie.mixin_colormap_attributes()...
end


function polparams(x, y, s, xmin, ptot)
    ptot && return ellipse_params(x, y, s, xmin)
    return lin_params(x, y, s, xmin)
end

function ellipse_params(x, y, s, xmin)
    e = polellipse(s)
    p = Point2((x), (y))
    len = Vec2(max(e.b, e.a / 20), e.a) / 2
    col = polintensity(s) / s.I * sign(s.V)
    rot = evpa(s)
    return p, len, col, rot
end

function lin_params(x, y, s, xmin)
    l = linearpol(s)
    p = Point2((x), (y))
    len = Vec2f(xmin / 1.5, xmin / 1.5 + abs(l))
    col = abs(l) / s.I
    rot = evpa(s)
    return p, len, col, rot
end

function Makie.plot!(plot::PolImage{<:Tuple{<:IntensityMap{<:StokesParams}}})
    # @extract plot (X, Y, img)

    img = plot[1]

    map!(plot.attributes, [:img], :imgI) do img
        return stokes(img, :I)
    end

    pa = @lift -ComradeBase.posang(axisdims($img))


    map!(
        plot.attributes,
        [:img, :nvec, :min_frac, :min_pol_frac, :length_norm, :plot_total, :adjust_length],
        [:p, :len, :col, :rot, :lenmul]
    ) do img, nvec, Icut, pcut, length_norm, ptot, adjust_length

        X = img.X
        Y = img.Y
        Xvec = range(X[begin + 1], X[end - 1]; length = nvec)
        Yvec = range(Y[begin + 1], Y[end - 1]; length = nvec)

        maxI = maximum((stokes(img, :I)))
        if ptot
            maxL = maximum(polintensity, img)
        else
            maxL = maximum(x -> abs(linearpol(x)), img)
        end

        fovx = last(X) - first(X)
        fovy = last(Y) - first(Y)
        dx = max(fovx, fovy)

        p = Point2{typeof(dx)}[]
        len = Vec2{typeof(dx)}[]
        col = eltype(stokes(img, :I))[]
        rot = eltype(stokes(img, :I))[]

        lenmul = 10 * dx / nvec / maxL .* length_norm
        dimg = img
        rm = rotmat(axisdims(img))
        for y0 in Yvec
            for x0 in Xvec
                s = dimg[X = Near(x0), Y = Near(y0)]
                xyr = rm * SVector(x0, y0)
                x = xyr[1]
                y = xyr[2]
                psi, leni, coli, roti = polparams(x, y, s, maxL ./ length_norm ./ 5, ptot)

                if ptot
                    pol = polintensity(s)
                else
                    pol = abs(linearpol(s))
                end

                if (pol / maxL > pcut && s.I / maxI > Icut)
                    push!(p, psi)
                    push!(len, lenmul .* leni)
                    push!(col, coli)
                    push!(rot, -roti)
                end
            end
        end

        if (!adjust_length && length(len) > 0)
            len2 = maximum(len)
        else
            len2 = len
        end


        return p, len2, col, rot, lenmul
    end

    # p = lift(x -> getindex(x, 1), points)
    # len = lift(x -> getindex(x, 2), points)
    # col = lift(x -> getindex(x, 3), points)
    # rot = lift(x -> -getindex(x, 4), points)
    # lenmul = lift(x -> getindex(x, 5), points)

    map!(plot.attributes, [:pcolorrange, :plot_total, :img], :pc) do pcr, pt, img
        if pcr != Makie.automatic
            return pcr
        end
        if pt
            maxp = min(maximum(x -> polintensity(x) / (x.I + eps()), img), 0.9090909)
            return (-1.01 * maxp, 1.01 * maxp)
        else
            maxp = min(maximum(x -> abs(linearpol(x)) / (x.I + eps()), img), 0.9090909)
            return (0.0, 1.01 * maxp)
        end
    end

    map!(plot.attributes, [:plot_total], :mrk) do pt
        pt ? '∘' : '⋅'
    end

    map!(plot.attributes, [:pcolormap, :plot_total], :pcm) do pcm, pt
        if pcm != Makie.automatic
            return pcm
        end
        if pt
            return Makie.Reverse(:RdBu)
        else
            return :rainbow1
        end
    end

    # len2 = lift(plot.adjust_length, len, plot.length_norm) do al, l, lm
    #     (!al && length(l) > 0) && return maximum(l)
    #     return l
    # end

    hm = heatmap!(
        plot, plot.attributes, plot.imgI
    )

    rotate!(hm, pa[])


    scatter!(
        plot, plot.p;
        marker = plot.mrk,
        markersize = plot.len,
        markerspace = :data,
        rotation = plot.rot,
        colorrange = plot.pc,
        color = plot.col,
        colormap = plot.pcm,
    )


    return plot
end

"""
    imageviz(img::IntensityMap; scale_length = fieldofview(img.X/4), kwargs...)

A default image visualization for a `IntensityMap`.

**If `eltype(img) <: Real`** i.e. an image of a single stokes parameter
this will plot the image with a colorbar in units of Jy/μas². The plot will
accept any `kwargs` that are a supported by the `Makie.Image` type an can
be queried by typing `?image` in the REPL

**If `eltype(img) <: StokesParams`** i.e. full polarized image
this will use `polimage`. The plot will
accept any `kwargs` that are a supported by the `PolImage` type an can
be queried by typing `?polimage` in the REPL.

!!! tip
    To customize the image, i.e. specify a specific axis we recommend to use
    `image` and `polimage` directly.

"""
function imageviz(
        img::IntensityMap;
        scale_length = rad2μas(fieldofview(img).X / 4),
        backgroundcolor = nothing,
        kwargs...
    )
    dkwargs = Dict(kwargs)
    if eltype(img) <: Real
        res = get(dkwargs, :size, (625, 500))
        cmap = get(dkwargs, :colormap, :inferno)
    else
        res = get(dkwargs, :size, (640, 600))
        cmap = get(dkwargs, :colormap, :grayC)
    end

    bkgcolor = isnothing(backgroundcolor) ? Makie.to_colormap(cmap)[begin] : backgroundcolor

    delete!(dkwargs, :size)
    fig = Figure(; size = res)
    ax = Axis(
        fig[1, 1]; xreversed = true, aspect = DataAspect(), tellheight = true,
        tellwidth = true, backgroundcolor = bkgcolor
    )
    hidedecorations!(ax)

    dxdy = prod(rad2μas.(values(pixelsizes(img))))
    gua = rebuild(axisdims(img); dims = (X(rad2μas(img.X)), Y(rad2μas(img.Y))))
    imguas = IntensityMap(
        parent(img) ./ dxdy,
        gua
    )
    pl = _imgviz!(fig, ax, imguas; scale_length, dkwargs...)
    resize_to_layout!(fig)
    return pl
end

function _imgviz!(
        fig, ax, img::IntensityMap{<:Real}; scale_length = fieldofview(img).X / 4,
        kwargs...
    )
    colorrange_default = (minimum(img), maximum(img))
    dkwargs = Dict(kwargs)
    crange = get(dkwargs, :colorrange, colorrange_default)
    delete!(dkwargs, :colorrange)
    cmap = get(dkwargs, :colormap, :inferno)
    delete!(dkwargs, :colormap)

    hm = heatmap!(ax, img; colorrange = crange, colormap = cmap, dkwargs...)
    rotate!(hm, -ComradeBase.posang(axisdims(img)))

    color = Makie.to_colormap(cmap)[end]
    add_scalebar!(ax, img, scale_length, color)

    Colorbar(fig[1, 2], hm; label = "Brightness (Jy/μas²)", tellheight = true)
    colgap!(fig.layout, 15)

    x1, y1 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), first(img.Y))
    x2, y2 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), last(img.Y))
    x3, y3 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), last(img.Y))
    x4, y4 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), first(img.Y))

    xl = min(x1, x2, x3, x4)
    xu = max(x1, x2, x3, x4)
    yl = min(y1, y2, y3, y4)
    yu = max(y1, y2, y3, y4)

    # Flip x l and u for astronomer conventions
    xlims!(ax, (xu, xl))
    ylims!(ax, (yl, yu))
    trim!(fig.layout)

    return Makie.FigureAxisPlot(fig, ax, hm)
end

function _imgviz!(
        fig, ax, img::IntensityMap{<:StokesParams};
        scale_length = fieldofview(img).X / 4, kwargs...
    )
    colorrange_default = (0.0, maximum(stokes(img, :I)))
    dkwargs = Dict(kwargs)
    crange = get(dkwargs, :colorrange, colorrange_default)
    delete!(dkwargs, :colorrange)
    cmap = get(dkwargs, :colormap, :grayC)
    delete!(dkwargs, :colormap)
    delete!(dkwargs, :size)

    pt = get(dkwargs, :plot_total, true)

    hm = polimage!(ax, img; colorrange = crange, colormap = cmap, dkwargs...)

    color = Makie.to_colormap(cmap)[end]
    add_scalebar!(ax, img, scale_length, color)

    Colorbar(
        fig[1, 2], getfield(hm, :plots)[1]; label = "Brightness (Jy/μas²)",
        tellheight = true
    )

    if pt
        plabel = "Signed Fractional Total Polarization sign(V)|mₜₒₜ|"
    else
        plabel = "Fractional Linear Polarization |m|"
    end
    Colorbar(
        fig[2, 1], getfield(hm, :plots)[2]; tellwidth = true, tellheight = true,
        label = plabel, vertical = false, flipaxis = false
    )
    colgap!(fig.layout, 15)
    rowgap!(fig.layout, 15)
    trim!(fig.layout)

    x1, y1 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), first(img.Y))
    x2, y2 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), last(img.Y))
    x3, y3 = rotmat(axisdims(img)) * VLBISkyModels.SVector(last(img.X), last(img.Y))
    x4, y4 = rotmat(axisdims(img)) * VLBISkyModels.SVector(first(img.X), first(img.Y))

    xl = min(x1, x2, x3, x4)
    xu = max(x1, x2, x3, x4)
    yl = min(y1, y2, y3, y4)
    yu = max(y1, y2, y3, y4)

    xlims!(ax, (xu, xl))
    ylims!(ax, (yl, yu))
    return Makie.FigureAxisPlot(fig, ax, hm)
end

function add_scalebar!(ax, img, scale_length, color)
    fovx, fovy = fieldofview(img)
    x0 = (last(img.X))
    y0 = (first(img.Y))

    sl = (scale_length)
    barx = [x0 - fovx / 32, x0 - fovx / 32 - sl]
    bary = fill(y0 + fovy / 32, 2)

    lines!(ax, barx, bary; color = color)
    return text!(
        ax, (barx[1] + (barx[2] - barx[1]) / 2), bary[1] + fovy / 64;
        text = "$(round(Int, sl)) μas",
        align = (:center, :bottom), color = color
    )
end


end
