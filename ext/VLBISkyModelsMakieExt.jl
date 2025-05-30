module VLBISkyModelsMakieExt

using VLBISkyModels
using Makie
using DimensionalData
using ComradeBase: basedim
using StaticArrays

const DD = DimensionalData

const DDM = Base.get_extension(DD, :DimensionalDataMakie)

import VLBISkyModels: polimage, polimage!, imageviz

# function Makie.convert_arguments(::GridBased, img::SpatialIntensityMap)
#     (;X, Y) = img
#     return (X), (Y), parent(img)
# end

# function Makie.expand_dimensions(::CellGrid, img::SpatialIntensityMap)
#     (; X, Y) = img
#     return (X), (Y), parent(img)
# end

function Makie.expand_dimensions(::NoConversion, img::SpatialIntensityMap)
    # (; X, Y) = img
    return (img,)
end

# function Makie.expand_dimensions(::ImageLike, img::SpatialIntensityMap)
#     (; X, Y) = img
#     rX, rY = ((X, Y))
#     return first(rX) .. last(rX), first(rY) .. last(rY), parent(img)
# end

# function Makie.convert_arguments(::CellGrid, x, y, img::AbstractMatrix{<:StokesParams})
#     return x, y, stokes(img, :I)
# end
# function Makie.convert_arguments(::VertexGrid, x, y, img::AbstractMatrix{<:StokesParams})
#     return x, y, stokes(img, :I)
# end
# function Makie.convert_arguments(::ImageLike, x, y, img::AbstractMatrix{<:StokesParams})
#     return x, y, stokes(img, :I)
# end

# function Makie.expand_dimensions(g::VertexGrid,
#                                  img::SpatialIntensityMap{<:StokesParams})
#     imgI = stokes(img, :I)
#     return Makie.expand_dimensions(g, imgI)
# end

# function Makie.expand_dimensions(g::CellGrid,
#                                  img::SpatialIntensityMap{<:StokesParams})
#     imgI = stokes(img, :I)
#     return Makie.expand_dimensions(g, imgI)
# end

function Makie.expand_dimensions(
        g::ImageLike,
        img::SpatialIntensityMap{<:StokesParams}
    )
    return img
end

const VectorDim = Union{AbstractVector, DD.Dimension}

function Makie.expand_dimensions(
        t::CellGrid, x::VectorDim, y::VectorDim,
        m::VLBISkyModels.AbstractModel
    )
    img = intensitymap(m, RectiGrid((X(x), Y(y))))
    return Makie.expand_dimensions(t, img)
end

function Makie.expand_dimensions(
        t::VertexGrid, x::VectorDim, y::VectorDim,
        m::VLBISkyModels.AbstractModel
    )
    img = intensitymap(m, RectiGrid((X(x), Y(y))))
    return Makie.expand_dimensions(t, img)
end

function Makie.expand_dimensions(
        t::ImageLike, x::VectorDim, y::VectorDim,
        m::VLBISkyModels.AbstractModel
    )
    img = intensitymap(m, RectiGrid((X(x), Y(y))))
    return Makie.expand_dimensions(t, img)
end

function Makie.expand_dimensions(
        t::CellGrid, g::VLBISkyModels.AbstractRectiGrid,
        m::VLBISkyModels.AbstractModel
    )
    img = intensitymap(m, g)
    return Makie.expand_dimensions(t, img)
end

function Makie.expand_dimensions(
        t::VertexGrid, g::VLBISkyModels.AbstractRectiGrid,
        m::VLBISkyModels.AbstractModel
    )
    img = intensitymap(m, g)
    return Makie.expand_dimensions(t, img)
end

function Makie.expand_dimensions(
        t::ImageLike, g::VLBISkyModels.AbstractRectiGrid,
        m::VLBISkyModels.AbstractModel
    )
    img = intensitymap(m, g)
    return Makie.expand_dimensions(t, img)
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

The stokes `I` image will be plotted with the attributes
  - `colormap`
  - `colorrange`
  - `alpha`
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

## Attributes
  - `colormap`: The colormap of the stokes `I` image. The default is `:bone`.
  - `colorrange`: The color range of the stokes `I` image. The default is `(0, maximum(stokes(img, :I)))`
  - `pcolorrange:` The color range for the polarized image
  - `pcolormap`: The colormap used for fractional total/linear polarization markers.
  - `nvec`: The number of polarization vectors to plot
  - `min_frac`: Any markers with `I < min_frac*maximum(I))` will not be plotted
  - `min_pol_frac`: Any markers with `P < min_frac*maximum(P))` where `P` is the total/linear polarization flux
                will not be plotted.
  - `length_norm`: Specifies the normalization used for the ticks. The default is that the pixel
                    with the largest polarization intensity will have a tick length = 10x the
                    pixel separation. For an image with a maximum polarized intensity of 10Jy/μas²
                    and pixel spacing of 1μas the marker length will be equal to 10μas.
  - `plot_total`: If true plot the total polarization. If false only plot the linear polarization.
  - `adjust_length`: If true the tick lengths will be adjusted according to the total polarized flux.
                     If false the tick length is fixed and the size can be adjusted with `length_norm`.

## Generic Attributes
$(Makie.ATTRIBUTES)


!!! warning
    The polarized plotting is intrinsically defined using astronomer/EHT polarization conventions
    This means that in order to have the polarization ticks plotted in a way that makes sense
    you need to have `xreversed=true` when defining your axis.

"""
Makie.@recipe(PolImage, img) do scene
    return Makie.Attributes(;
        colormap = :grayC,
        colorrange = Makie.automatic,
        pcolorrange = Makie.automatic,
        pcolormap = Makie.automatic,
        colorscale = identity,
        alpha = 1.0,
        nan_color = Makie.RGBAf(0, 0, 0, 0),
        nvec = 30,
        min_frac = 0.1,
        min_pol_frac = 0.1,
        length_norm = 1.0,
        adjust_length = false,
        lowclip = Makie.automatic,
        highclip = Makie.automatic,
        plot_total = true
    )
end

# # We need this because DimensionalData tries to be too dang smart
function Makie.convert_arguments(
        ::Type{<:PolImage}, img::IntensityMap{<:StokesParams, 2},
        args...
    )
    return (img,)
end

# function Makie.MakieCore.conversion_trait(P::Type{<:PolImage})
#     # @info "HERE"
#     return P
# end

function Makie.convert_arguments(::Type{<:PolImage}, img::IntensityMap)
    return (IntensityMap(img),)
end

# function Makie.plottype(::SpatialIntensityMap{<:StokesParams})
#     return PolImage{<:Tuple{<:IntensityMap{<:StokesParams}}}
# end

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

    imgI = lift(img) do img
        return parent(stokes(img, :I))
    end

    Xo = @lift $img.X
    Yo = @lift $img.Y

    pa = @lift -ComradeBase.posang(axisdims($img))

    # plot the stokes I image
    # cr = lift(plot.colorrange, imgI) do crange, imgI
    #     (crange == Makie.automatic) && return (0.0, maximum(imgI)*1.01)
    #     return crange
    # end
    hm = heatmap!(
        plot, Xo, Yo, imgI;
        colormap = plot.colormap,
        colorscale = plot.colorscale,
        colorrange = plot.colorrange,
        alpha = plot.alpha,
        nan_color = plot.nan_color,
        lowclip = plot.lowclip,
    )

    rotate!(hm, pa[])

    points = lift(
        img, plot.nvec,
        plot.min_frac, plot.min_pol_frac,
        plot.length_norm,
        plot.plot_total
    ) do img, nvec, Icut, pcut, length_norm, ptot
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
                    push!(rot, roti)
                end
            end
        end

        return p, len, col, rot, lenmul
    end

    p = lift(x -> getindex(x, 1), points)
    len = lift(x -> getindex(x, 2), points)
    col = lift(x -> getindex(x, 3), points)
    rot = lift(x -> -getindex(x, 4), points)
    lenmul = lift(x -> getindex(x, 5), points)

    pc = lift(plot.pcolorrange, plot.plot_total, img) do pc, pt, img
        (pc != Makie.automatic) && return pc
        if pt
            maxp = min(maximum(x -> polintensity(x) / (x.I + eps()), img), 0.9090909)
            return (-1.01 * maxp, 1.01 * maxp)
        else
            maxp = min(maximum(x -> abs(linearpol(x)) / (x.I + eps()), img), 0.9090909)
            return (0.0, 1.01 * maxp)
        end
    end
    m = lift(plot.plot_total) do pt
        pt && return '∘'
        return '⋅'
    end

    pcm = lift(plot.pcolormap, plot.plot_total) do pc, pt
        (pc != Makie.automatic) && return pc
        if pt
            return Makie.Reverse(:RdBu)
        else
            return :rainbow1
        end
    end

    len2 = lift(plot.adjust_length, len, plot.length_norm) do al, l, lm
        (!al && length(l) > 0) && return maximum(l)
        return l
    end
    scatter!(
        plot, p;
        marker = m,
        markersize = len2,
        markerspace = :data,
        rotation = rot,
        colorrange = pc,
        color = col,
        colormap = pcm,
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

    trim!(fig.layout)
    # xlims!(ax, (last(img.X)), (first(img.X)))
    # ylims!(ax, (first(img.Y)), (last(img.Y)))

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
    # xlims!(ax, (last(img.X)), (first(img.X)))
    # ylims!(ax, (first(img.Y)), (last(img.Y)))
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

# Horrible hack until I can figure out how to prevent DD from taking over
# my recipes
function DDM._surface2(A::IntensityMap, plotfunc, attributes, replacements)
    # Array/Dimension manipulation
    A1 = DDM._prepare_for_makie(A, replacements)
    lookup_attributes, newdims = DDM._split_attributes(A1)
    A2 = DDM._restore_dim_names(set(A1, map(Pair, newdims, newdims)...), A, replacements)
    P = Makie.Plot{plotfunc}
    PTrait = Makie.conversion_trait(P, A2)
    # We define conversions by trait for all of the explicitly overridden functions,
    # so we can just use the trait here.
    args = Makie.convert_arguments(PTrait, A2)

    # if status === true
    #     args = converted
    # else
    #     args = Makie.convert_arguments(P, converted...)
    # end

    # Plot attribute generation
    dx, dy = DD.dims(A2)
    user_attributes = Makie.Attributes(;
        transformation = (;
            rotation = -ComradeBase.posang(axisdims(A)),
        ),
        interpolate = false,
        attributes...
    )
    plot_attributes = Makie.Attributes(;
        axis = (;
            xlabel = DD.label(dx),
            ylabel = DD.label(dy),
            title = DD.refdims_title(A),
        ),
    )
    merged_attributes = merge(user_attributes, plot_attributes, lookup_attributes)

    return A1, A2, args, merged_attributes
end

end
