module VLBISkyModelsMakieExt

using VLBISkyModels
if isdefined(Base, :get_extension)
    using Makie
    using AxisKeys
else
    using ..Makie
    using ..AxisKeys
end

import VLBISkyModels: polimage, polimage!, imageviz


function Makie.convert_arguments(::SurfaceLike, img::IntensityMap{T, 2}) where {T}
    (;X, Y) = img
    return rad2μas(X), rad2μas(Y), VLBISkyModels.baseimage(img)
end

function Makie.convert_arguments(::DiscreteSurface, img::IntensityMap{T, 2}) where {T}
    (;X, Y) = img
    return rad2μas(X), rad2μas(Y), VLBISkyModels.baseimage(img)
end

function Makie.convert_arguments(::SurfaceLike, x::AbstractVector, y::AbstractVector, m::VLBISkyModels.AbstractModel)
    img = intensitymap(m, GriddedKeys((X=x, Y=y)))
    return rad2μas(x), rad2μas(y), VLBISkyModels.baseimage(img)
end

function Makie.convert_arguments(::SurfaceLike, g::VLBISkyModels.AbstractDims, m::VLBISkyModels.AbstractModel)
    img = intensitymap(m, g)
    return rad2μas(g.X), rad2μas(g.Y), VLBISkyModels.baseimage(img)
end


function polintensity(s::StokesParams)
    return sqrt(s.Q^2 + s.U^2 + s.V^2)
end

"""
    polimage(img::IntensityMap{<:StokesParams};
                colormap = :bone,
                colorrange = Makie.automatic,
                pcolorrange=Makie.automatic,
                pcolormap=Reverse(:jet1),
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


## Generic Attributes
$(Makie.ATTRIBUTES)


!!! warning
    The polarized plotting is intrinsically defined using astronomer/EHT polarization conventions
    This means that in order to have the polarization ticks plotted in a way that makes sense
    you need to have `xreversed=true` when defining your axis.

"""
Makie.@recipe(PolImage, img) do scene
    Makie.Attributes(;
        colormap = Reverse(:bone),
        colorrange = Makie.automatic,
        pcolorrange = Makie.automatic,
        pcolormap = Makie.automatic,
        colorscale = identity,
        alpha = 1.0,
        nan_color = Makie.RGBAf(0,0,0,0),
        nvec = 30,
        min_frac = 0.1,
        min_pol_frac = 0.1,
        length_norm = 1.0,
        lowclip = Makie.automatic,
        highclip = Makie.automatic,
        plot_total = true
    )
end

Makie.plottype(::SpatialIntensityMap{<:StokesParams}) = PolImage{<:Tuple{SpatialIntensityMap{<:StokesParams}}}

function polparams(x, y, s, xmin, ptot)
    ptot && return ellipse_params(x, y, s, xmin)
    return lin_params(x, y, s, xmin)
end

function ellipse_params(x, y, s, xmin)
    e = polellipse(s)
    p = Point2(rad2μas(x), rad2μas(y))
    len =  Vec2(max(e.b, e.a/10) , e.a)/2
    col =  polintensity(s)/s.I*sign(s.V)
    rot = evpa(s)
    return p, len, col, rot
end

function lin_params(x, y, s, xmin)
    l = linearpol(s)
    p = Point2(rad2μas(x), rad2μas(y))
    len =  Vec2f(xmin/1.5, xmin/1.5 + abs(l))
    col =  abs(l)/s.I
    rot = evpa(s)
    return p, len, col, rot
end

function Makie.plot!(plot::PolImage{<:Tuple{IntensityMap{<:StokesParams}}})
    @extract plot (img,)
    img = plot[:img]

    imgI = lift(img) do img
        return stokes(img, :I)
    end

    # plot the stokes I image
    # cr = lift(plot.colorrange, imgI) do crange, imgI
    #     (crange == Makie.automatic) && return (0.0, maximum(imgI)*1.01)
    #     return crange
    # end

    image!(plot, imgI;
        colormap=plot.colormap,
        colorscale = plot.colorscale,
        colorrange = plot.colorrange,
        alpha = plot.alpha,
        nan_color = plot.nan_color,
        lowclip=plot.lowclip
    )

    points = lift(img, plot.nvec,
                 plot.min_frac, plot.min_pol_frac,
                 plot.length_norm,
                 plot.plot_total) do img, nvec, Icut, pcut, length_norm, ptot
        Xvec = range(img.X[begin+1], img.X[end-1], length=nvec)
        Yvec = range(img.Y[begin+1], img.Y[end-1], length=nvec)

        maxI = maximum((stokes(img, :I)))
        if ptot
            maxL = maximum(polintensity, img)
        else
            maxL = maximum(x->abs(linearpol(x)), img)
        end

        dx = max(rad2μas.(values(fieldofview(img)))...)

        p   = Point2[]
        len = Vec2[]
        col = eltype(stokes(img, :I))[]
        rot = eltype(stokes(img, :I))[]

        lenmul = 10*dx/nvec/maxL*length_norm

        for y in Yvec
            for x in Xvec
                s = img[X=Near(x), Y=Near(y)]
                psi, leni, coli, roti = polparams(x, y, s, maxL/length_norm/5, ptot)

                if ptot
                    pol = polintensity(s)
                else
                    pol = abs(linearpol(s))
                end

                if (pol/maxL > pcut && s.I/maxI > Icut)
                    push!(p, psi)
                    push!(len, lenmul .* leni)
                    push!(col, coli)
                    push!(rot, roti)
                end
            end
        end

        return p, len, col, rot
    end

    p   = lift(x->getindex(x, 1), points)
    len = lift(x->getindex(x, 2), points)
    col = lift(x->getindex(x, 3), points)
    rot = lift(x->getindex(x, 4), points)

    pc = lift(plot.pcolorrange, plot.plot_total, img) do pc, pt, img
        (pc != Makie.automatic) && return pc
        if pt
            maxp = min(maximum(x->polintensity(x)/(x.I+eps()), img), 0.9090909)
            return (-1.01*maxp, 1.01*maxp)
        else
            maxp = min(maximum(x->abs(linearpol(x))/(x.I+eps()), img), 0.9090909)
            return (0.0, 1.01*maxp)
        end
    end
    m = lift(plot.plot_total) do pt
            pt && return '𝝾'
            return '⋅'
    end

    pcm = lift(plot.pcolormap, plot.plot_total) do pc, pt
        (pc != Makie.automatic) && return pc
        if pt
            return :diverging_bkr_55_10_c35_n256
        else
            return :rainbow1
        end
    end


    scatter!(plot, p;
        marker=m,
        markersize = len,
        markerspace = :data,
        rotations = rot,
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
function imageviz(img::IntensityMap; scale_length = fieldofview(img).X/4, kwargs...)
    dkwargs = Dict(kwargs)
    if eltype(img) <: Real
        res = get(dkwargs, :resolution, (600, 500))
    else
        res = get(dkwargs, :resolution, (610, 600))
    end
    delete!(dkwargs, :resolution)
    fig = Figure(;resolution = res)
    ax = Axis(fig[1,1], xreversed=true, aspect=DataAspect())
    hidedecorations!(ax)

    dxdy = prod(rad2μas.(values(pixelsizes(img))))
    imguas = img./dxdy

    pl = _imgviz!(fig, ax, imguas; scale_length, dkwargs...)
    return pl
end

function _imgviz!(fig, ax, img::IntensityMap{<:Real}; scale_length=fieldofview(img).X/4, kwargs...)
    colorrange_default = (minimum(img), maximum(img))
    dkwargs = Dict(kwargs)
    crange = get(dkwargs, :colorrange, colorrange_default)
    delete!(dkwargs, :colorrange)
    cmap = get(dkwargs, :colormap, :afmhot)
    delete!(dkwargs, :colormap)



    hm = image!(ax, img; colorrange=crange, colormap=cmap, dkwargs...)

    color = Makie.to_colormap(cmap)[end]
    add_scalebar!(ax, img, scale_length, color)


    Colorbar(fig[1,2], hm, label="Brightness (Jy/μas)")
    colgap!(fig.layout, 15)

    return Makie.FigureAxisPlot(fig, ax, hm)
end

function _imgviz!(fig, ax, img::IntensityMap{<:StokesParams}; scale_length=fieldofview(img).X/4, kwargs...)
    colorrange_default = (0.0, maximum(stokes(img, :I)))
    dkwargs = Dict(kwargs)
    crange = get(dkwargs, :colorrange, colorrange_default)
    delete!(dkwargs, :colorrange)
    cmap = get(dkwargs, :colormap, Reverse(:bone))
    delete!(dkwargs, :colormap)
    delete!(dkwargs, :resolution)

    pt = get(dkwargs, :plot_total, true)

    hm = polimage!(ax, img; colorrange=crange, colormap=cmap, dkwargs...)

    color = Makie.to_colormap(cmap)[end]
    add_scalebar!(ax, img, scale_length, color)

    Colorbar(fig[1,2], getfield(hm, :plots)[1], label="Brightness (Jy/μas)")

    if pt
        plabel = "Signed Fractional Total Polarization sign(V)|mₜₒₜ|"
    else
        plabel = "Fractional Linear Polarization |m|"
    end
    Colorbar(fig[2, 1], getfield(hm, :plots)[2], tellwidth=true, tellheight=true, label=plabel, vertical=false, flipaxis=false,)
    colgap!(fig.layout, 5)
    rowgap!(fig.layout, 5)

    return Makie.FigureAxisPlot(fig, ax, hm)
end

function add_scalebar!(ax, img, scale_length, color)
    fovx, fovy = map(rad2μas, fieldofview(img))
    dx, dy = map(rad2μas, pixelsizes(img))
    sl = rad2μas(scale_length)
    barx = [-fovx/2 + fovx/32, -fovx/2 + fovx/32 + sl]
    bary = fill(-fovy/2 + fovy/32, 2)

    lines!(ax, -barx, bary, color=color)
    text!(ax, -(barx[1] + (barx[2]-barx[1])/2), bary[1]+fovy/64;
            text = "$(round(Int, sl)) μas",
            align=(:center, :bottom), color=color)
end


end