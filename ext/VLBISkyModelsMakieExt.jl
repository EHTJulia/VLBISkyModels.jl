module VLBISkyModelsMakieExt

using VLBISkyModels
if isdefined(Base, :get_extension)
    using Makie
    using AxisKeys
else
    using ..Makie
    using ..AxisKeys
end

import VLBISkyModels: polimage, polimage!


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
  - `min_frac`: Any markers with `P < min_frac*maximum(P))` where `P` is the total/linear polarization flux
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
        pcolormap=Reverse(:jet1),
        colorscale = identity,
        alpha = 1.0,
        nan_color = Makie.RGBAf(0,0,0,0),
        nvec = 30,
        min_frac = 0.1,
        min_pol_frac = 0.2,
        length_norm = 1.0,
        lowclip = Makie.automatic,
        highclip = Makie.automatic,
        plot_total = true
    )
end

Makie.plottype(::SpatialIntensityMap{<:StokesParams}) = PolImage{<:Tuple{SpatialIntensityMap{<:StokesParams}}}

function polparams(x, y, s, ptot)
    ptot && return ellipse_params(x, y, s)
    return lin_params(x, y, s)
end

function ellipse_params(x, y, s)
    e = polellipse(s)
    p = Point2(rad2μas(x), rad2μas(y))
    len =  Vec2(e.b + 0.02*e.a, e.a)/2
    col =  polintensity(s)/s.I*sign(s.V)
    rot = evpa(s)
    return p, len, col, rot
end

function lin_params(x, y, s)
    l = linearpol(s)
    p = Point2(rad2μas(x), rad2μas(y))
    len =  Vec2f(abs(l)/10, abs(l))
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

        dx = max(rad2μas.(values(pixelsizes(img)))...)

        p   = Point2[]
        len = Vec2[]
        col = eltype(stokes(img, :I))[]
        rot = eltype(stokes(img, :I))[]

        lenmul = 10*dx/maxL*length_norm

        for y in Yvec
            for x in Xvec
                s = img[X=Near(x), Y=Near(y)]
                psi, leni, coli, roti = polparams(x, y, s, ptot)

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

    pc = lift(plot.pcolorrange, img) do pc, img
        pc != Makie.automatic && return pc
        maxp = maximum(x->polintensity(x)/x.I, img)
        return (-1.01*maxp, 1*01maxp)
    end
    m = lift(plot.plot_total) do pt
            pt && return '𝝾'
            return '⋅'
    end
    scatter!(plot, p;
        marker=m,
        markersize = len,
        markerspace = :data,
        rotations = rot,
        colorrange = pc,
        color = col,
        colormap = plot.pcolormap,
    )

end

function imgviz!(img::IntensityMap; kwargs...)
    res = get(kwargs, "resolution", (550, 500))
    fig = Figure(;resoluion = (550, 500))
    ax = Axis(fig[1,1], xreversed=true, aspect=1)

    dxdy = prod(rad2μas(values(pixelsizes(img))))
    imguas = img./dxdy

    imviz!(ax, imguas)
    colorrange_default = (0.0, maximum(imguas))

end

function imgviz!(ax::Axis, img::IntensityMap; kwargs...)

end


end
