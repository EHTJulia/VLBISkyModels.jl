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

function Makie.convert_arguments(::SurfaceLike, x::AbstractVector, y::AbstractVector, m::VLBISkyModels.AbstractModel)
    img = intensitymap(m, GriddedKeys((X=x, Y=y)))
    return x, y, VLBISkyModels.baseimage(img)
end

function Makie.convert_arguments(::SurfaceLike, g::VLBISkyModels.AbstractDims, m::VLBISkyModels.AbstractModel)
    img = intensitymap(m, g)
    return g.X, g.Y, VLBISkyModels.baseimage(img)
end

function polarization_ellipse(s::StokesParams)
    l = linearpol(s)
    p = polintensity(s)
    a = sqrt((p + abs(l))/2)
    b = sqrt((p - abs(l))/2)
    θ = angle(l)/2
    sn = sign(s.V)
    return (;a, b, θ, sn)
end

function polintensity(s::StokesParams)
    return sqrt(s.Q^2 + s.U^2 + s.V^2)
end

Makie.@recipe(PolImage, img) do scene
    Makie.Attributes(;
        colormap = :bone,
        colorrange = Makie.automatic,
        pcolorrange = Makie.automatic,
        pcolormap=Reverse(:jet1),
        colorscale = identity,
        alpha = 1.0,
        nan_color = Makie.RGBAf(0,0,0,0),
        nvec = 30,
        min_frac = 0.1,
        min_pol_frac = 0.2,
        length_norm = 1.2,
        plot_total = true
    )
end

Makie.plottype(::SpatialIntensityMap{<:StokesParams}) = PolImage{<:Tuple{SpatialIntensityMap{<:StokesParams}}}

function polparams(x, y, s, ptot)
    ptot && return ellipse_params(x, y, s)
    return lin_params(x, y, s)
end

function ellipse_params(x, y, s)
    e = polarization_ellipse(s)
    p = Point2f(rad2μas(x), rad2μas(y))
    len =  Vec2f(e.b+1e-2, e.a)
    col =  polintensity(s)/s.I*sign(s.V)
    rot = evpa(s)
    return p, len, col, rot
end

function lin_params(x, y, s)
    l = linearpol(s)
    p = Point2f(rad2μas(x), rad2μas(y))
    len =  Vec2f(5.0, abs(l))
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
    )

    points = lift(img, plot.nvec,
                 plot.min_frac, plot.min_pol_frac,
                 plot.length_norm,
                 plot.plot_total) do img, nvec, Icut, pcut, length_norm, ptot
        Xvec = range(first(img.X), last(img.X), length=nvec)
        Yvec = range(first(img.Y), last(img.Y), length=nvec)

        maxI = maximum((stokes(img, :I)))
        maxL = maximum(x->abs(linearpol(x)), img)
        dxdy = prod(rad2μas.(values(pixelsizes(img))))

        p   = Point2f[]
        len = Vec2f[]
        col = eltype(stokes(img, :I))[]
        rot = eltype(stokes(img, :I))[]

        for y in Yvec
            for x in Xvec
                s = img[X=Near(x), Y=Near(y)]
                suas = s/dxdy*1e6
                psi, leni, coli, roti = polparams(x, y, suas, ptot)
                if (abs(linearpol(s))/maxL > pcut && s.I/maxI > Icut)
                    push!(p, psi)
                    push!(len, length_norm .* leni)
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
            pt && return '𝗢'
            return '.'
    end
    scatter!(plot, p;
        marker=m,
        markersize = len,
        markerspace = :data,
        strokewidth=0.1,
        strokecolor=:black,
        rotations = rot,
        colorrange = pc,
        color = col,
        colormap = plot.pcolormap,
    )

end


end
