using RecipesBase
using Printf

@recipe function f(image::IntensityMap; uvscale = rad2μas)
    #Define some constants
    #Construct the image grid in μas

    (; X, Y) = image
    xitr, yitr = uvscale.((X, Y))
    x0, x1 = extrema(xitr)
    y0, y1 = extrema(yitr)
    tickfontsize --> 11
    guidefontsize --> 14

    tickfontsize --> 11
    guidefontsize --> 14
    if typeof(image) <: IntensityMap{<:StokesParams}

        # get the mean linear pol
        maxI = maximum(stokes(image, :I))

        layout --> (2, 2)
        size --> (500 * 2, 400 * 2)
        @series begin
            subplot := 1
            seriestype := :heatmap
            seriescolor --> :afmhot
            aspect_ratio --> 1
            bar_width --> 0
            xlims --> (x0, x1)
            ylims --> (y0, y1)
            z = baseimage(stokes(image, :I))'
            title --> "Stokes I"
            seriestype := :heatmap
            #fontfamily --> "sans serif"
            xflip --> true
            widen := false
            linecolor --> :black
            tick_direction --> :out
            #colorrange-->(0.0, maxI)

            collect(xitr), collect(yitr), z
        end
        @series begin
            subplot := 2
            seriestype := :heatmap
            seriescolor --> :seismic
            aspect_ratio --> 1
            bar_width --> 0
            xlims --> (x0, x1)
            ylims --> (y0, y1)
            z = baseimage(stokes(image, :Q))'
            title --> "Stokes Q"
            #fontfamily --> "sans serif"
            xflip --> true
            widen := false
            linecolor --> :black
            tick_direction --> :out
            clims --> (-maxI / 2, maxI / 2)

            collect(xitr), collect(yitr), z
        end
        @series begin
            subplot := 3
            seriestype := :heatmap
            xaxis --> "ΔRA  (μas)"
            yaxis --> "ΔDEC (μas)"
            seriescolor --> :seismic
            aspect_ratio --> 1
            bar_width --> 0
            xlims --> (x0, x1)
            ylims --> (y0, y1)
            z = baseimage(stokes(image, :U))'
            title --> "Stokes U"
            seriestype := :heatmap
            #fontfamily --> "sans serif"
            xflip --> true
            widen := false
            linecolor --> :black
            tick_direction --> :out
            clims --> (-maxI / 2, maxI / 2)

            collect(xitr), collect(yitr), z
        end
        @series begin
            subplot := 4
            seriestype := :heatmap
            xaxis --> "ΔRA  (μas)"
            yaxis --> "ΔDEC (μas)"
            seriescolor --> :seismic
            aspect_ratio --> 1
            bar_width --> 0
            xlims --> (x0, x1)
            ylims --> (y0, y1)
            z = baseimage(stokes(image, :V))'
            title --> "Stokes V"
            seriestype := :heatmap
            #fontfamily --> "sans serif"
            colorbar_title --> "Jy/px²"
            xflip --> true
            widen := false
            linecolor --> :black
            tick_direction --> :out
            clims --> (-maxI / 2, maxI / 2)

            collect(xitr), collect(yitr), z
        end
    else
        seriestype := :heatmap
        xaxis --> "ΔRA  (μas)"
        yaxis --> "ΔDEC (μas)"
        seriescolor --> :afmhot
        aspect_ratio --> 1
        bar_width --> 0
        xlims --> (x0, x1)
        ylims --> (y0, y1)
        z = baseimage(image)'
        title --> "Stokes I"
        seriestype := :heatmap
        #fontfamily --> "sans serif"
        colorbar_title --> "Jy/px²"
        xflip --> true
        widen := false
        linecolor --> :black
        tick_direction --> :out

        collect(xitr), collect(yitr), z
    end
end

@recipe function f(
        m::AbstractModel; uvscale = rad2μas,
        fovx = 2 * radialextent(m), fovy = 2 * radialextent(m),
        nx = 512, ny = 512,
        x0 = 0.0, y0 = 0.0
    )
    grid = imagepixels(fovx, fovy, nx, ny, x0, y0)
    image = intensitymap(m, grid)
    (; X, Y) = image
    xitr, yitr = uvscale.((X, Y))
    x0, x1 = uvscale.(extrema(xitr))
    y0, y1 = uvscale.(extrema(yitr))

    #Define some constants
    #Construct the image grid in μas
    fovx, fovy = uvscale.(values(fieldofview(image)))
    xitr, yitr = uvscale.((image.X, image.Y))

    tickfontsize --> 11
    guidefontsize --> 14
    if ispolarized(typeof(m)) === IsPolarized()
        # get the mean linear pol
        maxI = maximum(stokes(image, :I))

        layout --> (2, 2)
        size --> (500 * 2, 400 * 2)
        @series begin
            subplot := 1
            seriestype := :heatmap
            seriescolor --> :afmhot
            aspect_ratio --> 1
            bar_width --> 0
            xlims --> (x0, x1)
            ylims --> (y0, y1)
            z = ComradeBase.baseimage(stokes(image, :I))'
            title --> "Stokes I"
            seriestype := :heatmap
            #fontfamily --> "sans serif"
            xflip --> true
            widen := false
            linecolor --> :black
            tick_direction --> :out
            colorrange --> (0.0, maxI)

            collect(xitr), collect(yitr), z
        end
        @series begin
            subplot := 2
            seriestype := :heatmap
            seriescolor --> :berlin
            aspect_ratio --> 1
            bar_width --> 0
            xlims --> (x0, x1)
            ylims --> (y0, y1)
            z = ComradeBase.baseimage(stokes(image, :Q))'
            title --> "Stokes Q"
            seriestype := :heatmap
            #fontfamily --> "sans serif"
            xflip --> true
            widen := false
            linecolor --> :black
            tick_direction --> :out
            clims --> (-maxI / 2, maxI / 2)

            collect(xitr), collect(yitr), z
        end
        @series begin
            subplot := 3
            seriestype := :heatmap
            xaxis --> "ΔRA  (μas)"
            yaxis --> "ΔDEC (μas)"
            seriescolor --> :berlin
            aspect_ratio --> 1
            bar_width --> 0
            xlims --> (x0, x1)
            ylims --> (y0, y1)
            z = ComradeBase.baseimage(stokes(image, :U))'
            title --> "Stokes U"
            seriestype := :heatmap
            #fontfamily --> "sans serif"
            xflip --> true
            widen := false
            linecolor --> :black
            tick_direction --> :out
            clims --> (-maxI / 2, maxI / 2)

            collect(xitr), collect(yitr), z
        end
        @series begin
            subplot := 4
            seriestype := :heatmap
            xaxis --> "ΔRA  (μas)"
            yaxis --> "ΔDEC (μas)"
            seriescolor --> :berlin
            aspect_ratio --> 1
            bar_width --> 0
            xlims --> (x0, x1)
            ylims --> (y0, y1)
            z = ComradeBase.baseimage(stokes(image, :V))'
            title --> "Stokes V"
            seriestype := :heatmap
            #fontfamily --> "sans serif"
            colorbar_title --> "Jy/px²"
            xflip --> true
            widen := false
            linecolor --> :black
            tick_direction --> :out
            clims --> (-maxI / 2, maxI / 2)

            collect(xitr), collect(yitr), z
        end
    else
        seriestype := :heatmap
        xaxis --> "ΔRA  (μas)"
        yaxis --> "ΔDEC (μas)"
        seriescolor --> :afmhot
        aspect_ratio --> 1
        bar_width --> 0
        xlims --> (x0, x1)
        ylims --> (y0, y1)
        z = parent(image)'
        title --> "Stokes I"
        seriestype := :heatmap
        #fontfamily --> "sans serif"
        colorbar_title --> "Jy/px²"
        xflip --> true
        widen := false
        linecolor --> :black
        tick_direction --> :out

        collect(xitr), collect(yitr), z
    end
end
