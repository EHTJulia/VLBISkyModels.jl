@testset "Makie Visualizations" begin
    m = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.25 * Gaussian(), 0.1 * Gaussian())
    g = imagepixels(10.0, 10.0, 256, 256)
    img = intensitymap(m, g)
    imgsa = IntensityMap(
        StructArray{eltype(img)}(map(k -> parent(stokes(img, k)), (I = :I, Q = :Q, U = :U, V = :V))),
        g
    )
    mfrac = sqrt(0.1^2 + 0.25^2 + 0.1^2)
    lfrac = sqrt(0.1^2 + 0.25^2)

    for f in ((CM.heatmap), (CM.image), (CM.spy), (CM.contour), (CM.contourf))
        f(g, m)
        f(g.X, g.Y, m)
        f(img)
        f(imgsa)
        f(stokes(img, :Q))
    end

    for pimg in (img, imgsa)
        _, _, pl = polimage(pimg)
        @test !isempty(pl.p[])
        @test all(c -> c ≈ mfrac, pl.col[])
        @test pl.imgI[] == stokes(img, :I)

        _, _, pl = polimage(pimg; plot_total = false)
        @test all(c -> c ≈ lfrac, pl.col[])

        for fap in (imageviz(pimg), imageviz(pimg; plot_total = false))
            @test fap isa CM.Makie.FigureAxisPlot
            @test size(CM.colorbuffer(fap.figure)) > (0, 0)
        end
    end
    fap = imageviz(stokes(img, :I))
    @test fap isa CM.Makie.FigureAxisPlot
    @test size(CM.colorbuffer(fap.figure)) > (0, 0)
end
