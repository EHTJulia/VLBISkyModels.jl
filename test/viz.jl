@testset "Makie Visualizations" begin
    m = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.25 * Gaussian(), 0.1 * Gaussian())
    g = imagepixels(10.0, 10.0, 256, 256)
    img = intensitymap(m, g)

    for f in ((CM.heatmap), (CM.image), (CM.spy), (CM.contour), (CM.contourf))
        f(g, m)
        f(g.X, g.Y, m)
        f(img)
        f(stokes(img, :Q))
    end

    display(imageviz(stokes(img, :I)))
    display(imageviz(img))
    display(imageviz(img; plot_total = false))
end

@testset "polimage markers" begin
    m = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.25 * Gaussian(), 0.1 * Gaussian())
    img = intensitymap(m, imagepixels(10.0, 10.0, 128, 128))
    spacing = (last(img.X) - first(img.X)) / 30

    fig = CM.Figure()
    ax = CM.Axis(fig[1, 1])
    p = polimage!(ax, img; plot_total = false, tick_width = 0.3)
    @test !isempty(p.len[])
    @test all(l -> l[1] ≈ 0.3 * spacing, p.len[])
    # The image fills the axis, as a heatmap does.
    @test ax.xautolimitmargin[] == (0, 0)
    @test ax.yautolimitmargin[] == (0, 0)

    pe = polimage!(CM.Axis(fig[1, 2]), img; tick_width = 0.3)
    @test !isempty(pe.len[])
    @test all(l -> l[1] >= 0.3 * spacing * (1 - sqrt(eps())), pe.len[])

    @test_throws "tick_width must be positive" polimage(img; tick_width = 0)

    # The border is a copy of each marker, larger by `marker_border` tick widths on each side.
    pd = polimage!(CM.Axis(fig[1, 3]), img; plot_total = false)
    b = 0.12 * 0.25 * spacing
    @test pd.showborder[]
    @test pd.strokew[] == 0
    @test all(((l, bl),) -> bl ≈ l .+ 2b, zip(pd.len[], pd.blen[]))
    @test !polimage(img; marker_border = 0).plot.showborder[]
    @test_throws "marker_border must be nonnegative" polimage(img; marker_border = -0.1)

    # The deprecated attributes warn when the plot is computed and draw the border as a Makie
    # stroke.
    pw = polimage(img; marker_width = 0.5).plot
    @test_logs (:warn, r"marker_width. is deprecated") pw.stroked[]
    @test pw.stroked[] && !pw.showborder[] && pw.strokew[] == 0.75
    ps = polimage(img; marker_strokewidth = 0.3).plot
    @test_logs (:warn, r"marker_strokewidth. is deprecated") ps.stroked[]
    @test ps.stroked[] && !ps.showborder[] && ps.strokew[] == 0.3
    @test_throws "marker_width must be in (0, 1]" polimage(img; marker_width = 1.5).plot.stroked[]

    # imageviz's polarization colorbar follows the markers, not their borders.
    fv = imageviz(img)
    cb = only(filter(x -> x isa CM.Colorbar && !x.vertical[], fv.figure.content))
    @test all(isapprox.(cb.limits[], fv.plot.pc[]; rtol = 1.0e-6))
end
