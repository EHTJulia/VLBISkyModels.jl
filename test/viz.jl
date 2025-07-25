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
