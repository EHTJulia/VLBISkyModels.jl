@testset "Makie Visualizations" begin
    m = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.25 * Gaussian(), 0.1 * Gaussian())
    g = imagepixels(10.0, 10.0, 256, 256)
    img = intensitymap(m, g)

    CM.heatmap(stokes(img, :I))
    CM.image(stokes(img, :I))
    CM.image(g, stokes(m, :I))
    CM.image(g.X, g.Y, stokes(m, :I))
    CM.heatmap(g, stokes(m, :Q))
    CM.heatmap(g.X, g.Y, stokes(m, :Q))

    display(imageviz(stokes(img, :I)))
    display(imageviz(img))
    display(imageviz(img; plot_total=false))
end
