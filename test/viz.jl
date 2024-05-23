@testset "Makie Visualizations" begin
    m = PolarizedModel(Gaussian(), 0.1*Gaussian(), 0.25*Gaussian(), 0.1*Gaussian())
    g = imagepixels(10.0, 10.0, 256, 256)
    img = intensitymap(m, g)

    CM.heatmap(img)
    CM.image(img)
    CM.image(stokes(img, :I))
    CM.image(g, m)
    CM.image(g.X, g.Y, m)
    CM.heatmap(g, m)
    CM.heatmap(g.X, g.Y, m)


    display(imageviz(stokes(img, :I)))
    display(imageviz(img))
    display(imageviz(img, plot_total=false))
end
