function testrot(m, g, gr, uv; alg=NFFTAlg(), atoli=5e-4, atolu=1e-6)
    Ut = uv .* ones(length(uv))'
    Vt = ones(length(uv)) .* uv'

    guv = UnstructuredDomain((U=vec(Ut), V=vec(Vt)))
    gfour = FourierDualDomain(g, guv, NFFTAlg())
    grfour = FourierDualDomain(gr, guv, NFFTAlg())

    mn = VLBISkyModels.NonAnalyticTest(m)
    img = intensitymap(m, g)
    imgr = intensitymap(m, gr)

    @test isapprox(img, regrid(imgr, g), atol=atoli)

    va = visibilitymap(m, gfour)
    vn = visibilitymap(mn, gfour)
    var = visibilitymap(m, grfour)
    vnr = visibilitymap(mn, grfour)

    @test isapprox(va, vn, atol=atolu)
    @test isapprox(va, var)
    @test isapprox(vn, vnr, atol=atolu)
end

@testset "Rotated Grid" begin
    g = imagepixels(6.0, 6.0, 256, 256)
    gr = imagepixels(6.0, 6.0, 256, 256; posang=π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5))
    testrot(m, g, gr, uv)
    testrot(m, g, gr, uv; alg=FFTAlg())
    testrot(m, g, gr, uv; alg=DFTAlg())
end

@testset "Rotated Grid and shifted" begin
    g = imagepixels(6.0, 6.0, 256, 256, 0.25, 0.25)
    gr = imagepixels(6.0, 6.0, 256, 256, 0.25, 0.25; posang=π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5))
    testrot(m, g, gr, uv)
    testrot(m, g, gr, uv; alg=FFTAlg())
    testrot(m, g, gr, uv; alg=DFTAlg())
end

@testset "Rotated Grid and shifted model" begin
    g = imagepixels(6.0, 6.0, 256, 256)
    gr = imagepixels(6.0, 6.0, 256, 256; posang=π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5), Shift(0.1, 0.1))
    testrot(m, g, gr, uv)
    testrot(m, g, gr, uv; alg=FFTAlg())
    testrot(m, g, gr, uv; alg=DFTAlg())
end

@testset "Polarized Rotated Grid" begin
    g = imagepixels(6.0, 6.0, 256, 256)
    gr = imagepixels(6.0, 6.0, 256, 256; posang=π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5))
    pm = PolarizedModel(m, 0.1 * m, -0.2 * m, 0.01 * m)
    testrot(pm, g, gr, uv)
    testrot(pm, g, gr, uv; alg=FFTAlg())
    testrot(pm, g, gr, uv; alg=DFTAlg())
end

@testset "Rotated Grid and shifted" begin
    g = imagepixels(6.0, 6.0, 256, 256, 0.25, 0.25)
    gr = imagepixels(6.0, 6.0, 256, 256, 0.25, 0.25; posang=π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5))
    pm = PolarizedModel(m, 0.1 * m, -0.2 * m, 0.01 * m)
    testrot(pm, g, gr, uv)
    testrot(pm, g, gr, uv; alg=FFTAlg())
    testrot(pm, g, gr, uv; alg=DFTAlg())
end

@testset "Rotated Grid and shifted model" begin
    g = imagepixels(6.0, 6.0, 256, 256)
    gr = imagepixels(6.0, 6.0, 256, 256; posang=π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5), Shift(0.1, 0.1))
    pm = PolarizedModel(m, 0.1 * m, -0.2 * m, 0.01 * m)
    testrot(pm, g, gr, uv)
    testrot(pm, g, gr, uv; alg=FFTAlg())
    testrot(pm, g, gr, uv; alg=DFTAlg())
end
