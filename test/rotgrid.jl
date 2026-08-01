function testrot(m, g, gr, uv; alg = NFFTAlg(), atoli = 5.0e-4, atolu = 1.0e-6)
    Ut = uv .* ones(length(uv))'
    Vt = ones(length(uv)) .* uv'

    guv = UnstructuredDomain((U = vec(Ut), V = vec(Vt)))
    gfour = FourierDualDomain(g, guv, NFFTAlg())
    grfour = FourierDualDomain(gr, guv, NFFTAlg())

    mn = VLBISkyModels.NonAnalyticTest(m)
    img = intensitymap(m, g)
    imgr = intensitymap(m, gr)

    @test isapprox(img, regrid(imgr, g), atol = atoli)

    va = visibilitymap(m, gfour)
    vn = visibilitymap(mn, gfour)
    var = visibilitymap(m, grfour)
    vnr = visibilitymap(mn, grfour)

    @test isapprox(va, vn, atol = atolu)
    @test isapprox(va, var)
    return @test isapprox(vn, vnr, atol = atolu)
end

@testset "Rotated Grid" begin
    g = imagepixels(6.0, 6.0, 256, 256)
    gr = imagepixels(6.0, 6.0, 256, 256; posang = π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5))
    testrot(m, g, gr, uv)
    testrot(m, g, gr, uv; alg = FFTAlg())
    testrot(m, g, gr, uv; alg = DFTAlg())
end

@testset "Rotated Grid and shifted" begin
    g = imagepixels(6.0, 6.0, 256, 256, 0.25, 0.25)
    gr = imagepixels(6.0, 6.0, 256, 256, 0.25, 0.25; posang = π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5))
    testrot(m, g, gr, uv)
    testrot(m, g, gr, uv; alg = FFTAlg())
    testrot(m, g, gr, uv; alg = DFTAlg())
end

@testset "Rotated Grid and shifted model" begin
    g = imagepixels(6.0, 6.0, 256, 256)
    gr = imagepixels(6.0, 6.0, 256, 256; posang = π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5), Shift(0.1, 0.1))
    testrot(m, g, gr, uv)
    testrot(m, g, gr, uv; alg = FFTAlg())
    testrot(m, g, gr, uv; alg = DFTAlg())
end

@testset "Polarized Rotated Grid" begin
    g = imagepixels(6.0, 6.0, 256, 256)
    gr = imagepixels(6.0, 6.0, 256, 256; posang = π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5))
    pm = PolarizedModel(m, 0.1 * m, -0.2 * m, 0.01 * m)
    testrot(pm, g, gr, uv)
    testrot(pm, g, gr, uv; alg = FFTAlg())
    testrot(pm, g, gr, uv; alg = DFTAlg())
end

@testset "Rotated Grid and shifted" begin
    g = imagepixels(6.0, 6.0, 256, 256, 0.25, 0.25)
    gr = imagepixels(6.0, 6.0, 256, 256, 0.25, 0.25; posang = π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5))
    pm = PolarizedModel(m, 0.1 * m, -0.2 * m, 0.01 * m)
    testrot(pm, g, gr, uv)
    testrot(pm, g, gr, uv; alg = FFTAlg())
    testrot(pm, g, gr, uv; alg = DFTAlg())
end

@testset "Rotated Grid and shifted model" begin
    g = imagepixels(6.0, 6.0, 256, 256)
    gr = imagepixels(6.0, 6.0, 256, 256; posang = π / 4)

    uv = range(-2.0, 2.0)

    m = modify(Gaussian(), Stretch(0.25, 0.5), Shift(0.1, 0.1))
    pm = PolarizedModel(m, 0.1 * m, -0.2 * m, 0.01 * m)
    testrot(pm, g, gr, uv)
    testrot(pm, g, gr, uv; alg = FFTAlg())
    testrot(pm, g, gr, uv; alg = DFTAlg())
end

@testset "ContinuousImage on a rotated grid" begin
    # A rotated grid holds the same brightness distribution as an unrotated one, sampled at
    # the rotated point, so rotating both the grid and the sample point must agree. This
    # pins the support window and the pulse offset to the same frame: computing one on the
    # grid's axes and the other on the sky's silently returns zero away from the centre.
    n = 64
    g = imagepixels(6.0, 6.0, n, n)
    gr = imagepixels(6.0, 6.0, n, n; posang = π / 4)
    b = rand(n, n)
    R = ComradeBase.rotmat(gr)

    c = ContinuousImage(IntensityMap(b, g), BSplinePulse{3}())
    cr = ContinuousImage(IntensityMap(b, gr), BSplinePulse{3}())

    for (fx, fy) in ((3.3, 4.7), (10.5, -8.2), (0.0, 0.0), (-15.0, 20.0))
        px = fx * step(g.X)
        py = fy * step(g.Y)
        v = R * [px, py]
        @test ComradeBase.intensity_point(c, (X = px, Y = py)) ≈
            ComradeBase.intensity_point(cr, (X = v[1], Y = v[2]))
    end

    # the same must hold for a multidomain image, which shares the evaluation path
    ν₀ = 230.0e9
    mu = MultiDomainImage(IntensityMap(b, g), BSplinePulse{3}(), PolySpectral(1.5, ν₀))
    mr = MultiDomainImage(IntensityMap(b, gr), BSplinePulse{3}(), PolySpectral(1.5, ν₀))
    for (fx, fy) in ((3.3, 4.7), (10.5, -8.2))
        px = fx * step(g.X)
        py = fy * step(g.Y)
        v = R * [px, py]
        @test ComradeBase.intensity_point(mu, (X = px, Y = py, Fr = 2ν₀)) ≈
            ComradeBase.intensity_point(mr, (X = v[1], Y = v[2], Fr = 2ν₀))
    end
end
