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

# A raster on a rotated grid is the same image as that raster on an unrotated grid, turned by
# the grid's position angle: rendering it on rotated grid points must reproduce the unrotated
# rendering at the matching points, and its visibilities at (U, V) must equal the unrotated
# image's visibilities at the (U, V) of the grid's axes.
@testset "Rotated ContinuousImage" begin
    pa = π / 7
    g = imagepixels(6.0, 4.0, 48, 32, 0.3, -0.2)
    gr = imagepixels(6.0, 4.0, 48, 32, 0.3, -0.2; posang = pa)
    m = modify(Gaussian(), Stretch(0.8, 0.4), Shift(0.5, 0.2))
    raster = parent(intensitymap(m, g))
    pulse = BSplinePulse{3}()
    cimg = ContinuousImage(raster, g, pulse)
    cimgr = ContinuousImage(raster, gr, pulse)

    # Output grids finer than the raster, so the samples fall between the pixel centers.
    go = imagepixels(6.0, 4.0, 96, 64, 0.3, -0.2)
    gor = imagepixels(6.0, 4.0, 96, 64, 0.3, -0.2; posang = pa)
    img = intensitymap(cimg, go)
    imgr = intensitymap(cimgr, gor)
    @test parent(imgr) ≈ parent(img)
    @test sum(imgr) ≈ flux(cimgr) rtol = 1.0e-3

    uv = range(-1.0, 1.0; length = 7)
    U = vec(uv .* ones(length(uv))')
    V = vec(ones(length(uv)) .* uv')
    rm = ComradeBase.rotmat(gr)'
    Ug = VLBISkyModels._rotatex.(U, V, Ref(rm))
    Vg = VLBISkyModels._rotatey.(U, V, Ref(rm))
    guv = UnstructuredDomain((U = U, V = V))
    guvg = UnstructuredDomain((U = Ug, V = Vg))
    for alg in (NFFTAlg(), DFTAlg(), FFTAlg())
        vr = visibilitymap(cimgr, FourierDualDomain(gr, guv, alg))
        v = visibilitymap(cimg, FourierDualDomain(g, guvg, alg))
        @test parent(vr) ≈ parent(v) rtol = 1.0e-5
    end
end
