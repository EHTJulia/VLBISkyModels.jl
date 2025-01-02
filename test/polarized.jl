function testpol(m, uv)
    g = imagepixels(5.0, 5.0, 128, 128)
    img = intensitymap(m, g)
    img2 = zero(img)
    intensitymap!(img2, m)

    @test all(==(1), img .≈ img2)

    @inferred visibilitymap(m, uv)

    v = visibilitymap(m, uv)

    Plots.plot(m)
    Plots.plot(img)
    polimage(img)
    polimage(img; plot_total=false)
    Plots.closeall()
    return v
end

@testset "Polarized Analytic" begin
    m = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian())
    g = imagepixels(10.0, 10.0, 512, 512)
    s = map(length, dims(g))
    tsp = TaylorSpectral(0.1, 1.0, 230.0)
    mf= PolarizedModel(Gaussian(), tsp * Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian())
    @test ComradeBase.intensity_point(mf, (;X=0.1, Y=0.0, Fr=230.0)) ≈
          ComradeBase.intensity_point(m, (;X=0.1, Y=0.0, Fr=230.0))

    @test ComradeBase.visibility_point(mf, (;U=0.1, V=0.0, Fr=230.0)) ≈
          ComradeBase.visibility_point(m, (;U=0.1, V=0.0, Fr=230.0))

    u = fftshift(fftfreq(length(g.X), 1 / step(g.X)))
    uv = RectiGrid((U(u), V(-u)))
    uvus = UnstructuredDomain((U=randn(32), V=randn(32)))
    foo(x) = sum(norm,
                 VLBISkyModels.visibilitymap_analytic(PolarizedModel(Gaussian(),
                                                                     x[1] * Gaussian(),
                                                                     x[2] * Gaussian(),
                                                                     x[3] * Gaussian()),
                                                      uvus))
    x = rand(3)
    foo(x)
    testgrad(foo, x)

    testpol(m, uv)
end

@testset "Polarized Semi Analytic" begin
    m = PolarizedModel(ExtendedRing(8.0), 0.1 * Gaussian(), 0.1 * Gaussian(),
                       0.1 * Gaussian())
    g = imagepixels(10.0, 10.0, 512, 512)
    s = map(length, dims(g))

    u = fftshift(fftfreq(length(g.X), 1 / step(g.X)))
    uv = UnstructuredDomain((U=u, V=-u))

    gff = FourierDualDomain(g, uv, FFTAlg())
    gnf = FourierDualDomain(g, uv, NFFTAlg())
    gdf = FourierDualDomain(g, uv, DFTAlg())

    vff = testpol(m, gff)
    vnf = testpol(m, gnf)
    vdf = testpol(m, gdf)

    @test isapprox(vff, vnf, atol=1e-6)
    @test isapprox(vff, vdf, atol=1e-6)
    @test isapprox(vnf, vdf, atol=1e-6)
end

@testset "Polarized Modified" begin
    g = imagepixels(5.0, 5.0, 128, 128)
    u = fftshift(fftfreq(length(g.X), 1 / step(g.X)))
    uv = UnstructuredDomain((U=u, V=-u))

    s = map(length, dims(g))
    m0 = PolarizedModel(ExtendedRing(2.0), 0.1 * Gaussian(), 0.1 * Gaussian(),
                        0.1 * Gaussian())
    m = shifted(m0, 0.1, 0.1)
    gnf = FourierDualDomain(g, uv, NFFTAlg())

    testpol(m, gnf)

    m = rotated(m0, 0.1)
    testpol(m, gnf)

    m = renormed(m0, 0.1)
    testpol(m, gnf)

    m = stretched(m0, 0.1, 0.4)
    testpol(m, gnf)
end

@testset "Polarized Combinators" begin
    m1 = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian())
    m2 = PolarizedModel(Disk(), shifted(Disk(), 0.1, 1.0), ZeroModel(), ZeroModel())
    g = imagepixels(5.0, 5.0, 128, 128)
    u = fftshift(fftfreq(length(g.X), 1 / step(g.X)))
    uv = UnstructuredDomain((U=u, V=-u))

    testpol(convolved(m1, m2), uv)

    testpol(convolved(m1, Gaussian()), uv)
    testpol(convolved(stretched(m1, 0.5, 0.5), Gaussian()), uv)
end

@testset "Polarized NonAnalytic" begin
    m = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian())
    mna = VLBISkyModels.NonAnalyticTest(m)
    g = imagepixels(10.0, 10.0, 128, 128)
    guv = VLBISkyModels.uvgrid(g)
    v = visibilitymap(mna, guv)
    van = visibilitymap(m, guv)
    @test isapprox(maximum(norm, v .- van), 0.0, atol=1e-5)
end

@testset "Polarized All Mod" begin
    m1 = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian())
    m2 = PolarizedModel(ExtendedRing(8.0), shifted(Disk(), 0.1, 1.0), ZeroModel(),
                        ZeroModel())
    m = convolved(convolved(m1, Gaussian()), m2) + convolved(Gaussian(), m1)
    g = imagepixels(5.0, 5.0, 128, 128)
    s = map(length, dims(g))
    u = fftshift(fftfreq(length(g.X), 1 / step(g.X))) ./ 40
    uv = UnstructuredDomain((U=u, V=-u))
    gnf = FourierDualDomain(g, uv, NFFTAlg())

    testpol(m, gnf)
end

@testset "Rotation" begin
    m = PolarizedModel(Gaussian(), Gaussian(), ZeroModel(), 0.1 * Gaussian())
    g = imagepixels(5.0, 5.0, 128, 128)
    img1 = intensitymap(m, g)
    @test size(VLBISkyModels.padimage(img1, FFTAlg(; padfac=2))) == 2 .* size(img1)
    @test all(==(1), stokes(img1, :Q) .≈ stokes(img1, :I))
    @test all(==(1), stokes(img1, :U) .≈ 0.0 * stokes(img1, :I))
    @test all(==(1), stokes(img1, :V) .≈ 0.1 * stokes(img1, :I))

    # Now we rotate into perfect U
    img2 = intensitymap(rotated(m, π / 4), g)
    @test all(==(1), stokes(img2, :U) .≈ stokes(img1, :I))
    @test all(==(1), isapprox.(stokes(img2, :Q), 0.0, atol=1e-16))
    @test all(==(1), stokes(img1, :V) .≈ stokes(img2, :V))

    # Now we rotate into perfectly -Q
    img3 = intensitymap(rotated(m, π / 2), g)
    @test all(==(1), stokes(img3, :Q) .≈ -stokes(img3, :I))
    @test all(==(1), isapprox.(stokes(img3, :U), 0.0, atol=1e-16))
    @test all(==(1), stokes(img1, :V) .≈ stokes(img3, :V))

    # Now we rotate into perfectly -U
    img4 = intensitymap(rotated(m, 3π / 4), g)
    @test all(==(1), stokes(img4, :U) .≈ -stokes(img4, :I))
    @test all(==(1), isapprox.(stokes(img4, :Q), 0.0, atol=1e-16))
    @test all(==(1), stokes(img1, :V) .≈ stokes(img4, :V))

    # Now make sure it is π periodic
    img5 = intensitymap(rotated(m, -π / 4), g)
    @test all(==(1), img4 .≈ img5)
end

@testset "ContinuousImage" begin
    m = PolarizedModel(Gaussian(), Gaussian(), ZeroModel(), 0.1 * Gaussian())
    g = imagepixels(10.0, 10.0, 24, 24)
    img = intensitymap(m, g)
    cimg = ContinuousImage(img, BicubicPulse(0.0))
    @test ComradeBase.ispolarized(typeof(cimg)) === ComradeBase.IsPolarized()
    @test ComradeBase.ispolarized(typeof(stokes(cimg, :I))) === ComradeBase.NotPolarized()

    img0 = intensitymap(cimg, g)

    rimg1 = intensitymap(rotated(cimg, π / 4), g)
    @test all(==(1), stokes(rimg1, :U) .≈ stokes(rimg1, :I))
    @test all(==(1), isapprox.(stokes(rimg1, :Q), 0.0, atol=1e-16))
    @test all(==(1), stokes(rimg1, :V) .≈ 0.1 * stokes(rimg1, :I))

    u, v = randn(32) / 5, randn(32) / 5
    uv = UnstructuredDomain((U=u, V=v))
    gnf = FourierDualDomain(g, uv, NFFTAlg())
    gdf = FourierDualDomain(g, uv, DFTAlg())
    gff = FourierDualDomain(g, uv, FFTAlg())
    mimg = ContinuousImage(img, BSplinePulse{3}())
    vnf = visibilitymap(mimg, gnf)
    vdf = visibilitymap(mimg, gdf)
    vff = visibilitymap(mimg, gff)

    @test isapprox(maximum(norm, vnf .- vdf), 0.0, atol=1e-6)
    @test isapprox(maximum(norm, vff .- vdf), 0.0, atol=1e-3)
end

@testset "PoincareSphere2Map" begin
    m = PolarizedModel(Gaussian(), Gaussian(), ZeroModel(), 0.1 * Gaussian())
    g = imagepixels(5.0, 5, 24, 24)

    img = intensitymap(m, g)

    sI = baseimage(stokes(img, :I))
    sQ = baseimage(stokes(img, :Q))
    sU = baseimage(stokes(img, :U))
    sV = baseimage(stokes(img, :V))

    p = sqrt.(sQ^2 + sU^2 + sV^2)
    Χ = (sQ ./ p, sU ./ p, sV ./ p)

    pimg = PoincareSphere2Map(sI, p ./ sI, Χ, g)
    cimg = ContinuousImage(pimg, BSplinePulse{3}())
    gfour = FourierDualDomain(g, UnstructuredDomain((U=randn(32) / 5, V=randn(32) / 5)),
                              NFFTAlg())
    visibilitymap(cimg, gfour)
    @test (baseimage(stokes(pimg, :I))) ≈ sI
    @test (baseimage(stokes(pimg, :Q))) ≈ sQ
    @test (baseimage(stokes(pimg, :U))) ≈ sU
    @test (baseimage(stokes(pimg, :V))) ≈ sV
end

@testset "PolExp2Map" begin
    m = PolarizedModel(Gaussian(), Gaussian(), ZeroModel(), 0.1 * Gaussian())
    g = imagepixels(5.0, 5, 24, 24)

    img = intensitymap(m, g)
    pimg = PolExp2Map(randn(24, 24), randn(24, 24), randn(24, 24), randn(24, 24), g)
    @info typeof(pimg)
    @test mapreduce(*, pimg) do x
        v = (x.I^2 >= x.Q^2 + x.U^2 + x.V^2)
        return v
    end
end
