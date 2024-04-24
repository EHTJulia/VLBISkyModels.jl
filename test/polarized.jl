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
    m = PolarizedModel(Gaussian(), 0.1*Gaussian(), 0.1*Gaussian(), 0.1*Gaussian())
    g = imagepixels(10.0, 10.0, 512, 512)
    s = map(length, dims(g))

    u = fftshift(fftfreq(length(g.X), 1/step(g.X)))
    uv = RectiGrid((U=u, V=-u))

    testpol(m, uv)
end

@testset "Polarized Semi Analytic" begin
    m = PolarizedModel(ExtendedRing(8.0), 0.1*Gaussian(), 0.1*Gaussian(), 0.1*Gaussian())
    g = imagepixels(10.0, 10.0, 512, 512)
    s = map(length, dims(g))

    u = fftshift(fftfreq(length(g.X), 1/step(g.X)))
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
    u = fftshift(fftfreq(length(g.X), 1/step(g.X)))
    uv = UnstructuredDomain((U=u, V=-u))


    s = map(length, dims(g))
    m0 = PolarizedModel(ExtendedRing(2.0), 0.1*Gaussian(), 0.1*Gaussian(), 0.1*Gaussian())
    m = shifted(m0, 0.1 ,0.1)
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
    m1 = PolarizedModel(Gaussian(), 0.1*Gaussian(), 0.1*Gaussian(), 0.1*Gaussian())
    m2 = PolarizedModel(Disk(), shifted(Disk(), 0.1, 1.0), ZeroModel(), ZeroModel())
    g = imagepixels(5.0, 5.0, 128, 128)
    u = fftshift(fftfreq(length(g.X), 1/step(g.X)))
    uv = UnstructuredDomain((U=u, V=-u))

    testpol(convolved(m1,m2), uv)

    testpol(convolved(m1,Gaussian()), uv)
    testpol(convolved(stretched(m1, 0.5, 0.5),Gaussian()), uv)
end

@testset "Polarized All Mod" begin
    m1 = PolarizedModel(Gaussian(), 0.1*Gaussian(), 0.1*Gaussian(), 0.1*Gaussian())
    m2 = PolarizedModel(ExtendedRing(8.0), shifted(Disk(), 0.1, 1.0), ZeroModel(), ZeroModel())
    m = convolved(m1,m2)+m1
    g = imagepixels(5.0, 5.0, 128, 128)
    s = map(length, dims(g))
    u = fftshift(fftfreq(length(g.X), 1/step(g.X)))
    uv = UnstructuredDomain((U=u, V=-u))
    gnf = FourierDualDomain(g, uv, NFFTAlg())


    testpol(m, gnf)
end

@testset "Rotation" begin
    m = PolarizedModel(Gaussian(), Gaussian(), ZeroModel(), 0.1*Gaussian())
    g = imagepixels(5.0, 5.0, 128, 128)
    img1 = intensitymap(m, g)
    @test all(==(1), stokes(img1, :Q) .≈ stokes(img1, :I))
    @test all(==(1), stokes(img1, :U) .≈ 0.0*stokes(img1, :I))
    @test all(==(1), stokes(img1, :V) .≈ 0.1*stokes(img1, :I))

    # Now we rotate into perfect U
    img2 = intensitymap(rotated(m,  π/4), g)
    @test all(==(1), stokes(img2, :U) .≈ stokes(img1, :I))
    @test all(==(1), isapprox.(stokes(img2, :Q), 0.0, atol=1e-16))
    @test all(==(1), stokes(img1, :V) .≈ stokes(img2, :V))

    # Now we rotate into perfectly -Q
    img3 = intensitymap(rotated(m,  π/2), g)
    @test all(==(1), stokes(img3, :Q) .≈ -stokes(img3, :I))
    @test all(==(1), isapprox.(stokes(img3, :U), 0.0, atol=1e-16))
    @test all(==(1), stokes(img1, :V) .≈ stokes(img3, :V))

    # Now we rotate into perfectly -U
    img4 = intensitymap(rotated(m,  3π/4), g)
    @test all(==(1), stokes(img4, :U) .≈ -stokes(img4, :I))
    @test all(==(1), isapprox.(stokes(img4, :Q), 0.0, atol=1e-16))
    @test all(==(1), stokes(img1, :V) .≈ stokes(img4, :V))

    # Now make sure it is π periodic
    img5 = intensitymap(rotated(m,  -π/4), g)
    @test all(==(1), img4 .≈ img5)
end

@testset "ContinuousImage" begin
    m = PolarizedModel(Gaussian(), Gaussian(), ZeroModel(), 0.1*Gaussian())
    g = imagepixels(10.0, 10.0, 24, 24)
    img = intensitymap(m, g)
    cimg = ContinuousImage(img, BicubicPulse(0.0))
    @test ComradeBase.ispolarized(typeof(cimg)) === ComradeBase.IsPolarized()
    @test ComradeBase.ispolarized(typeof(stokes(cimg, :I))) === ComradeBase.NotPolarized()

    img0 = intensitymap(cimg, g)

    rimg1 = intensitymap(rotated(cimg, π/4), g)
    @test all(==(1), stokes(rimg1, :U) .≈ stokes(rimg1, :I))
    @test all(==(1), isapprox.(stokes(rimg1, :Q), 0.0, atol=1e-16))
    @test all(==(1), stokes(rimg1, :V) .≈ 0.1*stokes(rimg1, :I))

    u, v = rand(32), rand(32)
    uv = UnstructuredDomain((U=u, V=v))
    gnf = FourierDualDomain(g, uv, NFFTAlg())
    mimg = ContinuousImage(StokesIntensityMap(img), BSplinePulse{3}())
    visibilitymap(mimg, gnf)
end
