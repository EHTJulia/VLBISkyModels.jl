@testset "image modifiers" begin
    m = Gaussian()
    g = imagepixels(20.0, 20.0, 128, 128)
    img = intensitymap(m, g)

    mp = PolarizedModel(Gaussian(), Gaussian(), ZeroModel(), Gaussian())

    pimg = intensitymap(mp, g)

    @testset "Rotate invariant" begin
        img2 = rotated(img, pi / 4)
        @test isapprox(img2, img, rtol = 1.0e-2)

        pimg2 = rotated(pimg, π / 4)
        @test isapprox(stokes(pimg2, :U), stokes(pimg, :Q), rtol = 1.0e-2)
        @test isapprox(stokes(pimg2, :Q), stokes(pimg, :U), atol = 1.0e-6)
        @test isapprox(stokes(pimg2, :V), stokes(pimg, :V), rtol = 1.0e-2)
        @test isapprox(stokes(pimg2, :I), img2, rtol = 1.0e-2)
    end

    @testset "Stretched" begin
        g = imagepixels(20.0, 20.0, 128, 128)
        m2 = stretched(m, 2.0, 1.0)
        imgs = intensitymap(m2, g)
        imgs2 = stretched(img, 2.0, 1.0)
        @test isapprox(imgs2, imgs, rtol = 1.0e-2)

        pm2 = stretched(mp, 2.0, 1.0)
        pimgs = intensitymap(pm2, g)
        pimg2 = stretched(pimg, 2.0, 1.0)
        @test isapprox(pimg2, pimgs, rtol = 1.0e-2)
    end

    @testset "Stretch and rotate" begin
        g = imagepixels(20.0, 20.0, 128, 128)

        m2 = modify(m, Stretch(2.0, 1.0), Rotate(π / 4))
        imgs = intensitymap(m2, axisdims(img))
        imgs2 = modify(img, Stretch(2.0, 1.0), Rotate(π / 4))
        @test isapprox(imgs2, imgs, rtol = 1.0e-2)

        pm2 = modify(mp, Stretch(2.0, 1.0), Rotate(π / 4))
        pimgs = intensitymap(pm2, g)
        pimg2 = modify(pimg, Stretch(2.0, 1.0), Rotate(π / 4))
        @test isapprox(pimg2, pimgs, rtol = 1.0e-2)
    end

    @testset "convolve" begin
        cimg = VLBISkyModels.convolve(img, Gaussian())
        img2 = modify(img, Stretch(√(2.0)))
        @test isapprox(cimg, img2, rtol = 1.0e-2)

        cpimg = VLBISkyModels.convolve(pimg, Gaussian())
        pimg2 = modify(pimg, Stretch(sqrt(2)))
        @test isapprox(cpimg, pimg2, rtol = 1.0e-2)

        simg = VLBISkyModels.smooth(img, 1.0)
        @test isapprox(simg, cimg, rtol = 1.0e-2)
    end

    @testset "regrid" begin
        g = imagepixels(10.0, 10.0, 64, 64)
        rimg = regrid(img, g)
        @test size(rimg) == (64, 64)

        rpimg = regrid(pimg, g)
        @test size(rpimg) == (64, 64)
    end

    @testset "center image" begin
        img2 = shifted(img, 1.0, 1.0)
        pimg2 = shifted(pimg, 1.0, 1.0)

        img2_c = center_image(img2)
        @test isapprox(img, img2_c, rtol = 1.0e-2)

        pimg2_c = center_image(pimg2)
        @test isapprox(pimg, pimg2_c, rtol = 1.0e-2)

        imgs = [copy(img2) for _ in 1:10]
        cmimg = centroid_mean(imgs)
        @test isapprox(img, cmimg, rtol = 1.0e-2)
    end
end
