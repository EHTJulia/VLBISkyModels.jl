@testset "StokesIntensityMap" begin
    g = imagepixels(10.0, 10.0, 2, 2)
    m = PolarizedModel(Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian(), 0.1 * Gaussian())
    img = intensitymap(m, g)

    simg = StokesIntensityMap(img)
    simg2 = StokesIntensityMap(stokes(img, :I), stokes(img, :Q),
                               stokes(img, :U), stokes(img, :V))
    simg3 = StokesIntensityMap(baseimage(stokes(img, :I)), baseimage(stokes(img, :Q)),
                               baseimage(stokes(img, :U)), baseimage(stokes(img, :V)),
                               axisdims(img))

    @test simg == simg2
    @test simg == simg3

    summary(simg)
    show(img)

    imageviz(simg)
    CM.image(simg)
    polimage(simg)
    @test centroid(simg) == centroid(img)
    @test size(simg) == size(img)
    @test eltype(simg) == eltype(img)
    @test ndims(simg) == ndims(img)
    @test ndims(typeof(simg)) == ndims(img)
    @test simg[1] == img[1]
    @test simg[2, 2] == img[2, 2]
    @test pixelsizes(img) == pixelsizes(simg)
    @test fieldofview(img) == fieldofview(simg)
    @test domainpoints(simg) == domainpoints(g)
    @test flux(simg) ≈ flux(img)

    simg[1] = StokesParams(0.0, 0.0, 0.0, 0.0)
    @test simg[1] == StokesParams(0.0, 0.0, 0.0, 0.0)

    @test IntensityMap(simg) == img

    pimg = VLBISkyModels.padimage(simg, FFTAlg())
    pimgI = VLBISkyModels.padimage(stokes(simg, :I), FFTAlg())
    @test size(pimg) == size(pimgI)
end
