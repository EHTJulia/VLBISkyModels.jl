@testset "ContinuousImage Bspline0" begin
    g = imagepixels(12.0, 12.0, 12, 12)
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π / 8), g)
    cimg = ContinuousImage(img, BSplinePulse{0}())
    testmodel(InterpolatedModel(cimg, g; algorithm=FFTAlg()), 1024, 1e-2)
    testft_cimg(cimg)
end

@testset "ContinuousImage BSpline1" begin
    g = imagepixels(12.0, 12.0, 12, 12)
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π / 8), g)
    cimg = ContinuousImage(img, BSplinePulse{1}())
    # testmodel(InterpolatedModel(cimg, g; algorithm=FFTAlg()), 1024, 1e-3)
    testft_cimg(cimg)
end

@testset "ContinuousImage BSpline3" begin
    g = imagepixels(24.0, 24.0, 12, 12)
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π / 8), g)
    cimg = ContinuousImage(img, BSplinePulse{3}())
    # testmodel(InterpolatedModel(cimg, g; algorithm=FFTAlg()), 1024, 1e-3)
    testft_cimg(cimg)
    guv = UnstructuredDomain((U=randn(32) / 40, V=randn(32) / 40))
    gfour = FourierDualDomain(g, guv, NFFTAlg())
    foo(x) = sum(abs2,
                 VLBISkyModels.visibilitymap(ContinuousImage(IntensityMap(x, g),
                                                             BSplinePulse{3}()), gfour))
    testgrad(foo, rand(12, 12))

    foos(x) = sum(abs2,
                  VLBISkyModels.visibilitymap(modify(ContinuousImage(IntensityMap(reshape(@view(x[1:(end - 1)]),
                                                                                          size(g)),
                                                                                  g),
                                                                     BSplinePulse{3}()),
                                                     Shift(x[end], -x[end])), gfour))
    foos(rand(12 * 12 + 1))
    testgrad(foos, rand(12 * 12 + 1))
end

@testset "ContinuousImage Bicubic" begin
    g = imagepixels(24.0, 24.0, 12, 12)
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π / 8), g)
    cimg = ContinuousImage(img, BicubicPulse())
    # testmodel(InterpolatedModel(cimg, g), 1024, 1e-3)
    testft_cimg(cimg)
end

@testset "ContinuousImage" begin
    g = imagepixels(10.0, 10.0, 128, 128)
    data = rand(128, 128)
    img = ContinuousImage(IntensityMap(data, g), BSplinePulse{3}())
    @test img == ContinuousImage(data, g, BSplinePulse{3}())

    @test length(img) == length(data)
    @test size(img) == size(data)
    @test firstindex(img) == firstindex(data)
    @test lastindex(img) == lastindex(img)
    @test eltype(img) == eltype(data)
    @test img[1, 1] == data[1, 1]
    @test img[1:5, 1] == data[1:5, 1]

    centroid(img)
    @test size(img, 1) == 128
    @test axes(img) == axes(parent(img))
    @test domainpoints(img) == domainpoints(parent(img))

    # @test all(==(1), domainpoints(img) .== ComradeBase.grid(named_dims(axisdims(img))))
    @test VLBISkyModels.axisdims(img) == axisdims(img)

    @test g == axisdims(img)
    @test VLBISkyModels.radialextent(img) ≈ 10.0 / 2

    @test convolved(img, Gaussian()) isa ContinuousImage
    @test convolved(Gaussian(), img) isa ContinuousImage

    # test_rrule(ContinuousImage, IntensityMap(data, g), BSplinePulse{3}() ⊢ NoTangent())
end

