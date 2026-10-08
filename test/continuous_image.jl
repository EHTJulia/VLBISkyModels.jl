@testset "ContinuousImage Bspline0" begin
    g = spatialgrid(12.0, 12.0, 12, 12)
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π / 8), g)
    cimg = ContinuousImage(img, BSplinePulse{0}())
    # testmodel(InterpolatedModel(cimg, g; algorithm=FFTAlg()), 1024, 1e-2)
    testft_cimg(cimg)
end

@testset "ContinuousImage BSpline1" begin
    g = spatialgrid(12.0, 12.0, 12, 12)
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π / 8), g)
    cimg = ContinuousImage(img, BSplinePulse{1}())
    # testmodel(InterpolatedModel(cimg, g; algorithm=FFTAlg()), 1024, 1e-3)
    testft_cimg(cimg)
end

@testset "ContinuousImage BSpline3" begin
    g = spatialgrid(24.0, 24.0, 12, 12)
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π / 8), g)
    cimg = ContinuousImage(img, BSplinePulse{3}())
    # testmodel(InterpolatedModel(cimg, g; algorithm=FFTAlg()), 1024, 1e-3)
    testft_cimg(cimg)
    guv = UnstructuredDomain((U = randn(32) / 40, V = randn(32) / 40))
    gfour = FourierDualDomain(g, guv, NFFTAlg())
    foo(x) = sum(
        abs2,
        VLBISkyModels.visibilitymap(
            ContinuousImage(
                IntensityMap(x, g),
                BSplinePulse{3}()
            ), gfour
        )
    )
    testgrad(foo, rand(12, 12))

    foos(x) = sum(
        abs2,
        VLBISkyModels.visibilitymap(
            modify(
                ContinuousImage(
                    IntensityMap(
                        reshape(
                            @view(x[1:(end - 1)]),
                            size(g)
                        ),
                        g
                    ),
                    BSplinePulse{3}()
                ),
                Shift(x[end], -x[end])
            ), gfour
        )
    )
    foos(rand(12 * 12 + 1))
    testgrad(foos, rand(12 * 12 + 1))
end

@testset "ContinuousImage Bicubic" begin
    g = spatialgrid(24.0, 24.0, 12, 12)
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π / 8), g)
    cimg = ContinuousImage(img, BicubicPulse())
    # testmodel(InterpolatedModel(cimg, g), 1024, 1e-3)
    testft_cimg(cimg)
end

@testset "ContinuousImage" begin
    g = spatialgrid(10.0, 10.0, 16, 16)
    data = rand(16, 16)
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
    @test size(img, 1) == 16
    @test axes(img) == axes(parent(img))
    @test domainpoints(img) == domainpoints(parent(img))

    # @test all(==(1), domainpoints(img) .== ComradeBase.grid(named_dims(axisdims(img))))
    @test VLBISkyModels.axisdims(img) == axisdims(img)

    @test g == axisdims(img)
    @test VLBISkyModels.radialextent(img) ≈ 10.0 / 2

    @test convolved(img, Gaussian()) isa ContinuousImage
    @test convolved(Gaussian(), img) isa ContinuousImage

    guv = UnstructuredDomain((U = randn(32) / 40, V = randn(32) / 40))
    gfour = FourierDualDomain(g, guv, NFFTAlg())

    dm = dualmap(img, gfour)
    @test ComradeBase.vismap(dm) ≈ visibilitymap(img, gfour)

    # This is separate because only the DeltaPulse can use `data` directly
    # The others are convolutions and do not preserve the image data.
    img0 = ContinuousImage(data, g, DeltaPulse())
    dm0 = dualmap(img0, gfour)
    @test parent(ComradeBase.imgmap(dm0)) ≈ data

    imgg1 = img + Gaussian()
    imgg2 = Gaussian() + img
    dm1 = dualmap(imgg1, gfour)
    dm2 = dualmap(imgg2, gfour)
    @test ComradeBase.imgmap(dm1) ≈ ComradeBase.imgmap(dm2)
    @test ComradeBase.vismap(dm1) ≈ ComradeBase.vismap(dm2)

    imgg = intensitymap(Gaussian(), gfour)
    @test ComradeBase.imgmap(dm1) ≈ intensitymap(img, g) .+ imgg

    @test intensitymap(img, g) ≈ intensitymap(img, gfour)
    @test intensitymap(img, g) ≈ ComradeBase.imgmap(dualmap(img, gfour))

    gbg = spatialgrid(12.1, 12.1, 96, 96)
    @test collect(centroid(img)) ≈ collect(centroid(img, gbg)) rtol = 1.0e-3
    @test flux(img) ≈ flux(img, gbg) rtol = 1.0e-4
end

@testset "separable resampling agrees with the support window" begin
    g = spatialgrid(10.0, 8.0, 16, 12, 0.3, -0.2)
    for kernel in (BSplinePulse{0}(), BSplinePulse{1}(), BSplinePulse{3}(), BicubicPulse(), RaisedCosinePulse())
        for gout in (spatialgrid(10.0, 8.0, 32, 24, 0.3, -0.2), spatialgrid(14.0, 11.0, 21, 17), spatialgrid(6.0, 5.0, 9, 7, 1.0, 0.5))
            src = IntensityMap(rand(16, 12), g)
            a = allocate_imgmap(ContinuousImage(src, kernel), gout)
            b = similar(a)
            VLBISkyModels._resample!(a, src, kernel)
            VLBISkyModels._resample_window!(b, src, kernel)
            @test a ≈ b
        end
    end

    psrc = IntensityMap(FieldDimArray{StokesParams}(rand(16, 12, 4)), g)
    gfr = RectiGrid((; X = range(-5.0, 5.0; length = 20), Y = range(-4.0, 4.0; length = 18), Fr = [230.0e9, 345.0e9]))
    c = ContinuousImage(psrc, BSplinePulse{3}())
    img = @inferred intensitymap(c, gfr)
    for s in (:I, :Q, :U, :V), f in 1:2
        ref = intensitymap(ContinuousImage(stokes(psrc, s), BSplinePulse{3}()), VLBISkyModels.spatialdims(gfr))
        @test baseimage(stokes(img, s))[:, :, f] ≈ baseimage(ref)
    end

    grot = RectiGrid((; X = range(-5.0, 5.0; length = 20), Y = range(-4.0, 4.0; length = 18)); posang = 0.3)
    @test intensitymap(ContinuousImage(IntensityMap(rand(16, 12), g), BSplinePulse{3}()), grot) isa IntensityMap

    gout = spatialgrid(10.0, 8.0, 24, 20)
    loss(x) = sum(abs2, baseimage(intensitymap(ContinuousImage(IntensityMap(x, g), BSplinePulse{3}()), gout)))
    testgrad(loss, rand(16, 12))
end
