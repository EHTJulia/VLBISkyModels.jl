
@testset "Moments" begin
    g = imagepixels(30.0, 30.0, 512, 512)
    m1 = Gaussian()
    img = VLBISkyModels.allocate_imgmap(m1, g)
    intensitymap!(img, m1)
    @test isapprox(centroid(img)[1], 0.0, atol=1e-5)
    @test isapprox(centroid(img)[2], 0.0, atol=1e-5)

    I = second_moment(img)
    I2 = second_moment(img; center=false)
    @test isapprox(I, [1.0 0.0; 0.0 1.0], atol=1e-5)
    @test I ≈ I2

    m2 = shifted(m1, 1.0, 1.0)
    intensitymap!(img, m2)
    @test isapprox(centroid(img)[1], 1.0, atol=1e-5)
    @test isapprox(centroid(img)[2], 1.0, atol=1e-5)
    @test isapprox(second_moment(img), I, atol=1e-5)

    m3 = stretched(m1, 2.0, 1.0)
    intensitymap!(img, m3)
    @test isapprox(centroid(img)[1], 0.0, atol=1e-5)
    @test isapprox(centroid(img)[2], 0.0, atol=1e-5)
    I3 = second_moment(img)
    @test isapprox(I3, [4.0 0.0; 0.0 1.0], atol=1e-5)
end

@testset "FFTTest" begin
    @testset "Base" begin
        m = Gaussian()
        testft(m)
    end

    @testset "Mod" begin
        m = rotated(stretched(Gaussian(), 0.5, 1.0), π / 3)
        testft(m)
        ms = shifted(m, 1.0, 1.0)
        testft(ms)
    end

    @testset "Add" begin
        m1 = rotated(stretched(Gaussian(), 0.5, 1.0), π / 3) + shifted(Gaussian(), 1.0, 1.0)
        testft(m1)
    end
end

@testset "Primitive models" begin
    u = randn(100) * 0.5
    v = randn(100) * 0.5
    t = sort(rand(100) * 0.5)
    f = fill(230e9, 100)
    g = UnstructuredDomain((U=u, V=v, Ti=t, Fr=f))

    @testset "Gaussian" begin
        m = Gaussian()
        test_opt(m)
        @test amplitude(m, (U=0.0, V=0.0)) == abs(visibility(m, (U=0.0, V=0.0)))
        @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m, (X=0.0, Y=0.0))

        foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(x[1] * Gaussian(), g))
        x = rand(1)
        foo(x)
        testgrad(foo, x)

        testmodel(m, 128, 1e-4)
    end

    @testset "Disk" begin
        m = smoothed(Disk(), 0.25)
        test_opt(Disk())
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))
        testmodel(m)

        foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(x[1] * Disk(), g))
        x = rand(1)
        foo(x)
        testgrad(foo, x)
    end

    @testset "SlashedDisk" begin
        m = smoothed(SlashedDisk(0.5), 0.25)
        test_opt(SlashedDisk(0.5))
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))
        testmodel(m)

        foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(SlashedDisk(x[1]), g))
        x = rand(1)
        foo(x)
        testgrad(foo, x)
    end

    @testset "Pulses" begin
        m0 = BSplinePulse{0}()
        test_opt(m0)
        @inferred VLBISkyModels.visibility(m0, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m0, (X=0.0, Y=0.0))
        testmodel(m0, 2048, 1e-3; radmul=1.5)
        m1 = BSplinePulse{1}()
        test_opt(m1)
        testmodel(m1, 512, 1e-3)
        @inferred VLBISkyModels.visibility(m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m1, (X=0.0, Y=0.0))
        m3 = BSplinePulse{3}()
        test_opt(m3)
        testmodel(m3, 512, 1e-3; radmul=2.5)
        @inferred VLBISkyModels.visibility(m3, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m3, (X=0.0, Y=0.0))
        m4 = BicubicPulse()
        test_opt(m4)
        testmodel(m4)
        @inferred VLBISkyModels.visibility(m4, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m4, (X=0.0, Y=0.0))
        m5 = RaisedCosinePulse()
        test_opt(m5)
        testmodel(m5)
        @inferred VLBISkyModels.visibility(m5, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m5, (X=0.0, Y=0.0))
    end

    @testset "Butterworth" begin
        m1 = Butterworth{1}()
        test_opt(m1)
        testmodel(m1)
        @inferred VLBISkyModels.visibility(m1, (U=0.0, V=0.0))
        m2 = Butterworth{2}()
        test_opt(m2)
        testmodel(m2)
        @inferred VLBISkyModels.visibility(m2, (U=0.0, V=0.0))
        m3 = Butterworth{3}()
        test_opt(m3)
        testmodel(m3)
        @inferred VLBISkyModels.visibility(m3, (U=0.0, V=0.0))
    end

    @testset "Ring" begin
        m = smoothed(Ring(), 0.25)
        test_opt(Ring())
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))
        testmodel(m, 1024, 1e-3, 0.25)

        foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(x[1] * Ring(), g))
        x = rand(1)
        foo(x)
        testgrad(foo, x)
    end

    @testset "ParabolicSegment" begin
        m = ParabolicSegment()
        test_opt(ParabolicSegment())
        m2 = ParabolicSegment(2.0, 2.0)
        @test stretched(m, 2.0, 2.0) == m2
        @test ComradeBase.intensity_point(m, (X=0.0, Y=1.0)) != 0.0
        @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m, (X=0.0, Y=0.0))
        testmodel(m, 4800, 1e-2, 0.25)

        # TODO why is this broken?
        # foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(ParabolicSegment(x[1], x[2]), g))
        # x = rand(2)
        # foo(x)
        # testgrad(foo, x)
    end

    @testset "MRing1" begin
        α = [0.25]
        β = [0.1]
        #test_rrule(VLBISkyModels.visibility_point, MRing(α, β), 0.5, 0.25)
        # We convolve it to remove some pixel effects
        test_opt(MRing(α, β))
        m = convolved(MRing(α, β), stretched(Gaussian(), 0.1, 0.1))
        m2 = convolved(MRing(α[1], β[1]), stretched(Gaussian(), 0.1, 0.1))
        @test visibility(m, (U=0.1, V=0.1)) == visibility(m2, (U=0.1, V=0.1))
        testmodel(m, 1024, 1e-3)
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))

        testmodel(MRing(0.5, 0.0), 3200, 1e-2, 0.25)

        foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(MRing(x[1], x[2]), g))
        x = rand(2)
        foo(x)
        testgrad(foo, x)
    end

    @testset "MRing2" begin
        α = [0.25, -0.1]
        β = [0.1, 0.2]
        #test_rrule(VLBISkyModels.visibility_point, MRing(α, β), 0.5, 0.25)

        # We convolve it to remove some pixel effects
        m = convolved(MRing(α, β), stretched(Gaussian(), 0.1, 0.1))
        testmodel(m, 2048, 1e-3)
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))

        foo(x) = sum(abs2,
                     VLBISkyModels.visibilitymap_analytic(MRing((x[1], x[2]), (x[3], x[4])),
                                                          g))
        x = rand(4)
        foo(x)
        testgrad(foo, x)
    end

    @testset "ConcordanceCrescent" begin
        m = ConcordanceCrescent(20.0, 10.0, 5.0, 0.5)
        testmodel(m, 2048, 1e-3)
        @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m, (X=0.0, Y=0.0))

        foo(x) = sum(abs2,
                     VLBISkyModels.visibilitymap_analytic(ConcordanceCrescent(x[1], x[2],
                                                                              x[3], x[4]),
                                                          g))
        x = rand(4)
        foo(x)
        testgrad(foo, x; atol=0)
    end

    @testset "Crescent" begin
        m = smoothed(Crescent(5.0, 2.0, 1.0, 0.5), 1.0)
        testmodel(m, 1024, 1e-3)
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))

        u = randn(50)
        v = randn(50)
        t = fill(1.0, 50)
        f = fill(230.0, 50)
        foo(x) = sum(abs2,
                     VLBISkyModels.visibilitymap_analytic(Crescent(x[1], x[2], x[3], x[4]),
                                                          g))
        x = rand(4)
        foo(x)
        testgrad(foo, x; atol=0, rtol=1e-5)
    end

    @testset "ExtendedRing" begin
        mr = ExtendedRing(8.0)
        rad = 2.5 * VLBISkyModels.radialextent(mr)
        testft_nonan(mr)
        @inferred VLBISkyModels.intensity_point(mr, (X=0.0, Y=0.0))
    end

    @testset "M87 model test" begin
        xopt = (rad=21.895093363492155,
                wid=2.1113838380637815,
                a=-0.3276141879612847,
                b=-0.13845264228109883,
                f=0.4584364142294795,
                sig=30.902344705962914,
                asy=0.8036630375887827,
                pa=0.6955748496122764,
                x=-43.84496132303754,
                y=-18.750141889035508)
        function model(θ)
            (; rad, wid, a, b, f, sig, asy, pa, x, y) = θ
            ring = f * smoothed(stretched(MRing((a,), (b,)), μas2rad(rad), μas2rad(rad)),
                                μas2rad(wid))
            g = (1 - f) *
                shifted(rotated(stretched(Gaussian(), μas2rad(sig) * asy, μas2rad(sig)),
                                pa), μas2rad(x), μas2rad(y))
            return ring + g
        end

        m = model(xopt)
        testmodel(m)

        @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
        # k = keys(xopt)
        # foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(model(NamedTuple{k}(Tuple(x))), g))
        # x = collect(values(xopt))
        # foo(x)
        # testgrad(foo, x)
    end
end

@testset "ModelImage" begin
    m1 = Gaussian()
    m2 = ExtendedRing(10.0)
    m = m1 + m2
    g = imagepixels(2 * VLBISkyModels.radialextent(m), 2 * VLBISkyModels.radialextent(m),
                    128, 128)
    img = intensitymap(m2, g)
    img2 = similar(img)
    intensitymap!(img2, m2)
    @test isapprox(maximum(parent(img) - img2), 0.0, atol=1e-8)
    @test isapprox(maximum(parent(img) - img2), 0.0, atol=1e-8)
end

@testset "Modifiers" begin
    u = randn(100) * 0.5
    v = randn(100) * 0.5
    t = sort(rand(100) * 0.5)
    f = fill(230e9, 100)
    g = UnstructuredDomain((U=u, V=v, Ti=t, Fr=f))
    ma = Gaussian()
    mb = ExtendedRing(8.0)
    @testset "Shifted" begin
        mas = shifted(ma, 0.1, 0.1)
        mbs = shifted(mb, 0.1, 0.1)
        testmodel(mas)
        @test ComradeBase.intensity_point(mbs, (X=0.5, Y=0.5)) ≈
              ComradeBase.intensity_point(mb, (X=0.4, Y=0.4))
        testft_nonan(mbs)

        foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(shifted(ma, x[1], x[2]), g))
        x = rand(2)
        foo(x)
        testgrad(foo, x)
    end

    @testset "Renormed" begin
        m1 = 3.0 * ma
        m2 = ma * 3.0
        m2inv = ma / (1 / 3)
        p = (U=4.0, V=0.0)
        @test visibility(m1, p) == visibility(m2, p)
        @test visibility(m2, p) == visibility(m2inv, p)
        mbs = 3.0 * mb
        testmodel(3 * m1)
        testft_nonan(mbs)
        @test ComradeBase.intensity_point(mbs, (X=0.5, Y=0.5)) ≈
              3 * ComradeBase.intensity_point(mb, (X=0.5, Y=0.5))

        foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(x[1] * ma, g))
        x = rand(1)
        foo(x)
        testgrad(foo, x)
    end

    @testset "Stretched" begin
        mas = stretched(ma, 5.0, 4.0)
        mbs = stretched(mb, 5.0, 4.0)
        testmodel(mas)
        @test ComradeBase.intensity_point(mbs, (X=0.5, Y=0.5)) ≈
              ComradeBase.intensity_point(mb, (X=0.5 / 5, Y=0.5 / 4)) / 20
        testft_nonan(mbs)
        foo(x) = sum(abs2,
                     VLBISkyModels.visibilitymap_analytic(stretched(ma, x[1], x[2]), g))
        x = rand(2)
        foo(x)
        testgrad(foo, x)
    end

    @testset "Rotated" begin
        mas = rotated(ma, π / 3)
        mbs = rotated(mb, π / 3)
        testmodel(mas)
        @test ComradeBase.intensity_point(mbs, (X=0.5, Y=0.5)) ≈
              ComradeBase.intensity_point(mb,
                                          (X=0.5 * cos(π / 3) + 0.5 * sin(π / 3),
                                           Y=-0.5 * sin(π / 3) + 0.5 * cos(π / 3)))
        testft_nonan(mbs)
        foo(x) = sum(abs2, VLBISkyModels.visibilitymap_analytic(rotated(ma, x[1]), g))
        x = rand(1)
        foo(x)
        testgrad(foo, x)
    end

    @testset "AllMods" begin
        mas = rotated(stretched(shifted(ma, 0.5, 0.5), 5.0, 4.0), π / 3)
        mas2 = modify(ma, Shift(0.1, 0.1), Stretch(5.0, 4.0), Rotate(π / 4))
        @test typeof(mas2) === typeof(mas)
        mbs = rotated(stretched(shifted(mb, 0.5, 0.5), 5.0, 4.0), π / 3)
        testmodel(mas)
        @test ComradeBase.intensity_point(mbs, (X=0.5, Y=0.5)) ≈
              ComradeBase.intensity_point(mb,
                                          (X=0.5 * cos(π / 3) / 5 - 0.5 * sin(π / 3) / 5 -
                                             0.5,
                                           Y=0.5 * sin(π / 3) / 4 + 0.5 * cos(π / 3) / 4 -
                                             0.5)) / 20
        testft_nonan(mbs)

        foo(x) = sum(abs2,
                     VLBISkyModels.visibilitymap_analytic(modify(ma, Shift(x[1], x[2]),
                                                                 Stretch(x[3], x[4]),
                                                                 Rotate(x[5]),
                                                                 Renormalize(x[6])), g))
        x = rand(6)
        foo(x)
        testgrad(foo, x)
    end
end

@testset "CompositeModels" begin
    m1 = Gaussian()
    m2 = ExtendedRing(8.0)

    u = randn(100) * 0.5
    v = randn(100) * 0.5
    t = sort(rand(100) * 0.5)
    f = fill(230e9, 100)
    guv = UnstructuredDomain((U=u, V=v, Ti=t, Fr=f))

    @testset "Add models" begin
        g = imagepixels(20.0, 20.0, 256, 256)
        mt1 = m1 + m2
        img = allocate_imgmap(mt1, g)
        mt2 = shifted(m1, 1.0, 1.0) + m2
        mt3 = shifted(m1, 1.0, 1.0) + 0.5 * stretched(m2, 0.9, 0.8)
        mc = VLBISkyModels.components(mt1)
        @test mc[1] === m1
        @test mc[2] === m2
        @test flux(mt1) ≈ flux(m1) + flux(m2)
        @test ComradeBase.intensity_point(mt1, (X=0.5, Y=-0.5)) ≈
              ComradeBase.intensity_point(m1, (X=0.5, Y=-0.5)) +
              ComradeBase.intensity_point(m2, (X=0.5, Y=-0.5))

        foo(x) = sum(abs2,
                     VLBISkyModels.visibilitymap_analytic(x[1] *
                                                          stretched(Disk(), x[2], x[3]) +
                                                          stretched(Ring(), x[4], x[4]),
                                                          guv))
        x = rand(4)
        foo(x)
        testgrad(foo, x)

        fooi(x) = sum(abs2,
                      VLBISkyModels.intensitymap_analytic(x[1] *
                                                          stretched(Gaussian(), x[2],
                                                                    x[3]) +
                                                          shifted(GaussianRing(1.0), x[4],
                                                                  x[4]), g))
        x = rand(4)
        fooi(x)
        testgrad(fooi, x)

        gfour = FourierDualDomain(g, guv, NFFTAlg())
        testmodel(m1 + m1)
        testft_nonan(mt1)
        testft_nonan(mt2)
        testft_nonan(mt3)
        @test intensitymap(mt1, g) ≈
              intensitymap(m1, gfour) + intensitymap(m2, gfour)
        @test intensitymap(m1 + m1, g) ≈ 2 * intensitymap(m1, g)
        @test visibilitymap(mt1, gfour) ≈
              visibilitymap(m1, gfour) + visibilitymap(m2, gfour)
        @test visibilitymap(m1 + m1, gfour) ≈ 2 * visibilitymap(m1, gfour)
    end

    @testset "Convolved models" begin
        g = imagepixels(20.0, 20.0, 256, 256)
        guv = UnstructuredDomain((U=randn(32), V=randn(32)))
        gfour = FourierDualDomain(g, guv, NFFTAlg())
        mt1 = convolved(m1, m2)
        img = allocate_imgmap(mt1, g)
        @test intensitymap(mt1, g) ≈ intensitymap(mt1, gfour)

        @test intensitymap(stretched(m1, sqrt(2)), gfour) ≈
              intensitymap(convolved(m1, m1), gfour)

        @test visibilitymap(stretched(m1, sqrt(2)), gfour) ≈
              visibilitymap(convolved(m1, m1), gfour)

        mt2 = convolved(shifted(m1, 1.0, 1.0), m2)
        mt3 = convolved(shifted(m1, 1.0, 1.0), 0.5 * stretched(m2, 0.9, 0.8))
        mc = VLBISkyModels.components(mt1)
        @test mc[1] === m1
        @test mc[2] === m2

        testft_nonan(mt1)
        testft_nonan(mt2)
        testft_nonan(mt3)
        @test visibilitymap(mt1, gfour) ≈
              visibilitymap(m1, gfour) .* visibilitymap(m2, gfour)
        @test visibilitymap(mt2, gfour) ≈
              visibilitymap(mt2.m1, gfour) .* visibilitymap(m2, gfour)
        @test visibilitymap(mt3, gfour) ≈
              visibilitymap(mt3.m1, gfour) .* visibilitymap(mt3.m2, gfour)

        foo(x) = sum(abs2,
                     VLBISkyModels.visibilitymap_analytic(convolved(x[1] *
                                                                    stretched(Disk(), x[2],
                                                                              x[3]),
                                                                    stretched(Ring(), x[4],
                                                                              x[4])), guv))
        x = rand(4)
        foo(x)
        testgrad(foo, x)
    end

    @testset "Convolved Non-analytic" begin
        m1 = Gaussian()
        m2 = VLBISkyModels.NonAnalyticTest(Gaussian())
        g = imagepixels(20.0, 20.0, 256, 256)
        mt = convolved(m1, m2)
        img1 = intensitymap(mt, g)
        img2 = intensitymap(convolved(m1, m2.model), g)

        @test isapprox(parent(img1), parent(img2), rtol=1e-8)
    end

    @testset "All composite" begin
        g = imagepixels(20.0, 20.0, 256, 256)
        guv = UnstructuredDomain((U=randn(32), V=randn(32)))
        gfour = FourierDualDomain(g, guv, NFFTAlg())

        mt = m1 + convolved(m1, m2)
        img = allocate_imgmap(mt, g)
        @test intensitymap(mt, g) ≈ intensitymap(mt, gfour)

        @test VLBISkyModels.visibilitymap_numeric(mt, gfour) ≈ visibilitymap(mt, gfour)

        mc = VLBISkyModels.components(mt)
        @test mc[1] === m1
        @test mc[2] === m1
        @test mc[3] === m2

        testft_nonan(mt)
        foo(x) = sum(abs2,
                     VLBISkyModels.visibilitymap_analytic(smoothed(x[1] *
                                                                   stretched(Disk(), x[2],
                                                                             x[3]), x[4]) +
                                                          stretched(Ring(), x[5], x[4]),
                                                          guv))
        x = rand(5)
        foo(x)
        testgrad(foo, x)
    end
end

@testset "Multicomponent" begin
    fname = Downloads.download("http://www.cv.nrao.edu/2cmVLBA/data/0003-066/2012_03_04/0003-066.u.2012_03_04.icn.fits.gz",
                               joinpath(@__DIR__, "test.fits"))
    mod = VLBISkyModels.load_clean_components(fname)
    mod2 = VLBISkyModels.load_clean_components(fname, modify(Gaussian(), μas2rad(100.0)))
    rm(fname)
    u = randn(100) * 0.5
    v = randn(100) * 0.5
    t = sort(rand(100) * 0.5)
    f = fill(230e9, 100)
    guv = UnstructuredDomain((U=u, V=v, Ti=t, Fr=f))

    m = MultiComponentModel(Gaussian(), rand(10), randn(10), randn(10))
    @test amplitude(m, (U=0.0, V=0.0)) == abs(visibility(m, (U=0.0, V=0.0)))
    @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
    @inferred VLBISkyModels.intensity_point(m, (X=0.0, Y=0.0))

    @test ComradeBase.visanalytic(typeof(m)) == ComradeBase.visanalytic(typeof(m.base))

    @test convolved(m, Gaussian()) == convolved(Gaussian(), m)

    foo(x) = sum(abs2,
                 VLBISkyModels.visibilitymap_analytic(MultiComponentModel(Gaussian(),
                                                                          @view(x[:, 1]),
                                                                          @view(x[:, 2]),
                                                                          @view(x[:, 3])),
                                                      guv))
    x = randn(10, 4)
    foo(x)
    testgrad(foo, x)

    testmodel(m, 512, 1e-4)
end

@testset "PolarizedModel" begin
    mI = stretched(MRing((0.2,), (0.1,)), 20.0, 20.0)
    mQ = 0.2 * stretched(MRing((0.0,), (0.6,)), 20.0, 20.0)
    mU = 0.2 * stretched(MRing((0.1,), (-0.6,)), 20.0, 20.0)
    mV = 0.0 * stretched(MRing((0.0,), (-0.6,)), 20.0, 20.0)
    m = PolarizedModel(mI, mQ, mU, mV)
    @inferred visibility(m, (U=0.0, V=0.0))
    @inferred ComradeBase.intensity_point(m, (X=0.0, Y=0.0))
    guv = UnstructuredDomain((U=randn(64), V=randn(64)))
    g = imagepixels(10.0, 10.0, 64, 64)

    foo(x) = sum(norm,
                 VLBISkyModels.visibilitymap_analytic(rotated(PolarizedModel(Gaussian(),
                                                                             x[1] *
                                                                             Gaussian(),
                                                                             x[2] *
                                                                             Gaussian(),
                                                                             x[3] *
                                                                             Gaussian()),
                                                              x[4]), guv))
    testgrad(foo, rand(4))
    fooi(x) = sum(norm,
                  VLBISkyModels.intensitymap_analytic(rotated(PolarizedModel(Gaussian(),
                                                                             x[1] *
                                                                             Gaussian(),
                                                                             x[2] *
                                                                             Gaussian(),
                                                                             x[3] *
                                                                             Gaussian()),
                                                              x[4]), g))
    testgrad(fooi, rand(4))

    mG = PolarizedModel(Gaussian(), Gaussian(), Gaussian(), Gaussian())
    cm = convolved(m, Gaussian())
    @test cm == convolved(m, mG)
    @inferred cm + mG
    show(m)

    p = (U=0.005, V=0.01)
    v = visibility(m, p)
    @test m̆(v) ≈ m̆(m, p)
    @test mbreve(v) ≈ mbreve(m, p)

    g = imagepixels(60.0, 60.0, 128, 128)
    img = intensitymap(m, g)
    p0 = (X=g.X[64], Y=g.Y[64])
    @test linearpol(m, p0) ≈ linearpol(img[64, 64])
    @test mpol(m, p0) ≈ mpol(img[64, 64])
    @test polarization(m, p0) ≈ polarization(img[64, 64])
    @test fracpolarization(m, p0) ≈ fracpolarization(img[64, 64])
    @test evpa(m, p0) ≈ evpa(img[64, 64])
    map((x, y) -> (@test x ≈ y), polellipse(m, p0), polellipse(img[64, 64]))

    g = imagepixels(100.0, 100.0, 1024, 1024)
    pI = IntensityMap(zeros(1024, 1024), g)
    pQ = similar(pI)
    pU = similar(pI)
    pV = similar(pI)
    pimg1 = stokes_intensitymap(pI, pQ, pU, pV)
    intensitymap!(pimg1, m)
    pimg2 = intensitymap(m, axisdims(pI))
    @test isapprox(sum(abs, (stokes(pimg1, :I) .- stokes(pimg2, :I))), 0.0, atol=1e-12)
    @test isapprox(sum(abs, (stokes(pimg1, :Q) .- stokes(pimg2, :Q))), 0.0, atol=1e-12)
    @test isapprox(sum(abs, (stokes(pimg1, :U) .- stokes(pimg2, :U))), 0.0, atol=1e-12)
    @test isapprox(sum(abs, (stokes(pimg1, :V) .- stokes(pimg2, :V))), 0.0, atol=1e-12)
end

@testset "SingleStokes" begin
    mI = stretched(MRing((0.2,), (0.1,)), 20.0, 20.0)
    mQ = 0.2 * stretched(MRing((0.0,), (0.6,)), 20.0, 20.0)
    mU = 0.2 * stretched(MRing((0.1,), (-0.6,)), 20.0, 20.0)
    mV = 0.0 * stretched(MRing((0.0,), (-0.6,)), 20.0, 20.0)
    m = PolarizedModel(mI, mQ, mU, mV)

    msI = SingleStokes(m, :I)
    msQ = SingleStokes(m, :Q)
    msU = SingleStokes(m, :U)
    msV = SingleStokes(m, :V)

    @test_throws ArgumentError SingleStokes(m, :X)

    @test flux(msI) ≈ flux(mI)
    @test flux(msQ) ≈ flux(mQ)
    @test flux(msU) ≈ flux(mU)
    @test flux(msV) ≈ flux(mV)

    p = (U=0.5, V=0.1)
    @test ComradeBase.visibility_point(mI, p) ≈ ComradeBase.visibility_point(msI, p)
    @test ComradeBase.visibility_point(mQ, p) ≈ ComradeBase.visibility_point(msQ, p)
    @test ComradeBase.visibility_point(mU, p) ≈ ComradeBase.visibility_point(msU, p)
    @test ComradeBase.visibility_point(mV, p) ≈ ComradeBase.visibility_point(msV, p)

    pI = (X=0.5, Y=0.6)
    @test ComradeBase.intensity_point(mI, pI) ≈ ComradeBase.intensity_point(msI, pI)
    @test ComradeBase.intensity_point(mQ, pI) ≈ ComradeBase.intensity_point(msQ, pI)
    @test ComradeBase.intensity_point(mU, pI) ≈ ComradeBase.intensity_point(msU, pI)
    @test ComradeBase.intensity_point(mV, pI) ≈ ComradeBase.intensity_point(msV, pI)

    mmod = modify(m, Rotate(π / 3))
    mmodI = SingleStokes(mmod, :I)
    mmodQ = SingleStokes(mmod, :Q)
    mmodU = SingleStokes(mmod, :U)
    mmodV = SingleStokes(mmod, :V)

    @test ComradeBase.visibility_point(mmodI, p) ≈ ComradeBase.visibility_point(mmod, p).I
    @test ComradeBase.visibility_point(mmodQ, p) ≈ ComradeBase.visibility_point(mmod, p).Q
    @test ComradeBase.visibility_point(mmodU, p) ≈ ComradeBase.visibility_point(mmod, p).U
    @test ComradeBase.visibility_point(mmodV, p) ≈ ComradeBase.visibility_point(mmod, p).V

    @test ComradeBase.intensity_point(mmodI, pI) ≈ ComradeBase.intensity_point(mmod, pI).I
    @test ComradeBase.intensity_point(mmodQ, pI) ≈ ComradeBase.intensity_point(mmod, pI).Q
    @test ComradeBase.intensity_point(mmodU, pI) ≈ ComradeBase.intensity_point(mmod, pI).U
    @test ComradeBase.intensity_point(mmodV, pI) ≈ ComradeBase.intensity_point(mmod, pI).V
end

@testset "Serialization" begin
    gim = imagepixels(10.0, 10.0, 128, 128)
    u = randn(100) * 0.5
    v = randn(100) * 0.5
    t = sort(rand(100) * 0.5)
    f = fill(230e9, 100)
    guv = UnstructuredDomain((U=u, V=v, Ti=t, Fr=f))
    m = ExtendedRing(10.0)
    @testset "NFFT" begin
        gnf = FourierDualDomain(gim, guv, NFFTAlg())

        serialize("foo.jls", gnf)
        gnfs = deserialize("foo.jls")
        @test visibilitymap(m, gnfs) ≈ visibilitymap(m, gnf)
        rm("foo.jls")
    end

    @testset "FFT" begin
        gff = FourierDualDomain(gim, guv, FFTAlg())
        serialize("foo.jls", gff)
        gffs = deserialize("foo.jls")
        @test visibilitymap(m, gffs) ≈ visibilitymap(m, gff)
        rm("foo.jls")
    end

    @testset "DFT" begin
        gff = FourierDualDomain(gim, guv, DFTAlg())
        serialize("foo.jls", gff)
        gffs = deserialize("foo.jls")
        @test visibilitymap(m, gffs) ≈ visibilitymap(m, gff)
        rm("foo.jls")
    end
end

@testset "methods " begin
    u1cp = 10e9 * rand(100) .- 5e9
    v1cp = 10e9 * rand(100) .- 5e9
    u2cp = 10e9 * rand(100) .- 5e9
    v2cp = 10e9 * rand(100) .- 5e9
    u3cp = -u1cp - u2cp
    v3cp = -v1cp - v2cp

    u1lc = 10e9 * rand(100) .- 5e9
    v1lc = 10e9 * rand(100) .- 5e9
    u2lc = 10e9 * rand(100) .- 5e9
    v2lc = 10e9 * rand(100) .- 5e9
    u3lc = 10e9 * rand(100) .- 5e9
    v3lc = 10e9 * rand(100) .- 5e9
    u4lc = -u1lc - u2lc - u3lc
    v4lc = -v1lc - v2lc - v3lc

    # m = rotated(stretched(Gaussian(), μas2rad(2.0), μas2rad(1.0)), π/8)
    # @test closure_phasemap(m, (U=u1cp, V=v1cp), (U=u2cp, V=v2cp), (U=u3cp, V=v3cp)) ≈ zero(u1cp)
    # logclosure_amplitudemap(m, (U=u1lc, V=v1lc), (U=u2lc, V=v2lc), (U=u3lc, V=v3lc), (U=u4lc, V=v4lc))

end

@testset "FourierDualDomain" begin
    g = imagepixels(μas2rad(12.0), μas2rad(12.0), 24, 12)
    m = rotated(stretched(Gaussian(), μas2rad(2.0), μas2rad(1.0)), π / 8)
    img = intensitymap(m, g)
    u1 = 10e9 * rand(100) .- 5e9
    v1 = 10e9 * rand(100) .- 5e9

    @testset "Dual domain test" begin
        grid = imagepixels(10.0, 10.0, 12, 12)
        g2 = VLBISkyModels.xygrid(VLBISkyModels.uvgrid(grid))
        @test g2.X ≈ grid.X
        @test g2.Y ≈ grid.Y

        grid = imagepixels(10.0, 10.0, 13, 13)
        g2 = VLBISkyModels.xygrid(VLBISkyModels.uvgrid(grid))
        @test g2.X ≈ grid.X
        @test g2.Y ≈ grid.Y
    end

    ComradeBase.visibilitymap_numeric(m, VLBISkyModels.uvgrid(g))

    guv = UnstructuredDomain((U=u1, V=v1))
    @test_throws "UnstructuredDomain not supported" ComradeBase.visibilitymap_numeric(m,
                                                                                      guv)
    @test_throws "UnstructuredDomain not supported" ComradeBase.intensitymap_numeric(m, guv)

    gnf = FourierDualDomain(g, guv, NFFTAlg())
    @test intensitymap(m, gnf) ≈ img

    show(IOBuffer(), MIME"text/plain"(), gnf)
    cimg = ContinuousImage(img, DeltaPulse())
    # test_rrule(ContinuousImage, img, DeltaPulse() ⊢ NoTangent())

    plan = VLBISkyModels.forward_plan(gnf).plan

    # @testset "nuft pullback" begin
    #     x = rand(24, 12)
    #     temp(x) = sum(abs2, VLBISkyModels._nuft(plan, x))
    #     gfo = ForwardDiff.gradient(temp, x)
    #     gz, = Zygote.gradient(temp, x)
    #     dx = zero(x)
    #     autodiff(Enzyme.Reverse, temp, Active, Duplicated(x, dx))
    #     @test gfo ≈ gz
    #     @test dx ≈ gz
    #     # test_rrule(VLBISkyModels._nuft, plan ⊢ NoTangent(), baseimage(img))
    # end
end

using ForwardDiff
@testset "Rules" begin
    data = rand(32, 32)
    g = imagepixels(10.0, 10.0, 32, 32)
    # test_rrule(IntensityMap, data, g ⊢ NoTangent())

    gfour = FourierDualDomain(g, FFTAlg())
    plan = VLBISkyModels.forward_plan(gfour)

    data = rand(8, 8)
    g = imagepixels(10.0, 10.0, 8, 8)
    gfour = FourierDualDomain(g, FFTAlg())
    plan = VLBISkyModels.forward_plan(gfour)

    x = rand(8, 8)
    temp(x) = sum(abs2, VLBISkyModels.applyft(plan, IntensityMap(x, g)))
    gfo = ForwardDiff.gradient(temp, x)
    fdm = central_fdm(5, 1)
    gfd, = grad(fdm, temp, x)
    @test gfo ≈ gfd
end
