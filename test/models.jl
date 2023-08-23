function testmodel(m::VLBISkyModels.AbstractModel, npix=1024, atol=1e-4)
    plot(m)
    g = imagepixels(4*VLBISkyModels.radialextent(m), 4*VLBISkyModels.radialextent(m), npix, npix)
    CM.image(g.X, g.Y, m)
    img = intensitymap(m, g)
    imgt = intensitymap(m, g, true)
    imgt2 = intensitymap(m, g, false)
    @test isapprox(maximum(img .- imgt), 0.0, atol=1e-8)
    @test isapprox(maximum(img .- imgt2), 0.0, atol=1e-8)
    plot(img)
    CM.image(img)
    img2 = similar(img)
    intensitymap!(img2, m)
    @test eltype(img) === Float64
    @test isapprox(flux(m), flux(img), atol=atol)
    @test isapprox(maximum(parent(img) .- parent(img2)), 0, atol=1e-8)
    cache = VLBISkyModels.create_cache(VLBISkyModels.FFTAlg(padfac=3), img/flux(img)*flux(m))
    dx, dy = pixelsizes(img)
    u = fftshift(fftfreq(size(img,1), 1/dx))./30
    Plots.closeall()
    @test isapprox(maximum(abs, (visibility.(Ref(m), NamedTuple{(:U, :V)}.(u', u)) .- cache.sitp.(u', u))), 0.0, atol=atol*10)
    img = nothing
    img2 =nothing
    cache = nothing
    u = nothing
    GC.gc()
end


function testft(m, npix=256, atol=1e-4)
    mn = VLBISkyModels.NonAnalyticTest(m)
    uu = 0.25*randn(1000)
    vv = 0.25*randn(1000)
    img = intensitymap(m, 2*VLBISkyModels.radialextent(m), 2*VLBISkyModels.radialextent(m), npix, npix)
    mimg_ff = modelimage(mn, zero(img), FFTAlg(padfac=4))
    mimg_nf = modelimage(mn, zero(img), NFFTAlg())
    mimg_df = modelimage(mn, zero(img), DFTAlg())
    cache = create_cache(FFTAlg(padfac=4), zero(img))
    cache_nf = create_cache(NFFTAlg(), zero(img))
    mimg_ff2 = modelimage(mn, cache)

    p = (U=uu, V=vv)
    va = visibilities(m, p)
    vff = visibilities(mimg_ff, p)
    vff2 = visibilities(mimg_ff2, p)
    vnf = visibilities(mimg_nf, p)
    vdf = visibilities(mimg_df, p)
    visibilities(modelimage(mn, cache_nf), p)

    @test isapprox(maximum(abs, vff2-vff), 0, atol=atol)
    @test isapprox(maximum(abs, va-vff), 0, atol=atol*5)
    @test isapprox(maximum(abs, va-vnf), 0, atol=atol)
    @test isapprox(maximum(abs, va-vdf), 0, atol=atol)
    img = nothing
    mimg_ff = nothing
    mimg_nf = nothing
    mimg_df = nothing
    GC.gc()
end


function testft_cimg(m, atol=1e-4)
    dx, dy = pixelsizes(m.img)
    u = fftshift(fftfreq(500, 1/dx))
    v = fftshift(fftfreq(500, 1/dy))
    mimg_ff = modelimage(m, FFTAlg(padfac=8))
    mimg_nf = modelimage(m, NFFTAlg(u, v))
    mimg_df = modelimage(m, DFTAlg(u, v))

    p = (U=u, V=v)
    vff = visibilities(mimg_ff, p)
    vnf = visibilities(mimg_nf, p)
    vdf = visibilities(mimg_df, p)

    @test isapprox(maximum(abs, vdf .- vnf), 0, atol=atol)
    @test isapprox(maximum(abs, vff .- vdf), 0, atol=atol)
    img = nothing
    mimg_ff = nothing
    mimg_nf = nothing
    mimg_df = nothing
    GC.gc()
end

@testset "Moments" begin
    img = IntensityMap(zeros(512, 512), 30.0, 30.0)
    m1 = Gaussian()
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
        m = rotated(stretched(Gaussian(), 0.5, 1.0), π/3)
        testft(m)
        ms = shifted(m, 1.0,1.0)
        testft(ms)
    end

    @testset "Add" begin
        m1 = rotated(stretched(Gaussian(), 0.5, 1.0), π/3) + shifted(Gaussian(), 1.0, 1.0)
        testft(m1)
    end
end

# 1.7x Enzyme fails (GC?) so we skip this.
if VERSION >= v"1.8"
    function testgrad(f, args...)
        gz = Zygote.gradient(f, args...)
        fdm = central_fdm(5, 1)
        gf = grad(fdm, f, args...)
        map(gz, gf) do dgz, dgf
            @test isapprox(dgz, dgf, atol=1e-5)
        end
    end
else
    function testgrad(f, x)
        return nothing
    end
end

@testset "Primitive models" begin

    u = randn(100)*0.5
    v = randn(100)*0.5
    t = sort(rand(100)*0.5)
    f = fill(230e9, 100)

    @testset "Gaussian" begin
        m = Gaussian()
        @test amplitude(m, (U=0.0, V=0.0)) == abs(visibility(m, (U=0.0, V=0.0)))
        @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m, (X=0.0, Y=0.0))

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(x[1]*Gaussian(), u, v, t, f))
        x = rand(1)
        foo(x)
        testgrad(foo, x)

        testmodel(m, 1024, 1e-5)
    end

    @testset "Disk" begin
        m = smoothed(Disk(), 0.25)
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))
        testmodel(m)

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(x[1]*Disk(), u, v, t, f))
        x = rand(1)
        foo(x)
        testgrad(foo, x)

    end

    @testset "SlashedDisk" begin
        m = smoothed(SlashedDisk(0.5), 0.25)
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))
        testmodel(m)

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(SlashedDisk(x[1]), u, v, t, f))
        x = rand(1)
        foo(x)
        testgrad(foo, x)
    end

    @testset "Pulses" begin
        m0 = BSplinePulse{0}()
        @inferred VLBISkyModels.visibility(m0, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m0, (X=0.0, Y=0.0))
        testmodel(m0)
        m1 = BSplinePulse{1}()
        testmodel(m1)
        @inferred VLBISkyModels.visibility(m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m1, (X=0.0, Y=0.0))
        m3 = BSplinePulse{3}()
        testmodel(m3)
        @inferred VLBISkyModels.visibility(m3, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m3, (X=0.0, Y=0.0))
        m4 = BicubicPulse()
        testmodel(m4)
        @inferred VLBISkyModels.visibility(m4, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m4, (X=0.0, Y=0.0))
        m5 = RaisedCosinePulse()
        testmodel(m5)
        @inferred VLBISkyModels.visibility(m5, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m5, (X=0.0, Y=0.0))
    end

    @testset "Butterworth" begin
        m1 = Butterworth{1}()
        testmodel(m1)
        @inferred VLBISkyModels.visibility(m1, (U=0.0, V=0.0))
        m2 = Butterworth{2}()
        testmodel(m2)
        @inferred VLBISkyModels.visibility(m2, (U=0.0, V=0.0))
        m3 = Butterworth{3}()
        testmodel(m3)
        @inferred VLBISkyModels.visibility(m3, (U=0.0, V=0.0))
    end


    @testset "Ring" begin
        m = smoothed(Ring(), 0.25)
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))
        testmodel(m, 2048)

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(x[1]*Ring(), u, v, t, f))
        x = rand(1)
        foo(x)
        testgrad(foo, x)

    end

    @testset "ParabolicSegment" begin
        m = ParabolicSegment()
        m2 = ParabolicSegment(2.0, 2.0)
        @test stretched(m, 2.0, 2.0) == m2
        @test ComradeBase.intensity_point(m, (X=0.0, Y=1.0)) != 0.0
        @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m, (X=0.0, Y=0.0))
        testmodel(m, 2400, 1e-2)

        # TODO why is this broken?
        # foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(ParabolicSegment(x[1], x[2]), u, v, t, f))
        # x = rand(2)
        # foo(x)
        # testgrad(foo, x)
    end


    @testset "MRing1" begin
        α = [0.25,]
        β = [0.1,]
        #test_rrule(VLBISkyModels.visibility_point, MRing(α, β), 0.5, 0.25)
        # We convolve it to remove some pixel effects
        m = convolved(MRing(α, β), stretched(Gaussian(), 0.1, 0.1))
        m2 = convolved(MRing(α[1], β[1]), stretched(Gaussian(), 0.1, 0.1))
        @test visibility(m, (U=0.1, V=0.1)) == visibility(m2, (U=0.1, V=0.1))
        testmodel(m, 2048, 1e-3)
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))

        testmodel(MRing(0.5, 0.0), 2048, 1e-3)

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(MRing(x[1], x[2]), u, v, t, f))
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

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(MRing((x[1],x[2]), (x[3], x[4])), u, v, t, f))
        x = rand(4)
        foo(x)
        testgrad(foo, x)
    end


    @testset "ConcordanceCrescent" begin
        m = ConcordanceCrescent(20.0, 10.0, 5.0, 0.5)
        testmodel(m, 2048, 1e-3)
        @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m, (X=0.0, Y=0.0))

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(ConcordanceCrescent(x[1], x[2], x[3], x[4]), u, v, t, f))
        x = rand(4)
        foo(x)
        testgrad(foo, x)
    end


    @testset "Crescent" begin
        m = smoothed(Crescent(5.0, 2.0, 1.0, 0.5), 1.0)
        testmodel(m,1024,1e-3)
        @inferred VLBISkyModels.visibility(m.m1, (U=0.0, V=0.0))
        @inferred VLBISkyModels.intensity_point(m.m1, (X=0.0, Y=0.0))

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(Crescent(x[1], x[2], x[3], x[4]), u, v, t, f))
        x = rand(4)
        foo(x)
        testgrad(foo, x)

    end


    @testset "ExtendedRing" begin
        mr = ExtendedRing(8.0)
        rad = 2.5*VLBISkyModels.radialextent(mr)
        m = modelimage(mr, IntensityMap(zeros(1024,1024), rad, rad), VLBISkyModels.FFTAlg(padfac=4))
        testmodel(m)
        @inferred VLBISkyModels.intensity_point(mr, (X=0.0, Y=0.0))
    end

    @testset "M87 model test" begin
        xopt = (rad = 21.895093363492155,
                wid = 2.1113838380637815,
                a = -0.3276141879612847,
                b = -0.13845264228109883,
                f = 0.4584364142294795,
                sig = 30.902344705962914,
                asy = 0.8036630375887827,
                pa = 0.6955748496122764,
                x = -43.84496132303754,
                y = -18.750141889035508
               )
        function model(θ)
            (;rad, wid, a, b, f, sig, asy, pa, x, y) = θ
            ring = f*smoothed(stretched(MRing((a,), (b,)), μas2rad(rad), μas2rad(rad)), μas2rad(wid))
            g = (1-f)*shifted(rotated(stretched(Gaussian(), μas2rad(sig)*asy, μas2rad(sig)), pa), μas2rad(x), μas2rad(y))
            return ring + g
        end

        m = model(xopt)
        testmodel(m)


        @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
        # k = keys(xopt)
        # foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(model(NamedTuple{k}(Tuple(x))), u, v, t, f))
        # x = collect(values(xopt))
        # foo(x)
        # testgrad(foo, x)
    end
end

@testset "ModelImage" begin
    m1 = Gaussian()
    m2 = ExtendedRing(10.0)
    mimg1 = modelimage(m1)
    mimg2 = modelimage(m2)

    show(mimg1)

    img = similar(mimg2.image)
    intensitymap!(img, m2)
    @test m1 == mimg1
    @test isapprox(maximum(parent(img) - parent(mimg2.image)), 0.0, atol=1e-8)
end



@testset "Modifiers" begin

    u = randn(100)*0.5
    v = randn(100)*0.5
    t = sort(rand(100)*0.5)
    f = fill(230e9, 100)

    ma = Gaussian()
    mb = ExtendedRing(8.0)
    @testset "Shifted" begin
        mas = shifted(ma, 0.1, 0.1)
        mbs = shifted(mb, 0.1, 0.1)
        testmodel(mas)
        testmodel(modelimage(mbs, IntensityMap(zeros(1024, 1024),
                                               2*VLBISkyModels.radialextent(mbs),
                                               2*VLBISkyModels.radialextent(mbs))))

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(shifted(ma, x[1], x[2]), u, v, t, f))
        x = rand(2)
        foo(x)
        testgrad(foo, x)
    end

    @testset "Renormed" begin
        m1 = 3.0*ma
        m2 = ma*3.0
        m2inv = ma/(1/3)
        p = (U=4.0, V = 0.0)
        @test visibility(m1, p) == visibility(m2, p)
        @test visibility(m2, p) == visibility(m2inv, p)
        mbs = 3.0*mb
        testmodel(m1)
        testmodel(modelimage(mbs, IntensityMap(zeros(1024, 1024),
                                               2.5*VLBISkyModels.radialextent(mbs),
                                               2.5*VLBISkyModels.radialextent(mbs))))

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(x[1]*ma, u, v, t, f))
        x = rand(1)
        foo(x)
        testgrad(foo, x)

    end

    @testset "Stretched" begin
        mas = stretched(ma, 5.0, 4.0)
        mbs = stretched(mb, 5.0, 4.0)
        testmodel(mas)
        testmodel(modelimage(mbs, IntensityMap(zeros(2024, 2024),
                                               2*VLBISkyModels.radialextent(mbs),
                                               2*VLBISkyModels.radialextent(mbs))), 1024, 1e-3)

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(stretched(ma, x[1], x[2]), u, v, t, f))
        x = rand(2)
        foo(x)
        testgrad(foo, x)
    end

    @testset "Rotated" begin
        mas = rotated(ma, π/3)
        mbs = rotated(mb, π/3)
        testmodel(mas)
        testmodel(modelimage(mbs, IntensityMap(zeros(1024, 1024),
                                               2*VLBISkyModels.radialextent(mbs),
                                               2*VLBISkyModels.radialextent(mbs))))

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(rotated(ma, x[1]), u, v, t, f))
        x = rand(1)
        foo(x)
        testgrad(foo, x)
    end

    @testset "AllMods" begin
        mas = rotated(stretched(shifted(ma, 0.5, 0.5), 5.0, 4.0), π/3)
        mas2 = modify(ma, Shift(0.5, 0.5), Stretch(5.0, 4.0), Rotate(π/4))
        @test typeof(mas2) === typeof(mas)
        mbs = rotated(stretched(shifted(mb, 0.5, 0.5), 5.0, 4.0), π/3)
        testmodel(mas)
        testmodel(modelimage(mbs, IntensityMap(zeros(2024, 2024),
                                               2*VLBISkyModels.radialextent(mbs),
                                               2*VLBISkyModels.radialextent(mbs))), 1024, 1e-3)

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(modify(ma, Shift(x[1], x[2]), Stretch(x[3], x[4]), Rotate(x[5]), Renormalize(x[6])), u, v, t, f))
        x = rand(6)
        foo(x)
        testgrad(foo, x)

    end
end

@testset "CompositeModels" begin
    m1 = Gaussian()
    m2 = ExtendedRing(8.0)

    u = randn(100)*0.5
    v = randn(100)*0.5
    t = sort(rand(100)*0.5)
    f = fill(230e9, 100)


    @testset "Add models" begin
        img = IntensityMap(
                zeros(1024, 1024),
                20.0,20.0
                )
        mt1 = m1 + m2
        mt2 = shifted(m1, 1.0, 1.0) + m2
        mt3 = shifted(m1, 1.0, 1.0) + 0.5*stretched(m2, 0.9, 0.8)
        mc = VLBISkyModels.components(mt1)
        @test mc[1] === m1
        @test mc[2] === m2
        @test flux(mt1) ≈ flux(m1) + flux(m2)

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(x[1]*stretched(Disk(), x[2], x[3]) + stretched(Ring(), x[4], x[4]), u, v, t, f))
        x = rand(4)
        foo(x)
        testgrad(foo, x)


        testmodel(modelimage(mt1, img))
        testmodel(modelimage(mt2, img))
        testmodel(modelimage(mt3, img))
    end

    @testset "Convolved models" begin
        img = IntensityMap(
                zeros(1024, 1024),
                20.0,20.0
                )
        mt1 = convolved(m1, m2)
        mt2 = convolved(shifted(m1, 1.0, 1.0), m2)
        mt3 = convolved(shifted(m1, 1.0, 1.0), 0.5*stretched(m2, 0.9, 0.8))
        mc = VLBISkyModels.components(mt1)
        @test mc[1] === m1
        @test mc[2] === m2

        testmodel(modelimage(mt1, img))
        testmodel(modelimage(mt2, img))
        testmodel(modelimage(mt3, img))

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(convolved(x[1]*stretched(Disk(), x[2], x[3]),stretched(Ring(), x[4], x[4])), u, v, t, f))
        x = rand(4)
        foo(x)
        testgrad(foo, x)

    end

    @testset "All composite" begin
        img = IntensityMap(
                zeros(1024, 1024),
                20.0,20.0
                )

        mt = m1 + convolved(m1, m2)
        mc = VLBISkyModels.components(mt)
        @test mc[1] === m1
        @test mc[2] === m1
        @test mc[3] === m2

        testmodel(modelimage(mt, img))

        foo(x) = sum(abs2, VLBISkyModels.visibilities_analytic(smoothed(x[1]*stretched(Disk(), x[2], x[3]), x[4]) + stretched(Ring(), x[5], x[4]), u, v, t, f))
        x = rand(5)
        foo(x)
        testgrad(foo, x)
    end
end

@testset "Multicomponent" begin
    u = randn(100)*0.5
    v = randn(100)*0.5
    t = sort(rand(100)*0.5)
    f = fill(230e9, 100)


    m = MultiComponentModel(Gaussian(), rand(10), randn(10), randn(10))
    @test amplitude(m, (U=0.0, V=0.0)) == abs(visibility(m, (U=0.0, V=0.0)))
    @inferred VLBISkyModels.visibility(m, (U=0.0, V=0.0))
    @inferred VLBISkyModels.intensity_point(m, (X=0.0, Y=0.0))

    foo(fl, x, y) = sum(abs2, VLBISkyModels.visibilities_analytic(MultiComponentModel(Gaussian(), fl, x, y), u, v, t, f))
    x = randn(10)
    y = randn(10)
    fl = rand(10)
    foo(fl, x, y)
    testgrad(foo, fl, x, y)

    testmodel(m, 1024, 1e-5)

end

@testset "PolarizedModel" begin
    u = randn(100)*0.5
    v = randn(100)*0.5
    t = sort(rand(100)*0.5)
    f = fill(230e9, 100)


    mI = stretched(MRing((0.2,), (0.1,)), 20.0, 20.0)
    mQ = 0.2*stretched(MRing((0.0,), (0.6,)), 20.0, 20.0)
    mU = 0.2*stretched(MRing((0.1,), (-0.6,)), 20.0, 20.0)
    mV = 0.0*stretched(MRing((0.0,), (-0.6,)), 20.0, 20.0)
    m = PolarizedModel(mI, mQ, mU, mV)
    @inferred visibility(m, (U=0.0, V=0.0))
    @inferred ComradeBase.intensity_point(m, (X=0.0, Y=0.0))


    mG = PolarizedModel(Gaussian(), Gaussian(), Gaussian(), Gaussian())
    cm = convolved(m, Gaussian())
    @test cm == convolved(m, mG)
    @inferred cm+mG
    show(m)

    p = (U = 0.005, V=0.01)
    v = visibility(m, p)
    @test m̆(v) ≈ m̆(m, p)
    @test mbreve(v) ≈ mbreve(m, p)

    g = GriddedKeys(imagepixels(60.0, 60.0, 128, 128))
    img = intensitymap(m, g)
    p0 = (X=g.X[64], Y=g.Y[64])
    @test linearpol(m, p0) ≈ linearpol(img[64, 64])
    @test mpol(m, p0) ≈ mpol(img[64, 64])
    @test polarization(m, p0) ≈ polarization(img[64, 64])
    @test fracpolarization(m, p0) ≈ fracpolarization(img[64, 64])
    @test evpa(m, p0) ≈ evpa(img[64, 64])
    map((x,y)->(@test x≈y), polellipse(m, p0), polellipse(img[64, 64]))


    I = IntensityMap(zeros(1024,1024), 100.0, 100.0)
    Q = similar(I)
    U = similar(I)
    V = similar(I)
    pimg1 = StokesIntensityMap(I,Q,U,V)
    intensitymap!(pimg1, m)
    pimg2 = intensitymap(m, 100.0, 100.0, 1024, 1024)
    @test isapprox(sum(abs, (stokes(pimg1, :I) .- stokes(pimg2, :I))), 0.0, atol=1e-12)
    @test isapprox(sum(abs, (stokes(pimg1, :Q) .- stokes(pimg2, :Q))), 0.0, atol=1e-12)
    @test isapprox(sum(abs, (stokes(pimg1, :U) .- stokes(pimg2, :U))), 0.0, atol=1e-12)
    @test isapprox(sum(abs, (stokes(pimg1, :V) .- stokes(pimg2, :V))), 0.0, atol=1e-12)

end


@testset "ContinuousImage Bspline0" begin
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π/8), 12.0, 12.0, 12, 12)
    cimg = ContinuousImage(img, BSplinePulse{0}())
    testmodel(modelimage(cimg, FFTAlg(padfac=4)), 1024, 1e-2)
    testft_cimg(cimg)
end

@testset "ContinuousImage BSpline1" begin
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π/8), 12.0, 12.0, 12, 12)
    cimg = ContinuousImage(img, BSplinePulse{1}())
    testmodel(modelimage(cimg, FFTAlg(padfac=4)), 1024, 1e-3)
    testft_cimg(cimg)
end

@testset "ContinuousImage BSpline3" begin
    img = intensitymap(rotated(stretched(Gaussian(), 2.0, 1.0), π/8), 12.0, 12.0, 12, 12)
    cimg = ContinuousImage(img, BSplinePulse{3}())
    testmodel(modelimage(cimg, FFTAlg(padfac=3)), 1024, 1e-3)
    testft_cimg(cimg)
end

@testset "ContinuousImage Bicubic" begin
    img = intensitymap(shifted(rotated(stretched(smoothed(Ring(), 0.5), 2.0, 1.0), π/8), 0.1, 0.1), 24.0, 24.0, 12, 12)
    cimg = ContinuousImage(img, BicubicPulse())
    testmodel(modelimage(cimg, FFTAlg(padfac=3)), 1024, 1e-3)
    testft_cimg(cimg)
end

@testset "methods " begin
    u1cp = 10e9*rand(100) .- 5e9
    v1cp = 10e9*rand(100) .- 5e9
    u2cp = 10e9*rand(100) .- 5e9
    v2cp = 10e9*rand(100) .- 5e9
    u3cp = -u1cp - u2cp
    v3cp = -v1cp - v2cp

    u1lc = 10e9*rand(100) .- 5e9
    v1lc = 10e9*rand(100) .- 5e9
    u2lc = 10e9*rand(100) .- 5e9
    v2lc = 10e9*rand(100) .- 5e9
    u3lc = 10e9*rand(100) .- 5e9
    v3lc = 10e9*rand(100) .- 5e9
    u4lc = -u1lc - u2lc - u3lc
    v4lc = -v1lc - v2lc - v3lc

    m = rotated(stretched(Gaussian(), μas2rad(2.0), μas2rad(1.0)), π/8)
    @test closure_phases(m, (U=u1cp, V=v1cp), (U=u2cp, V=v2cp), (U=u3cp, V=v3cp)) ≈ zero(u1cp)
    logclosure_amplitudes(m, (U=u1lc, V=v1lc), (U=u2lc, V=v2lc), (U=u3lc, V=v3lc), (U=u4lc, V=v4lc))

end

@testset "modelimage cache" begin
    img = intensitymap(rotated(stretched(Gaussian(), μas2rad(2.0), μas2rad(1.0)), π/8),
                       μas2rad(12.0), μas2rad(12.0), 24, 12)
    u1cp = 10e9*rand(100) .- 5e9
    v1cp = 10e9*rand(100) .- 5e9
    u2cp = 10e9*rand(100) .- 5e9
    v2cp = 10e9*rand(100) .- 5e9
    u3cp = -u1cp - u2cp
    v3cp = -v1cp - v2cp

    u1lc = 10e9*rand(100) .- 5e9
    v1lc = 10e9*rand(100) .- 5e9
    u2lc = 10e9*rand(100) .- 5e9
    v2lc = 10e9*rand(100) .- 5e9
    u3lc = 10e9*rand(100) .- 5e9
    v3lc = 10e9*rand(100) .- 5e9
    u4lc = -u1lc - u2lc - u3lc
    v4lc = -v1lc - v2lc - v3lc

    cimg = ContinuousImage(img, DeltaPulse())
    cache_nf = create_cache(NFFTAlg(u1cp, v1cp), img, DeltaPulse())
    cimg2 = ContinuousImage(img, cache_nf)
    cache_df = create_cache(DFTAlg(u1cp, v1cp), img, DeltaPulse())
    cimg3 = ContinuousImage(img, cache_df)

    mimg_nf = modelimage(cimg, cache_nf)
    mimg_df = modelimage(cimg, cache_df)

    vnf = visibilities(mimg_nf, (U=u1cp, V=v1cp))
    vdf = visibilities(mimg_df, (U=u1cp, V=v1cp))

    atol = 1e-5

    @test isapprox(maximum(abs, vnf-vdf), 0, atol=atol)


    @testset "nuft pullback" begin
        test_rrule(VLBISkyModels.nuft, cache_nf.plan ⊢ NoTangent(), complex.(parent(parent(img))))
    end
end

@testset "ContinuousImage" begin
    g = imagepixels(10.0, 10.0, 128, 128)
    data = rand(128, 128)
    img = ContinuousImage(IntensityMap(data, g), BSplinePulse{3}())
    img2 = ContinuousImage(data, 10.0, 10.0, 0.0, 0.0, BSplinePulse{3}())

    @test length(img) == length(data)
    @test size(img) == size(data)
    @test firstindex(img) == firstindex(data)
    @test lastindex(img) == lastindex(img)
    collect(iterate(img))
    @test eltype(img) == eltype(data)
    @test img[1,1] == data[1,1]
    @test img[1:5,1] == data[1:5,1]

    @test all(==(1), imagegrid(img) .== ComradeBase.grid(named_dims(axiskeys(img))))
    @test VLBISkyModels.axisdims(img) == axiskeys(img)

    @test g == imagepixels(img)
    @test VLBISkyModels.radialextent(img) ≈ 10.0/2

    @test convolved(img, Gaussian()) isa ContinuousImage
    @test convolved(Gaussian(), img) isa ContinuousImage

end
