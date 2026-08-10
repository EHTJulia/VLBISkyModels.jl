using Reactant
using LinearAlgebra
using Random
using VLBISkyModels
using VLBISkyModels: NFFT
using Test


function test_analytic(m, mr, gf, gfr)

    if ComradeBase.visanalytic(typeof(m)) isa ComradeBase.IsAnalytic
        vnf = visibilitymap(m, gf)
        vrf = @jit visibilitymap(mr, gfr)
        @test parent(vrf) ≈ vnf
    end

    return if ComradeBase.imanalytic(typeof(m)) isa ComradeBase.IsAnalytic
        img = intensitymap(m, gf)
        imgr = @jit intensitymap(mr, gfr)
        @test parent(imgr) ≈ img
    end
end

@testset "Reactant" begin
    @testset "VisibilityMap Parity" begin
        gim = imagepixels(10.0, 10.0, 128, 128)
        gimr = @jit(identity(gim))

        rast = rand(128, 128)
        rastr = Reactant.to_rarray(rast)

        mr = ContinuousImage(rastr, gimr, BSplinePulse{3}())
        m = ContinuousImage(rast, gim, BSplinePulse{3}())

        u = randn(10^2) / 5.0
        v = randn(10^2) / 5.0
        guv = UnstructuredDomain((U = u, V = v))

        gfn = FourierDualDomain(gim, guv, NFFTAlg())
        gfr = FourierDualDomain(gimr, Reactant.to_rarray(guv), VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-9))

        vnf = visibilitymap(m, gfn)
        vrf = @jit visibilitymap(mr, gfr)

        @test parent(vrf) ≈ vnf

        mrs = shifted(mr, ConcreteRNumber(1.0), ConcreteRNumber(1.0))
        ms = shifted(m, 1.0, 1.0)

        vnf_s = visibilitymap(ms, gfn)
        vrf_s = @jit visibilitymap(mrs, gfr)

        @test parent(vrf_s) ≈ vnf_s


    end

    # Large-M parity: M * w^2 * 16 above INTERP_WIDE_WS_LIMIT (128 MiB) routes
    # the type-2 interp through `_interp_wide` instead of the chunk loop.
    # Regression test for the wide-path miscompilation/segfault chain seen on
    # Reactant 0.2.271 (complex gather-from-concatenate folded to zeros, then
    # DotGeneralSimplify segfaulted on the zero-operand complex dot_general).
    @testset "VisibilityMap Parity wide path (large M)" begin
        gim = imagepixels(10.0, 10.0, 32, 32)
        gimr = @jit(identity(gim))

        rast = rand(32, 32)
        rastr = Reactant.to_rarray(rast)

        mr = ContinuousImage(rastr, gimr, BSplinePulse{3}())
        m = ContinuousImage(rast, gim, BSplinePulse{3}())

        rng = Random.MersenneTwister(42)
        # eps=1e-9 gives w=10, so M=90_000 puts the workspace at
        # 90_000 * 100 * 16 B = 144 MB > 128 MiB — the wide path.
        M = 90_000
        u = randn(rng, M) / 5.0
        v = randn(rng, M) / 5.0
        guv = UnstructuredDomain((U = u, V = v))

        gfn = FourierDualDomain(gim, guv, NFFTAlg())
        gfr = FourierDualDomain(gimr, Reactant.to_rarray(guv), VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-9))

        vnf = visibilitymap(m, gfn)
        vrf = @jit visibilitymap(mr, gfr)

        @test parent(vrf) ≈ vnf
    end

    @testset "PolExp2Map" begin
        gim = imagepixels(1.0, 1.0, 128, 128)
        gimr = @jit(identity(gim))

        a = randn(128, 128)
        b = randn(128, 128)
        c = randn(128, 128)
        d = randn(128, 128)

        am = Reactant.to_rarray(a)
        bm = Reactant.to_rarray(b)
        cm = Reactant.to_rarray(c)
        dm = Reactant.to_rarray(d)

        m = VLBISkyModels.PolExp2Map(a, b, c, d, gim)
        mr = @jit VLBISkyModels.PolExp2Map(am, bm, cm, dm, gimr)

        u = randn(10^2) / 5.0
        v = randn(10^2) / 5.0
        guv = UnstructuredDomain((U = u, V = v))

        gfn = FourierDualDomain(gim, guv, NFFTAlg())
        gfr = FourierDualDomain(gimr, Reactant.to_rarray(guv), VLBISkyModels.ReactantNUFFTAlg(; eps = 1.0e-12))

        pm = ContinuousImage(m, DeltaPulse())
        ppmr = ContinuousImage(Reactant.to_rarray(m), DeltaPulse())
        vnf = visibilitymap(pm, gfn)
        vrf = @jit(visibilitymap(ppmr, gfr))

        @test parent(vrf).I ≈ parent(vnf).I
        @test parent(vrf).Q ≈ parent(vnf).Q
        @test parent(vrf).U ≈ parent(vnf).U
        @test parent(vrf).V ≈ parent(vnf).V
    end

    @testset "Analytic Models" begin
        g = imagepixels(10.0, 10.0, 128, 128)
        gr = @jit(identity(g))

        guv = UnstructuredDomain((U = randn(10^2) / 5.0, V = randn(10^2) / 5.0))
        guvr = Reactant.to_rarray(guv)

        gf = FourierDualDomain(g, guv, NFFTAlg())
        gfr = FourierDualDomain(gr, guvr, VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-9))

        @testset "Gaussian" begin
            m = Gaussian()
            mr = Gaussian()
            test_analytic(m, mr, gf, gfr)
        end

        @testset "Modifed Gaussian" begin
            m = 5.0 * modify(Gaussian(), Stretch(1.0, 2.0), Rotate(π / 4), Shift(0.5, -0.5))
            mr = Reactant.to_rarray(m; track_numbers = true)
            test_analytic(m, mr, gf, gfr)
        end

        @testset "TBlob" begin
            m = TBlob(4.0)
            mr = Reactant.to_rarray(m; track_numbers = true)
            test_analytic(m, mr, gf, gfr)
        end

        @testset "ExtendedRing" begin
            m = ExtendedRing(4.0)
            mr = Reactant.to_rarray(m; track_numbers = true)
            test_analytic(m, mr, gf, gfr)
        end

        @testset "RingTemplate" begin
            m = RingTemplate(RadialDblPower(1.0, 2.0), AzimuthalUniform())
            mr = Reactant.to_rarray(m; track_numbers = true)
            test_analytic(m, mr, gf, gfr)

            m = RingTemplate(RadialDblPower(1.0, 2.0), AzimuthalCosine((0.5, 1.0), (-0.5, 0.5)))
            mr = Reactant.to_rarray(m; track_numbers = true)
            test_analytic(m, mr, gf, gfr)
        end

        # Other geometric models are currently broken due to the lack of bessel functions
        # TODO add:
        # MRing
        # Disk
        # SlashedDisk
        # Ring
        # Crescent
        # Pulses (this is because of the branches)
        # ParabolicSegment (missing erf)
    end


end
