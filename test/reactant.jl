using Reactant
using LinearAlgebra
using Random
using VLBISkyModels
using VLBISkyModels: NFFT
using Test

# Force the CPU backend so Reactant does not try to initialize a GPU client.
# On some machines (e.g. AMD/ROCm) probing the GPU during the first compilation
# crashes the whole process. These tests only check correctness, not the GPU.
Reactant.set_default_backend("cpu")


polarized_gaussian(σ) = PolarizedModel(stretched(Gaussian(), σ, σ), ZeroModel(), 0.2 * stretched(Gaussian(), σ, σ), ZeroModel())
polarized_numeric(σ, g) = baseimage(VLBISkyModels.visibilitymap_numeric(polarized_gaussian(σ), g))

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
        gim = spatialgrid(10.0, 10.0, 128, 128)
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

    # Above `INTERP_WIDE_WS_LIMIT` (M * w^2 * 16 bytes > 128 MiB) the type-2 interpolation
    # gathers each point's whole stencil at once: eps = 1e-9 gives w = 10, so M = 90 000
    # needs 144 MB.
    @testset "VisibilityMap Parity wide path (large M)" begin
        gim = spatialgrid(10.0, 10.0, 32, 32)
        rast = rand(32, 32)
        m = ContinuousImage(rast, gim, BSplinePulse{3}())
        mr = ContinuousImage(Reactant.to_rarray(rast), @jit(identity(gim)), BSplinePulse{3}())
        guv = UnstructuredDomain((U = randn(90_000) ./ 5, V = randn(90_000) ./ 5))
        gfn = FourierDualDomain(gim, guv, NFFTAlg())
        gfr = FourierDualDomain(@jit(identity(gim)), Reactant.to_rarray(guv), VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-9))
        @test parent(@jit visibilitymap(mr, gfr)) ≈ visibilitymap(m, gfn)
    end

    @testset "plane stack, wide path (large M)" begin
        g = spatialgrid(10.0, 10.0, 32, 32) ⊗ Fr([230.0e9, 345.0e9])
        cube = rand(32, 32, 2)
        m = ContinuousImage(IntensityMap(cube, g), BSplinePulse{3}())
        mr = ContinuousImage(IntensityMap(Reactant.to_rarray(cube), g), BSplinePulse{3}())
        guv = UnstructuredDomain((U = randn(90_000) ./ 5, V = randn(90_000) ./ 5, Fr = rand([230.0e9, 345.0e9], 90_000)))
        gfn = FourierDualDomain(g, guv, NFFTAlg())
        gfr = FourierDualDomain(g, Reactant.to_rarray(guv), VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-9))
        @test parent(@jit visibilitymap(mr, gfr)) ≈ parent(visibilitymap(m, gfn))
    end

    @testset "PolExp2Map" begin
        gim = spatialgrid(1.0, 1.0, 128, 128)
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

        @test vrf isa StokesMap
        for s in (:I, :Q, :U, :V)
            @test Array(baseimage(stokes(vrf, s))) ≈ baseimage(stokes(vnf, s))
        end
    end

    @testset "Polarized MultiDomainImage" begin
        ref = 230.0e9
        g = spatialgrid(10.0, 10.0, 16, 16)
        img = IntensityMap(FieldDimArray{StokesParams}(rand(16, 16, 4)), g)
        spec = PolySpectral((1.0, 0.1), ref, 0.01)
        cimg = MultiDomainImage(img, BSplinePulse{3}(), spec)
        gcube = RectiGrid((; X = g.X, Y = g.Y, Fr = [ref, 1.5 * ref]))
        guv = UnstructuredDomain(
            (; U = randn(40) ./ 5, V = randn(40) ./ 5, Fr = vcat(fill(ref, 20), fill(1.5 * ref, 20)))
        )
        vis = visibilitymap(cimg, FourierDualDomain(gcube, guv, NFFTAlg()))
        @test vis isa StokesMap

        gcuber = @jit(identity(gcube))
        cimgr = MultiDomainImage(Reactant.to_rarray(img), BSplinePulse{3}(), spec)
        gfr = FourierDualDomain(gcuber, Reactant.to_rarray(guv), VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-10))
        visr = @jit visibilitymap(cimgr, gfr)
        imr = @jit intensitymap(cimgr, gcuber)
        im = intensitymap(cimg, gcube)
        for s in (:I, :Q, :U, :V)
            @test Array(baseimage(stokes(visr, s))) ≈ baseimage(stokes(vis, s))
            @test Array(baseimage(stokes(imr, s))) ≈ baseimage(stokes(im, s))
        end
    end

    @testset "(Pt, Fr) visibility domain" begin
        ref = 230.0e9
        c = 299_792_458.0
        g = spatialgrid(10.0, 10.0, 16, 16)
        frs = [ref, 1.5 * ref]
        n = 30
        dpf = StructuredDomain((Pt(n), Fr(frs)); u = randn(n) ./ 10 .* c ./ ref, v = randn(n) ./ 10 .* c ./ ref)
        base = rand(16, 16)
        spec = PolySpectral(1.2, ref)
        gcube = g ⊗ Fr(frs)
        vis = visibilitymap(MultiDomainImage(IntensityMap(base, g), BSplinePulse{3}(), spec), FourierDualDomain(gcube, dpf, DFTAlg()))
        cimgr = MultiDomainImage(IntensityMap(Reactant.to_rarray(base), g), BSplinePulse{3}(), spec)
        gfr = FourierDualDomain(gcube, Reactant.to_rarray(dpf), VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-10))
        visr = @jit visibilitymap(cimgr, gfr)
        @test size(visr) == (n, 2)
        @test Array(baseimage(visr)) ≈ baseimage(vis)
    end

    @testset "time-variable images" begin
        pulse = BSplinePulse{3}()
        gs = spatialgrid(10.0, 10.0, 16, 16)
        ti = frames(Ti, [0.0, 1.5, 4.0], [1.0, 3.0, 5.0])
        gf = gs ⊗ ti ⊗ Fr([230.0e9, 345.0e9])
        ps = PolySpectral(-0.7, 230.0e9)
        # Planes in scattered order, and plane (0.5, 230 GHz) with a single point.
        tf = [
            (0.5, 230.0e9), (2.0, 345.0e9), (4.5, 230.0e9), (2.0, 230.0e9), (0.5, 345.0e9), (4.5, 345.0e9),
            (2.0, 345.0e9), (4.5, 230.0e9), (0.5, 345.0e9), (2.0, 230.0e9), (4.5, 345.0e9),
        ]
        dom = UnstructuredDomain((; U = randn(11) ./ 10, V = randn(11) ./ 10, Ti = first.(tf), Fr = last.(tf)))
        gfh = FourierDualDomain(gf, dom, DFTAlg())
        gfr = FourierDualDomain(gf, Reactant.to_rarray(dom), VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-10))
        cases = (
            ("per-frame base", a -> MultiDomainImage(IntensityMap(a, gs ⊗ ti), pulse, ps), rand(16, 16, 3)),
            ("PolyTemporal", a -> MultiDomainImage(IntensityMap(a, gs), pulse, PolyTemporal(0.2, 2.0), ps), rand(16, 16)),
            ("identity link", a -> MultiDomainImage(IntensityMap(a, gs), pulse, PolyTemporal(0.2, 2.0; link = identity), ps), rand(16, 16)),
        )
        for (lbl, mk, a) in cases
            @testset "$lbl" begin
                vf(a, g) = baseimage(visibilitymap(mk(a), g))
                imf(a, g) = baseimage(intensitymap(mk(a), g))
                ar = Reactant.to_rarray(a)
                @test Array(@jit vf(ar, gfr)) ≈ vf(a, gfh)
                @test Array(@jit imf(ar, gf)) ≈ imf(a, gf)
                hlo = string(@code_hlo vf(ar, gfr))
                @test !occursin("while", hlo)
                @test count("stablehlo.fft", hlo) == 1
                @test !occursin("while", string(@code_hlo imf(ar, gf)))
                lh(a) = sum(abs2, vf(a, gfh))
                lr(a, g) = sum(abs2, vf(a, g))
                gr(a, g) = Enzyme.gradient(Enzyme.Reverse, lr, a, Const(g))[1]
                dh = Enzyme.gradient(set_runtime_activity(Enzyme.Reverse), Const(lh), a)[1]
                @test Array(@jit gr(ar, gfr)) ≈ dh
            end
        end
    end

    @testset "sharded plane stack" begin
        ndev = length(Reactant.devices())
        if ndev < 4
            @warn "Sharded NUFFT tests skipped: start Julia with XLA_FLAGS=--xla_force_host_platform_device_count=4"
        else
            blocks(a, d) = sort!(unique(map(s -> s[d], a.sharding.device_to_array_slices)); by = first)
            nogather(f, args...) = !occursin("all-gather", repr(@code_xla shardy_passes = :to_mhlo_shardings f(args...)))
            mesh = Reactant.Sharding.Mesh(reshape(Reactant.devices()[1:4], 2, 2), (:t, :f))
            layout = ShardLayout(mesh; Ti = :t, Fr = :f)
            pulse = BSplinePulse{3}()
            gs = spatialgrid(10.0, 10.0, 16, 16)
            ti = frames(Ti, [0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
            frs = [230.0e9, 345.0e9]
            g = gs ⊗ ti ⊗ Fr(frs)
            alg = VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-10)
            n = 200
            dom = UnstructuredDomain((; U = randn(n) ./ 10, V = randn(n) ./ 10, Ti = 4 .* rand(n), Fr = rand(frs, n)))
            gfh = FourierDualDomain(g, dom, DFTAlg())
            sgf = shard(FourierDualDomain(g, Reactant.to_rarray(dom), alg), layout)
            @test imgdomain(sgf) === g

            onmesh(a, gr, lay) = lay === nothing ?
                Reactant.to_rarray(a; sharding = Reactant.Sharding.Replicated(mesh)) :
                baseimage(shard(IntensityMap(a, gr), lay))
            gpf = gs ⊗ ti
            cases = (
                ("stored cube", a -> ContinuousImage(IntensityMap(a, g), pulse), rand(16, 16, 4, 2), g, layout, (3, 4)),
                ("per-frame base", a -> MultiDomainImage(IntensityMap(a, gpf), pulse, PolySpectral(-0.7, 230.0e9)), rand(16, 16, 4), gpf, ShardLayout(mesh; Ti = :t), (3,)),
                ("time and frequency families", a -> MultiDomainImage(IntensityMap(a, gs), pulse, PolyTemporal(0.2, 2.0), PolySpectral(-0.7, 230.0e9)), rand(16, 16), gs, nothing, ()),
            )
            for (lbl, mk, a, ga, lay, sdims) in cases
                @testset "$lbl" begin
                    vf(a, g) = baseimage(visibilitymap(mk(a), g))
                    loss(a, g) = sum(abs2, vf(a, g))
                    function grad!(da, a, g)
                        Enzyme.autodiff(Enzyme.Reverse, loss, Active, Duplicated(a, da), Const(g))
                        return da
                    end
                    ar = onmesh(a, ga, lay)
                    @test Array(@jit vf(ar, sgf)) ≈ vf(a, gfh)
                    @test nogather(vf, ar, sgf)
                    dar = onmesh(zero(a), ga, lay)
                    dr = @jit grad!(dar, ar, sgf)
                    dh = Enzyme.gradient(set_runtime_activity(Enzyme.Reverse), Const(x -> loss(x, gfh)), a)[1]
                    @test Array(dr) ≈ dh
                    for d in sdims
                        @test length(blocks(dr, d)) == 2
                    end
                    @test nogather(grad!, dar, ar, sgf)
                end
            end

            @testset "polarized stored cube" begin
                a = rand(16, 16, 4, 2, 4)
                vf(a, gd) = parent(baseimage(visibilitymap(ContinuousImage(IntensityMap(FieldDimArray{StokesParams}(a), g), pulse), gd)))
                ar = Reactant.to_rarray(a; sharding = Reactant.Sharding.DimsSharding(mesh, (3, 4), (:t, :f)))
                @test Array(@jit vf(ar, sgf)) ≈ vf(a, gfh)
                @test nogather(vf, ar, sgf)
            end

            @testset "(Pt, Fr) domain split along Fr" begin
                np = 60
                dpf = StructuredDomain((Pt(np), Fr(frs)); U = randn(np, 2) ./ 10, V = randn(np, 2) ./ 10, Ti = repeat(4 .* rand(np), 1, 2))
                sg = shard(FourierDualDomain(g, Reactant.to_rarray(dpf), alg), layout)
                vf(a, g) = baseimage(visibilitymap(ContinuousImage(IntensityMap(a, imgdomain(g)), pulse), g))
                a = rand(16, 16, 4, 2)
                ar = baseimage(shard(IntensityMap(a, g), layout))
                v = @jit vf(ar, sg)
                @test Array(v) ≈ vf(a, FourierDualDomain(g, dpf, DFTAlg()))
                # The (Pt, Fr) result is a reshape of the flat points, which are in Fr-major order.
                @test blocks(parent(v), 1) == [1:np, (np + 1):(2np)]
                @test nogather(vf, ar, sg)
            end
        end
    end

    @testset "Polarized plus unpolarized" begin
        g = spatialgrid(10.0, 10.0, 32, 32)
        img = IntensityMap(FieldDimArray{StokesParams}(rand(32, 32, 4)), g)
        guv = UnstructuredDomain((; U = randn(50) ./ 5, V = randn(50) ./ 5))
        gf = FourierDualDomain(g, guv, NFFTAlg())
        gfr = FourierDualDomain(@jit(identity(g)), Reactant.to_rarray(guv), VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-10))
        m = ContinuousImage(img, BSplinePulse{3}()) + 0.5 * Gaussian()
        mr = ContinuousImage(Reactant.to_rarray(img), BSplinePulse{3}()) + 0.5 * Gaussian()
        vr = @jit visibilitymap(mr, gfr)
        @test vr isa StokesMap
        @test Array(parent(baseimage(vr))) ≈ parent(baseimage(visibilitymap(m, gf)))
        @test !occursin("scatter", string(@code_hlo visibilitymap(mr, gfr)))
    end

    @testset "Polarized numeric FFT" begin
        guv = VLBISkyModels.uvgrid(spatialgrid(10.0, 10.0, 32, 32))
        guvr = @jit(identity(guv))
        σ = Reactant.ConcreteRNumber(2.0)
        vr = @jit polarized_numeric(σ, guvr)
        @test Array(parent(vr)) ≈ parent(polarized_numeric(2.0, guv))
        @test count("stablehlo.fft", string(@code_hlo optimize = true polarized_numeric(σ, guvr))) == 1
    end

    @testset "Analytic Models" begin
        g = spatialgrid(10.0, 10.0, 128, 128)
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
