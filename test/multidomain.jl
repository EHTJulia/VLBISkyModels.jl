function create_domains(Nx, alg; Nt = 0, Nf = 0, fov = 12.0, swap_tf = false)
    X = Y = range(-fov, fov; length = Nx)
    Ti = Nt > 0 ? sort(10 * rand(Nt)) : Float64[]
    Fr = Nf > 0 ? sort(1.0e11 * rand(Nf)) : Float64[]

    if isempty(Ti) && isempty(Fr)
        imgdomain = RectiGrid((; X, Y))
    elseif isempty(Ti)
        imgdomain = RectiGrid((; X, Y, Fr))
    elseif isempty(Fr)
        imgdomain = RectiGrid((; X, Y, Ti))
    else
        # Choose the ordering based on swap_tf flag
        imgdomain = swap_tf ? RectiGrid((; X, Y, Fr, Ti)) : RectiGrid((; X, Y, Ti, Fr))
    end

    dx, dy = pixelsizes(imgdomain)
    U_vals = fftshift(fftfreq(500, 1 / dx))
    V_vals = fftshift(fftfreq(500, 1 / dy))

    if !isempty(Ti) && !isempty(Fr)
        if swap_tf
            # (X, Y, Fr, Ti)
            # Repeat U and V to match Fr dimensions
            U_repeated = repeat(vec(U_vals); outer = (length(Fr)))
            V_repeated = repeat(vec(V_vals); outer = (length(Fr)))
            Fr_repeated = repeat(Fr; inner = (Int(length(vec(U_vals)))))
            # Repeat U and V and Fr to match Ti dimensions
            U_repeated = repeat(U_repeated; outer = (length(Ti)))
            V_repeated = repeat(V_repeated; outer = (length(Ti)))
            Fr_repeated = repeat(Fr_repeated; outer = (length(Ti)))
            Ti_repeated = repeat(Ti; inner = (Int(length(U_repeated) / length(Ti))))
            visdomain = UnstructuredDomain(
                (;
                    U = U_repeated, V = V_repeated, Fr = Fr_repeated,
                    Ti = Ti_repeated,
                )
            )
        else
            # (X, Y, Ti, Fr)
            # Repeat U and V to match Ti dimensions
            U_repeated = repeat(vec(U_vals); outer = (length(Ti)))
            V_repeated = repeat(vec(V_vals); outer = (length(Ti)))
            Ti_repeated = repeat(Ti; inner = (Int(length(vec(U_vals)))))
            # Repeat U and V and Ti to match Fr dimensions
            U_repeated = repeat(U_repeated; outer = (length(Fr)))
            V_repeated = repeat(V_repeated; outer = (length(Fr)))
            Ti_repeated = repeat(Ti_repeated; outer = (length(Fr)))
            Fr_repeated = repeat(Fr; inner = (Int(length(U_repeated) / length(Fr))))
            visdomain = UnstructuredDomain(
                (;
                    U = U_repeated, V = V_repeated, Ti = Ti_repeated,
                    Fr = Fr_repeated,
                )
            )
        end
    elseif !isempty(Ti)
        # (X, Y, Ti)
        U_repeated = repeat(vec(U_vals); outer = (length(Ti)))
        V_repeated = repeat(vec(V_vals); outer = (length(Ti)))
        Ti_repeated = repeat(Ti; inner = (Int(length(vec(U_vals)))))
        visdomain = UnstructuredDomain((; U = U_repeated, V = V_repeated, Ti = Ti_repeated))
    elseif !isempty(Fr)
        # (X, Y, Fr)
        U_repeated = repeat(vec(U_vals); outer = (length(Fr)))
        V_repeated = repeat(vec(V_vals); outer = (length(Fr)))
        Fr_repeated = repeat(Fr; inner = (Int(length(vec(U_vals)))))
        visdomain = UnstructuredDomain((; U = U_repeated, V = V_repeated, Fr = Fr_repeated))
    else
        # (X, Y)
        visdomain = UnstructuredDomain((; U = vec(U_vals), V = vec(V_vals)))
    end
    p = FourierDualDomain(imgdomain, visdomain, alg)
    return p
end

# Function to calculate visibilities
function foo4D(x, p)
    cimg = ContinuousImage(IntensityMap(x, VLBISkyModels.imgdomain(p)), DeltaPulse())
    vis = VLBISkyModels.visibilitymap(cimg, p)
    return sum(abs2, vis)
end

# Test function to check autodiff
function check4dautodiff(p, x, dx)
    Enzyme.autodiff(
        set_runtime_activity(Enzyme.Reverse), foo4D, Active,
        Duplicated(x, fill!(dx, 0)), Const(p)
    )
    return nothing
end

# Function to test gradient against finite differences
function test4Dgrad(p, x)
    finite_dx = grad(central_fdm(5, 1), x -> foo4D(x, p), x)[1]
    return finite_dx
end

# Check autodiff with Enzyme and compare grad
@testset "Enzyme autodiff for 4D NFFT/DFT" begin
    # Example usage in test cases
    Nx, Nt, Nf = 24, 2, 2
    x = randn(Nx, Nx, Nt, Nf)
    dx = zeros(Nx, Nx, Nt, Nf)

    @testset "NFFT.jl" begin
        alg = NFFTAlg()
        pnf = create_domains(Nx, alg; Nt = Nt, Nf = Nf)
        check4dautodiff(pnf, x, dx)
        finite_dx = test4Dgrad(pnf, x)
        @test isapprox(dx, finite_dx, atol = 1.0e-2)
    end

    @testset "FINUFFT" begin
        alg = FINUFFTAlg()
        pnf = create_domains(Nx, alg; Nt = Nt, Nf = Nf)
        check4dautodiff(pnf, x, dx)
        finite_dx = test4Dgrad(pnf, x)
        @test isapprox(dx, finite_dx, atol = 1.0e-2)
    end

    @testset "NonuniformFFTs" begin
        alg = NonuniformFFTsAlg()
        pnf = create_domains(Nx, alg; Nt = Nt, Nf = Nf)
        check4dautodiff(pnf, x, dx)
        finite_dx = test4Dgrad(pnf, x)
        @test isapprox(dx, finite_dx, atol = 1.0e-2)
    end


    @testset "DFT" begin
        alg = DFTAlg()
        pnf = create_domains(Nx, alg; Nt = Nt, Nf = Nf)
        check4dautodiff(pnf, x, dx)
        finite_dx = test4Dgrad(pnf, x)
        @test isapprox(dx, finite_dx, atol = 1.0e-2)
    end
end

# Time complexity tests
# function testtimecomplexity(Nx, Nt1, Nt2, Nf1, Nf2, alg)
#     p1 = create_domains(Nx, alg; Nt=Nt1, Nf=Nf1)
#     cimg1 = ContinuousImage(IntensityMap(randn(Nx, Nx, Nt1, Nf1),
#                                          VLBISkyModels.imgdomain(p1)),
#                             BSplinePulse{3}())
#     t1 = @benchmark VLBISkyModels.visibilitymap_numeric($cimg1, $p1)
#     median_t1 = median(t1).time / 1e6

#     p2 = create_domains(Nx, alg; Nt=Nt2, Nf=Nf2)
#     cimg2 = ContinuousImage(IntensityMap(randn(Nx, Nx, Nt2, Nf2),
#                                          VLBISkyModels.imgdomain(p2)),
#                             BSplinePulse{3}())
#     t2 = @benchmark VLBISkyModels.visibilitymap_numeric($cimg2, $p2)
#     median_t2 = median(t2).time / 1e6

#     return median_t2 / median_t1
# end

# @testset "Check time complexity for time and freq image FT" begin
#     @test isapprox(testtimecomplexity(24, 1, 2, 1, 2, NFFTAlg()), 4.0, atol=0.5)
#     @test isapprox(testtimecomplexity(24, 1, 2, 1, 1, NFFTAlg()), 2.0, atol=0.5)
#     @test isapprox(testtimecomplexity(24, 1, 1, 1, 2, NFFTAlg()), 2.0, atol=0.5)
# end

function rotating4dgaussian(p)
    # Elliptical gaussians rotating with a constant stretch and varying rotation
    gaussians = [
        modify(
                Gaussian(), Stretch(2, 1), Shift(2.0, 1.0),
                Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Ti) + π / 4),
                Renormalize(1.0)
            ) for (i, t) in enumerate(p.imgdomain.Ti)
    ]
    intensity_maps = [
        intensitymap(
                mpr,
                RectiGrid(
                    (;
                        X = p.imgdomain.X, Y = p.imgdomain.Y, Ti = [t],
                        Fr = p.imgdomain.Fr,
                    )
                )
            )
            for (t, mpr) in zip(p.imgdomain.Ti, gaussians)
    ]
    combined_img = cat(intensity_maps...; dims = 3)  # Concatenate along the third dimension (Ti)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function test4dgaussiansft(Nx, Nt, alg)
    p = create_domains(Nx, alg; Nt = Nt, Nf = 1)
    cimg, gaussians = rotating4dgaussian(p)
    vis_numeric = VLBISkyModels.visibilitymap(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, t) in enumerate(p.imgdomain.Ti)
        imgdomain_analytic = RectiGrid(
            (;
                X = p.imgdomain.X, Y = p.imgdomain.Y, Ti = [t],
                Fr = [p.imgdomain.Fr[1]],
            )
        )
        visdomain_analytic = p.visdomain[Ti = t, Fr = p.imgdomain.Fr[1]]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, alg)
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap_analytic(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol = 1.0e-3)
end

function test4dft_individual(Nx, Nt, alg)
    p = create_domains(Nx, alg; Nt = Nt, Nf = 1)
    cimg, gaussians = rotating4dgaussian(p)
    vis_numeric = VLBISkyModels.visibilitymap(cimg, p)
    vis_ind = similar(vis_numeric, 0)

    for (i, t) in enumerate(p.imgdomain.Ti)
        imgdomain_ind = RectiGrid(
            (;
                X = p.imgdomain.X, Y = p.imgdomain.Y, Ti = [t],
                Fr = [p.imgdomain.Fr[1]],
            )
        )
        visdomain_ind = p.visdomain[Ti = t, Fr = p.imgdomain.Fr[1]]
        p_ind = FourierDualDomain(imgdomain_ind, visdomain_ind, alg)
        img = intensitymap(gaussians[i], imgdomain_ind)
        cimg = ContinuousImage(img, BSplinePulse{3}())
        vis_ind_t = VLBISkyModels.visibilitymap(cimg, p_ind)
        append!(vis_ind, vis_ind_t)
    end
    return vis_numeric == vis_ind
end

function rotating4dgaussian_swap(p)
    # Elliptical gaussians rotating with a constant stretch and varying rotation
    gaussians = [
        modify(
                Gaussian(), Stretch(2, 1), Shift(2.0, 1.0),
                Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Ti) + π / 4),
                Renormalize(1.0)
            ) for (i, t) in enumerate(p.imgdomain.Ti)
    ]
    intensity_maps = [
        intensitymap(
                mpr,
                RectiGrid(
                    (;
                        X = p.imgdomain.X, Y = p.imgdomain.Y,
                        Fr = p.imgdomain.Fr, Ti = [t],
                    )
                )
            )
            for (t, mpr) in zip(p.imgdomain.Ti, gaussians)
    ]
    combined_img = cat(intensity_maps...; dims = 4)  # Concatenate along the fourth dimension (Ti)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function test4dgaussiansft_swap(Nx, Nt, alg)
    p = create_domains(Nx, alg; Nt = Nt, Nf = 1, swap_tf = true)
    cimg, gaussians = rotating4dgaussian_swap(p)
    vis_numeric = VLBISkyModels.visibilitymap_numeric(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, t) in enumerate(p.imgdomain.Ti)
        imgdomain_analytic = RectiGrid(
            (;
                X = p.imgdomain.X, Y = p.imgdomain.Y,
                Fr = [p.imgdomain.Fr[1]], Ti = [t],
            )
        )
        visdomain_analytic = p.visdomain[Fr = p.imgdomain.Fr[1], Ti = t]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, alg)
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol = 1.0e-3)
end

function rotating3dgaussian(p)
    # Elliptical gaussians rotating with a constant stretch and varying rotation
    gaussians = [
        modify(
                Gaussian(), Stretch(2, 1), Shift(2.0, 1.0),
                Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Ti) + π / 4),
                Renormalize(1.0)
            ) for (i, t) in enumerate(p.imgdomain.Ti)
    ]
    intensity_maps = [
        intensitymap(
                mpr,
                RectiGrid((; X = p.imgdomain.X, Y = p.imgdomain.Y, Ti = [t]))
            )
            for (t, mpr) in zip(p.imgdomain.Ti, gaussians)
    ]
    combined_img = cat(intensity_maps...; dims = 3)  # Concatenate along the third dimension (Ti)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function test3dgaussians(Nx, Nt, alg)
    p = create_domains(Nx, alg; Nt = Nt)
    cimg, gaussians = rotating3dgaussian(p)
    vis_numeric = VLBISkyModels.visibilitymap_numeric(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, t) in enumerate(p.imgdomain.Ti)
        imgdomain_analytic = RectiGrid((; X = p.imgdomain.X, Y = p.imgdomain.Y, Ti = [t]))
        visdomain_analytic = p.visdomain[Ti = t]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, alg)
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol = 1.0e-3)
end

function freqgaussians(p)
    gaussians = [
        modify(
                Gaussian(), Stretch(2, 1), Shift(2.0, 1.0),
                Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Fr) + π / 4),
                Renormalize(1.0)
            ) for (i, fr) in enumerate(p.imgdomain.Fr)
    ]
    intensity_maps = [
        intensitymap(
                mpr,
                RectiGrid((; X = p.imgdomain.X, Y = p.imgdomain.Y, Fr = [fr]))
            )
            for (fr, mpr) in zip(p.imgdomain.Fr, gaussians)
    ]
    combined_img = cat(intensity_maps...; dims = 3)  # Concatenate along the third dimension (Fr)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function test3dgaussians_freq(Nx, Nf, alg)
    p = create_domains(Nx, alg; Nf = Nf)
    cimg, gaussians = freqgaussians(p)

    vis_numeric = VLBISkyModels.visibilitymap(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, fr) in enumerate(p.imgdomain.Fr)
        imgdomain_analytic = RectiGrid((; X = p.imgdomain.X, Y = p.imgdomain.Y, Fr = [fr]))
        visdomain_analytic = p.visdomain[Fr = fr]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, alg)
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap_analytic(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol = 1.0e-3)
end

function test2dgaussian(Nx, alg)
    p = create_domains(Nx, alg)

    gaussian = modify(
        Gaussian(), Stretch(2, 1), Shift(2.0, 1.0), Rotate(π / 4),
        Renormalize(1.0)
    )
    intensity_map = intensitymap(gaussian, RectiGrid((; X = p.imgdomain.X, Y = p.imgdomain.Y)))
    cimg = ContinuousImage(intensity_map, BSplinePulse{3}())

    vis_numeric = VLBISkyModels.visibilitymap(cimg, p)

    imgdomain_analytic = RectiGrid((; X = p.imgdomain.X, Y = p.imgdomain.Y))
    visdomain_analytic = p.visdomain
    p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, alg)
    vis_analytic = VLBISkyModels.visibilitymap(gaussian, p_analytic)

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol = 1.0e-3)
end

@testset "3D/4D ContinuousImage FT Correctness" begin
    @test test4dgaussiansft(1024, 10, NFFTAlg())
    @test test4dgaussiansft_swap(1024, 10, NFFTAlg())
    @test test4dft_individual(1024, 10, NFFTAlg())
    @test test3dgaussians(1024, 10, NFFTAlg())
    @test test3dgaussians_freq(1024, 4, NFFTAlg())
    @test test2dgaussian(1024, NFFTAlg())

    @test test4dgaussiansft(1024, 10, FINUFFTAlg())
    @test test4dgaussiansft_swap(1024, 10, FINUFFTAlg())
    @test test4dft_individual(1024, 10, FINUFFTAlg())
    @test test3dgaussians(1024, 10, FINUFFTAlg())
    @test test3dgaussians_freq(1024, 4, FINUFFTAlg())
    @test test2dgaussian(1024, FINUFFTAlg())

    @test test4dgaussiansft(512, 2, DFTAlg())
    @test test4dgaussiansft_swap(512, 2, DFTAlg())
    @test test4dft_individual(512, 2, DFTAlg())
    @test test3dgaussians(512, 2, DFTAlg())
    @test test3dgaussians_freq(512, 2, DFTAlg())
    @test test2dgaussian(512, DFTAlg())
end

@testset "Multidomain models" begin
    @testset "PolySpectral" begin
        ts = MultiDomainParams(1.0, PolySpectral(1.0, 230.0, -1.0))
        @test ts((; Fr = 230.0)) ≈ 0.0
        @test ts((; Fr = 345.0)) ≈ 0.5

        ts2 = MultiDomainParams(1.0, PolySpectral((0.0, 1.0), 230.0))
        @test ts2((; Fr = 230.0)) ≈ 1.0
        @test ts2((; Fr = 345.0)) ≈ 1.0 * exp(log(1.5)^2)
    end

    function test_modifier(m, m230, m345, gfr)
        gXY = RectiGrid((; X = gfr.imgdomain.X, Y = gfr.imgdomain.Y))
        img = intensitymap(m, gfr)
        img230 = intensitymap(m230, gXY)
        img345 = intensitymap(m345, gXY)
        @test img[Fr = 1] ≈ img230 atol = 1.0e-8
        @test img[Fr = 2] ≈ img345 atol = 1.0e-8

        vmf = visibilitymap(m, gfr)
        v230 = visibilitymap(m230, gfr)[1:25]
        v345 = visibilitymap(m345, gfr)[26:50]
        @test vmf[1:25] ≈ v230 atol = 1.0e-8
        @test vmf[26:50] ≈ v345 atol = 1.0e-8
    end

    @testset "Modifiers Multidomain" begin
        gXY = imagepixels(40.0, 40.0, 256, 256)
        g = RectiGrid((; X = gXY.X, Y = gXY.Y, Fr = [230.0e9, 345.0e9]))
        u = randn(50) .* 0.25
        v = randn(50) .* 0.25
        ti = range(1.0, 3.0; length = 50)
        fr = vcat(fill(230.0e9, 25), fill(345.0e9, 25))
        guv = UnstructuredDomain((; U = u, V = v, Fr = fr, Ti = ti))
        gfr = FourierDualDomain(g, guv, NFFTAlg())

        @testset "Stretch" begin
            ts = MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9))
            m1 = modify(Gaussian(), Stretch(ts, 1.0))
            mn = modify(ExtendedRing(8.0), Stretch(ts, 1.0))
            test_modifier(m1, Gaussian(), modify(Gaussian(), Stretch(1.5, 1.0)), gfr)
            test_modifier(
                mn, ExtendedRing(8.0),
                modify(ExtendedRing(8.0), Stretch(1.5, 1.0)),
                gfr
            )

            m1 = modify(Gaussian(), Stretch(1.0, ts))
            mn = modify(TBlob(8.0), Stretch(1.0, ts))
            test_modifier(m1, Gaussian(), modify(Gaussian(), Stretch(1.0, 1.5)), gfr)
            test_modifier(
                mn, TBlob(8.0),
                modify(TBlob(8.0), Stretch(1.0, 1.5)),
                gfr
            )

            m1 = modify(Gaussian(), Stretch(ts, ts))
            mn = modify(ExtendedRing(8.0), Stretch(ts, ts))
            test_modifier(m1, Gaussian(), modify(Gaussian(), Stretch(1.5, 1.5)), gfr)
            test_modifier(
                mn, ExtendedRing(8.0),
                modify(ExtendedRing(8.0), Stretch(1.5, 1.5)),
                gfr
            )
        end

        @testset "Rotate" begin
            RM = 1.0
            mb = modify(Gaussian(), Stretch(2.0, 1.0))
            mbn = modify(ExtendedRing(8.0), Stretch(2.0, 1.0))
            # zeropoint the RM at 230 GHz, so the base RM of 1 is the value there
            tev = MultiDomainParams(1.0, PolySpectral(2.0, 230.0e9, -RM))
            m1 = modify(mb, Rotate(tev))
            mn = modify(mbn, Rotate(tev))
            test_modifier(m1, mb, modify(mb, Rotate(RM * (345 / 230)^2 - RM)), gfr)
            test_modifier(mn, mbn, modify(mbn, Rotate(RM * (345 / 230)^2 - RM)), gfr)
        end

        @testset "Shift" begin
            mb = Gaussian()
            mbn = ExtendedRing(8.0)
            ts = MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9, -1.0))
            m1 = modify(mb, Shift(ts, 0.0))
            mn = modify(mbn, Shift(ts, 0.0))
            test_modifier(m1, mb, modify(mb, Shift(0.5, 0.0)), gfr)
            test_modifier(mn, mbn, modify(mbn, Shift(0.5, 0.0)), gfr)
        end

        @testset "Renormalize" begin
            mb = Gaussian()
            mbn = ExtendedRing(8.0)
            ts = MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9))
            m1 = ts * mb
            mn = ts * mbn
            test_modifier(m1, mb, 1.5 * mb, gfr)
            test_modifier(mn, mbn, 1.5 * mbn, gfr)
        end

        @testset "Multi modifiers" begin
            mb = Gaussian()
            mbn = ExtendedRing(8.0)
            tss = MultiDomainParams(1.0, PolySpectral(1.0, 345.0e9))
            tsx = MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9, -1.0))
            tsr = MultiDomainParams(1.0, PolySpectral(1.0, 345.0e9, -1.0))

            m1 = modify(Gaussian(), Stretch(tss, 1.0), Shift(tsx, 0.0), Rotate(tsr))
            test_modifier(
                m1,
                modify(
                    Gaussian(),
                    Stretch(tss((; Fr = 230.0e9)), 1.0),
                    Shift(tsx((; Fr = 230.0e9)), 0.0),
                    Rotate(tsr((; Fr = 230.0e9)))
                ),
                modify(
                    Gaussian(),
                    Stretch(tss((; Fr = 345.0e9)), 1.0),
                    Shift(tsx((; Fr = 345.0e9)), 0.0),
                    Rotate(tsr((; Fr = 345.0e9)))
                ),
                gfr
            )
        end

        gfr = nothing
        GC.gc()
    end

    @testset "Add model" begin
        gXY = imagepixels(40.0, 40.0, 256, 256)
        g = RectiGrid((; X = gXY.X, Y = gXY.Y, Fr = [230.0e9, 345.0e9]))
        u = randn(50) .* 0.25
        v = randn(50) .* 0.25
        ti = range(1.0, 3.0; length = 50)
        fr = vcat(fill(230.0e9, 25), fill(345.0e9, 25))
        guv = UnstructuredDomain((; U = u, V = v, Fr = fr, Ti = ti))
        gfr = FourierDualDomain(g, guv, NFFTAlg())

        ts = MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9))
        m1 = modify(Gaussian(), Stretch(ts))
        m2 = ExtendedRing(8.0)
        ts3 = MultiDomainParams(8.0, PolySpectral(1.0, 230.0e9)) # scalar base lives in MultiDomainParams
        m3 = TBlob(ts3)

        test_modifier(m1 + m2, Gaussian() + m2, modify(Gaussian(), Stretch(1.5)) + m2, gfr)
        test_modifier(m1 + m1, 2 * Gaussian(), modify(Gaussian(), Stretch(1.5)) * 2, gfr)
        test_modifier(
            m1 + m3, Gaussian() + TBlob(8.0),
            modify(Gaussian(), Stretch(1.5)) + TBlob(8 * 1.5),
            gfr
        )

        gfr = nothing
        GC.gc()
    end

    @testset "Convolution Multdomain" begin
        @testset "Frequency only" begin
            m1 = modify(Gaussian(), Stretch(1.0))
            m2 = modify(Gaussian(), Stretch(MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9))))

            mtr230 = modify(Gaussian(), Stretch(sqrt(2)))
            mtr345 = modify(Gaussian(), Stretch(sqrt(1 + (345 / 230)^2)))
            gXY = imagepixels(20.0, 20.0, 256, 256)
            g = RectiGrid((; X = gXY.X, Y = gXY.Y, Fr = [230.0e9, 345.0e9]))
            @test intensitymap(convolved(m1, m2), g)[Fr = 1] ≈ intensitymap(mtr230, gXY) atol = 1.0e-8
            @test intensitymap(convolved(m1, m2), g)[Fr = 2] ≈ intensitymap(mtr345, gXY) atol = 1.0e-8

            u = randn(50) .* 0.25
            v = randn(50) .* 0.25
            ti = range(1.0, 3.0; length = 50)
            fr = vcat(fill(230.0e9, 25), fill(345.0e9, 25))
            guv = UnstructuredDomain((; U = u, V = v, Fr = fr, Ti = ti))
            vmf = visibilitymap(convolved(m1, m2), guv)
            v230 = visibilitymap(mtr230, guv)[1:25]
            v345 = visibilitymap(mtr345, guv)[26:50]

            @test vmf[1:25] ≈ v230 atol = 1.0e-8
            @test vmf[26:50] ≈ v345 atol = 1.0e-8
        end

        @testset "Frequency+Time" begin
            m1 = modify(Gaussian(), Stretch(1.0))
            m2 = modify(Gaussian(), Stretch(MultiDomainParams(1.0, PolySpectral(1.0, 230.0e9))))

            mtr230 = modify(Gaussian(), Stretch(sqrt(2)))
            mtr345 = modify(Gaussian(), Stretch(sqrt(1 + (345 / 230)^2)))
            gXY = imagepixels(20.0, 20.0, 256, 256)
            g = RectiGrid((; X = gXY.X, Y = gXY.Y, Ti = 1.0:2.0, Fr = [230.0e9, 345.0e9]))
            @test intensitymap(convolved(m1, m2), g)[Ti = 1, Fr = 1] ≈ intensitymap(mtr230, gXY) atol = 1.0e-8
            @test intensitymap(convolved(m1, m2), g)[Ti = 1, Fr = 2] ≈ intensitymap(mtr345, gXY) atol = 1.0e-8
            @test intensitymap(convolved(m1, m2), g)[Ti = 2, Fr = 1] ≈ intensitymap(mtr230, gXY) atol = 1.0e-8
            @test intensitymap(convolved(m1, m2), g)[Ti = 2, Fr = 2] ≈ intensitymap(mtr345, gXY) atol = 1.0e-8

            u = randn(50) .* 0.25
            v = randn(50) .* 0.25
            ti = range(1.0, 3.0; length = 50)
            fr = vcat(fill(230.0e9, 10), fill(345.0e9, 40))
            guv = UnstructuredDomain((; U = u, V = v, Fr = fr, Ti = ti))
            vmf = visibilitymap(convolved(m1, m2), guv)
            v230 = visibilitymap(mtr230, guv[Fr = 230.0e9])
            v345 = visibilitymap(mtr345, guv[Fr = 345.0e9])

            @test vmf[1:10] ≈ v230 atol = 1.0e-8
            @test vmf[11:50] ≈ v345 atol = 1.0e-8
        end
    end

    @testset "PolySpectral Array" begin # evaluating a chain, and the bare transform
        base = rand(64, 64)
        indices = (ones(64, 64), zeros(64, 64))
        ps = MultiDomainParams(base, PolySpectral(indices, 230.0e9))
        @test ComradeBase.build_param(ps, (; Fr = 230.0e9)) ≈ base
        @test ComradeBase.build_param(ps, (; Fr = 230.0e9 * 2)) ≈ base .* 2.0
        @test ComradeBase.build_param(ps, (; Fr = 230.0e9 / 2)) ≈ base .* inv(2)

        # The stored base is never written through.
        base_orig = copy(base)
        ComradeBase.build_param(ps, (; Fr = 230.0e9 * 2))
        @test base ≈ base_orig

        # A model transforms whichever base it is paired with, and must not alias it.
        spec = PolySpectral(indices, 230.0e9)
        supplied = fill(42.0, 64, 64)
        out = MultiDomainParams(supplied, spec)((; Fr = 230.0e9 * 2))
        @test out ≈ supplied .* 2.0
        @test supplied == fill(42.0, 64, 64)
    end

    @testset "element type follows the parameters" begin
        # The default offset must not widen a narrower model.
        ps = PolySpectral((1.0f0,), 230.0f9)
        @test ComradeBase.paramtype(typeof(ps)) === Float32
        @test MultiDomainParams(1.0f0, ps)((; Fr = 345.0f9)) isa Float32
        @test MultiDomainParams(fill(1.0f0, 4, 4), ps)((; Fr = 345.0f9)) isa
            AbstractArray{Float32}

        # An explicit offset still promotes.
        @test ComradeBase.paramtype(typeof(PolySpectral((1.0f0,), 230.0f9, 1.0))) === Float64
    end

    @testset "constructor fail-fast" begin
        # Array spectral coefficients must be tuple-wrapped, and a base is paired with the
        # spectral model via MultiDomainParams rather than passed positionally.
        @test_throws MethodError PolySpectral(rand(4, 4), 230.0e9)
        @test_throws MethodError PolySpectral(rand(4, 4), 1.5, 230.0e9)
        @test_throws MethodError PolySpectral(rand(4, 4), 1.5, 230.0e9, 0.0)
    end

end


@testset "MultiDomainParams" begin
    ref = 230.0e9

    @testset "Polyspectral constructing MultiDomainParams" begin
        base = reshape(collect(1.0:6.0), 2, 3)
        base_orig = copy(base)

        α = reshape(collect(range(-1.0, 2.0; length = 6)), 2, 3)
        β = reshape(collect(range(-0.25, 0.25; length = 6)), 2, 3)
        p0 = reshape(collect(range(0.1, 0.6; length = 6)), 2, 3)

        ps = MultiDomainParams(base, PolySpectral((α, β), ref, p0))

        @test ps isa MultiDomainParams
        @test ps.base ≈ base
        @test length(ps.models) == 1

        model = first(ps.models)
        @test model isa PolySpectral
        @test model.index == (α, β)
        @test model.freq0 == ref

        p = (; Fr = 2 * ref)

        x = log(p.Fr / ref)
        arg = α .* x .+ β .* x^2
        expected_out = base .* exp.(arg) .+ p0

        out = ComradeBase.build_param(ps, p)

        @test out ≈ expected_out
        @test out !== base
        @test base ≈ base_orig

        # Returned output should not alias the stored base image.
        out[1, 1] = -999.0
        @test base[1, 1] == base_orig[1, 1]
    end

    @testset "PolySpectral build_param" begin
        base = reshape(collect(1.0:(32 * 32)), 32, 32)
        base_orig = copy(base)

        α = reshape(collect(range(-1.0, 2.0; length = 32 * 32)), 32, 32)
        β = reshape(collect(range(-0.25, 0.25; length = 32 * 32)), 32, 32)
        p0 = reshape(collect(range(0.1, 1.0; length = 32 * 32)), 32, 32)

        spec = PolySpectral((α, β), ref, p0)
        ps_param = MultiDomainParams(base, spec)
        ps_unit = MultiDomainParams(1.0, spec)

        p = (; Fr = 2 * ref)

        x = log(p.Fr / ref)
        arg = α .* x .+ β .* x^2

        expected_param = base .* exp.(arg) .+ p0
        expected_unit = exp.(arg) .+ p0

        @test ps_param(p) ≈ ComradeBase.build_param(ps_param, p)
        @test ps_param(p) ≈ expected_param
        @test ps_unit(p) ≈ expected_unit
        @test base ≈ base_orig

        # A model transforms whichever base it is paired with, leaving it intact.
        current = fill(42.0, size(base))
        current_orig = copy(current)

        out = MultiDomainParams(current, spec)(p)

        @test out ≈ current_orig .* exp.(arg) .+ p0
        @test current ≈ current_orig

        # Test scalar param path.
        ps_scalar = PolySpectral(1.0, ref, 1.0)
        scalar_expected = 1.0 * exp(1.0 * log(2.0)) + 1.0

        @test MultiDomainParams(1.0, ps_scalar)(p) ≈ scalar_expected
    end

    @testset "MultiDomainParams recursion" begin
        ref = 230.0e9
        p = (; Fr = 2ref)

        base = [1.0 2.0; 3.0 4.0]

        # At Fr = 2ref:
        # m1 applies x -> 2x + 1
        # m2 applies x -> 3x + 10
        m1 = PolySpectral((fill(1.0, size(base)),), ref, fill(1.0, size(base)))
        m2 = PolySpectral((fill(log2(3.0), size(base)),), ref, fill(10.0, size(base)))

        md = MultiDomainParams(base, m1, m2)

        expected_forward = 3 .* (2 .* base .+ 1) .+ 10
        expected_reverse = 2 .* (3 .* base .+ 10) .+ 1

        # The chain applies m1, then m2.
        out = ComradeBase.build_param(md, p)

        @test out ≈ expected_forward
        @test !(out ≈ expected_reverse)

        # The stored base is read, never written.
        base_orig = copy(base)
        ComradeBase.build_param(md, p)
        @test base ≈ base_orig
    end

    @testset "MultiDomainImage constructors" begin
        ref = 230.0e9

        @testset "MultiDomainImage builds a ContinuousImage" begin
            gXY = imagepixels(10.0, 10.0, 8, 8)
            base = rand(8, 8)
            dom = PolySpectral((1.0,), ref)
            md = MultiDomainImage(IntensityMap(base, gXY), BSplinePulse{3}(), dom)

            @test md isa ContinuousImage
            @test md isa MultiDomainImage
            @test md.params isa MultiDomainParams
            @test md.params.base ≈ base
            @test md.params.models == (dom,)
            @test md.grid == gXY
            @test md.kernel isa BSplinePulse

            # Chaining extends the model tuple rather than nesting, and stays evaluable:
            # chained order-1 models multiply their spectral factors.
            md2 = MultiDomainImage(md, dom)
            @test md2 isa MultiDomainImage
            @test md2.params.base ≈ base
            @test length(md2.params.models) == 2
            @test !(md2.params.base isa MultiDomainParams)
            gcube = RectiGrid((; X = gXY.X, Y = gXY.Y, Fr = [ref, 2 * ref]))
            img1 = intensitymap(md, gcube)
            img2 = intensitymap(md2, gcube)
            @test parent(img2)[:, :, 1] ≈ parent(img1)[:, :, 1]
            @test parent(img2)[:, :, 2] ≈ 2 .* parent(img1)[:, :, 2]
            @test flux(md2) ≈ flux(md)
            @test eltype(md2) == Float64

            # Chaining onto an existing chain is the same model as building it flat.
            flat = MultiDomainImage(IntensityMap(base, gXY), BSplinePulse{3}(), dom, dom)
            @test flat.params.models == md2.params.models
            @test intensitymap(flat, gcube) ≈ img2

            # A chain may not appear among the models of another chain.
            @test_throws ArgumentError MultiDomainParams(
                3.0, MultiDomainParams(3.0, PolySpectral(1.0, ref))
            )
            @test_throws "cannot be a model in another chain" MultiDomainParams(
                3.0, MultiDomainParams(3.0, PolySpectral(1.0, ref))
            )

            # A chain as the base flattens, so the composite factor is unchanged.
            mdp2 = MultiDomainParams(
                MultiDomainParams(3.0, PolySpectral(1.0, ref)),
                PolySpectral(0.5, ref)
            )
            @test ComradeBase.build_param(mdp2, (; Fr = 2 * ref)) ≈ 3.0 * 2.0 * 2.0^0.5
            # The element type and the base come from the single base of the chain.
            @test ComradeBase.paramtype(typeof(mdp2)) == Float64
            @test mdp2.base === 3.0
            @test md2.params.base ≈ base
        end

        @testset "MultiDomainImage intensity/visibility correctness" begin
            α = 1.5
            gXY = imagepixels(10.0, 10.0, 32, 32)
            base = rand(32, 32)
            dom = PolySpectral((α,), ref)
            cimg = MultiDomainImage(IntensityMap(base, gXY), BSplinePulse{3}(), dom)

            frs = [ref, 1.5 * ref]
            gcube = RectiGrid((; X = gXY.X, Y = gXY.Y, Fr = frs))

            # intensitymap: each frequency slice must equal the explicitly scaled image.
            img_md = intensitymap(cimg, gcube)
            for (i, fr) in enumerate(frs)
                slice_ref = intensitymap(
                    ContinuousImage(IntensityMap(base .* (fr / ref)^α, gXY), BSplinePulse{3}()),
                    gXY
                )
                @test parent(img_md[Fr = i]) ≈ parent(slice_ref) atol = 1.0e-10
            end

            # visibilitymap: must match a plain cube of the materialized images.
            cube = cat((base .* (fr / ref)^α for fr in frs)...; dims = 3)
            cimg_ref = ContinuousImage(IntensityMap(cube, gcube), BSplinePulse{3}())

            u = randn(40) .* 0.25
            v = randn(40) .* 0.25
            fr = vcat(fill(frs[1], 20), fill(frs[2], 20))
            guv = UnstructuredDomain((; U = u, V = v, Fr = fr))
            gfr = FourierDualDomain(gcube, guv, NFFTAlg())

            @test visibilitymap(cimg, gfr) ≈ visibilitymap(cimg_ref, gfr) atol = 1.0e-8
        end

        @testset "MultiDomainImage evaluated off its own grid" begin
            α = 1.5
            gXY = imagepixels(10.0, 10.0, 32, 32)
            base = rand(32, 32)
            cimg = MultiDomainImage(
                IntensityMap(base, gXY), BSplinePulse{3}(), PolySpectral((α,), ref)
            )
            frs = [ref, 1.5 * ref]

            # A grid the image does not live on resamples through the kernel, exactly as a
            # plain ContinuousImage does, rather than reinterpreting the pixels as if they
            # spanned the new field of view.
            for gout in (imagepixels(20.0, 20.0, 32, 32), imagepixels(10.0, 10.0, 48, 48))
                img = intensitymap(cimg, RectiGrid((; X = gout.X, Y = gout.Y, Fr = frs)))
                for (i, fr) in enumerate(frs)
                    slice_ref = intensitymap(
                        ContinuousImage(
                            IntensityMap(base .* (fr / ref)^α, gXY), BSplinePulse{3}()
                        ),
                        gout
                    )
                    @test parent(img[Fr = i]) ≈ parent(slice_ref) atol = 1.0e-10
                end
            end

            # Time and frequency together, in either order: every (Ti, Fr) point carries the
            # spectrally scaled image, resampled onto the grid it is evaluated over.
            tis = [0.0, 1.0]
            gout = imagepixels(20.0, 20.0, 32, 32)
            slice_ref = intensitymap(
                ContinuousImage(
                    IntensityMap(base .* (frs[2] / ref)^α, gXY), BSplinePulse{3}()
                ),
                gout
            )
            img_tf = intensitymap(
                cimg, RectiGrid((; X = gout.X, Y = gout.Y, Ti = tis, Fr = frs))
            )
            img_ft = intensitymap(
                cimg, RectiGrid((; X = gout.X, Y = gout.Y, Fr = frs, Ti = tis))
            )
            @test size(img_tf) == (32, 32, length(tis), length(frs))
            @test size(img_ft) == (32, 32, length(frs), length(tis))
            for i in eachindex(tis)
                @test parent(img_tf)[:, :, i, 2] ≈ parent(slice_ref) atol = 1.0e-10
                @test parent(img_ft)[:, :, 2, i] ≈ parent(slice_ref) atol = 1.0e-10
            end

            # A rotated grid resamples onto the rotated pixel centers.
            grot = imagepixels(10.0, 10.0, 32, 32; posang = π / 4)
            imgrot = intensitymap(
                cimg, RectiGrid((; X = grot.X, Y = grot.Y, Fr = frs); posang = π / 4)
            )
            plain = ContinuousImage(IntensityMap(base, gXY), BSplinePulse{3}())
            @test parent(imgrot[Fr = 1]) ≈ parent(intensitymap(plain, grot)) atol = 1.0e-10

            # The Fourier plans are tied to the grid they are built from, so there the
            # mismatch is an error rather than a resampling.
            guv = UnstructuredDomain(
                (; U = randn(20) ./ 4, V = randn(20) ./ 4, Fr = fill(ref, 20))
            )
            gbad = imagepixels(20.0, 20.0, 32, 32)
            gfour = FourierDualDomain(
                RectiGrid((; X = gbad.X, Y = gbad.Y, Fr = frs)), guv, NFFTAlg()
            )
            @test_throws DimensionMismatch visibilitymap(cimg, gfour)
        end
    end
end

# A family that varies across the image but does not define `restrict_params`: the fallback
# leaves its field at full size while the base is restricted to the pulse's window.
struct UnrestrictedField{A} <: ComradeBase.DomainParams{Float64}
    fac::A
end
ComradeBase.paramfield(m::UnrestrictedField, p) = m.fac
ComradeBase.apply_param(base, ::UnrestrictedField, fac, p) = base .* fac

@testset "restrict_params" begin
    ref = 230.0e9
    g = imagepixels(10.0, 10.0, 8, 8)
    base = rand(8, 8)
    coeff = rand(8, 8)
    p0 = rand(8, 8)
    md = MultiDomainParams(base, PolySpectral((coeff,), ref, p0))

    ix, iy = 2:4, 3:5
    sub = VLBISkyModels.restrict_params(md, ix, iy)

    # Fields are viewed, not copied, and a scalar parameter passes through.
    @test sub.base == view(base, ix, iy)
    @test sub.models[1].index[1] == view(coeff, ix, iy)
    @test sub.models[1].p0 == view(p0, ix, iy)
    @test sub.models[1].freq0 === md.models[1].freq0

    # Restricting then evaluating agrees with evaluating then restricting.
    p = (; Fr = 2 * ref)
    @test ComradeBase.build_param(sub, p) ≈ ComradeBase.build_param(md, p)[ix, iy]

    # A spatially varying family that does not define `restrict_params` fails loudly rather
    # than combining its full-sized field with a restricted base.
    bad = ContinuousImage(
        MultiDomainParams(base, UnrestrictedField(fill(2.0, 8, 8))), g, BSplinePulse{3}()
    )
    pt = (; X = g.X[4], Y = g.Y[4], Fr = ref)
    @test_throws DimensionMismatch ComradeBase.intensity_point(bad, pt)
end

@testset "intensity_point respects Fr for multidomain images" begin
    g = imagepixels(10.0, 10.0, 24, 24)
    base = rand(24, 24)
    cimg = MultiDomainImage(IntensityMap(base, g), BSplinePulse{3}(), PolySpectral((1.5,), 230.0e9))

    p230 = (; X = 0.1, Y = 0.05, Fr = 230.0e9)
    p345 = (; X = 0.1, Y = 0.05, Fr = 345.0e9)
    i230 = ComradeBase.intensity_point(cimg, p230)
    i345 = ComradeBase.intensity_point(cimg, p345)
    @test i230 != 0
    @test i345 ≈ i230 * (345 / 230)^1.5

    # At the reference frequency the multidomain image matches the plain image.
    ci_plain = ContinuousImage(IntensityMap(base, g), BSplinePulse{3}())
    @test i230 ≈ ComradeBase.intensity_point(ci_plain, p230)

    # Composite models sum through intensity_point, so the frequency dependence must
    # survive there too.
    gcube = RectiGrid((; X = g.X, Y = g.Y, Fr = [230.0e9, 345.0e9]))
    msum = intensitymap(cimg + Gaussian(), gcube)
    gimg = intensitymap(Gaussian(), g)
    pm = parent(msum)
    diff230 = pm[:, :, 1] .- parent(gimg)
    diff345 = pm[:, :, 2] .- parent(gimg)
    @test diff345 ≈ diff230 .* (345 / 230)^1.5
end

@testset "scalar chains fold like array chains" begin
    ref = 230.0e9
    p = (; Fr = 2 * ref)
    m1 = PolySpectral(1.0, ref)
    m2 = PolySpectral(0.5, ref, 1.0)

    md1 = MultiDomainParams(5.0, m1)
    @test ComradeBase.build_param(md1, p) ≈ 5.0 * 2.0

    md2 = MultiDomainParams(5.0, m1, m2)
    @test ComradeBase.build_param(md2, p) ≈ (5.0 * 2.0) * exp(0.5 * log(2.0)) + 1.0
    # each model transforms the result of the previous one
    @test ComradeBase.build_param(md2, p) ≈
        MultiDomainParams(ComradeBase.build_param(md1, p), m2)(p)
    @test @inferred(ComradeBase.build_param(md2, p)) isa Float64
end

@testset "FFTAlg rejects multidomain images" begin
    g = imagepixels(10.0, 10.0, 16, 16)
    cimg = MultiDomainImage(
        IntensityMap(rand(16, 16), g), BSplinePulse{3}(),
        PolySpectral((1.0,), 230.0e9)
    )
    gcube = RectiGrid((; X = g.X, Y = g.Y, Fr = [230.0e9, 345.0e9]))
    guv = UnstructuredDomain((; U = randn(8), V = randn(8), Fr = fill(230.0e9, 8)))
    gfour = FourierDualDomain(gcube, guv, FFTAlg())
    @test_throws "FFTAlg does not support multidomain" visibilitymap(cimg, gfour)
end

@testset "single visibility per Fr bin" begin
    g = imagepixels(10.0, 10.0, 16, 16)
    gcube = RectiGrid((; X = g.X, Y = g.Y, Fr = [230.0e9, 345.0e9]))
    U = [0.05, 0.1]
    V = [0.05, -0.1]
    Frs = [230.0e9, 345.0e9]
    guv = UnstructuredDomain((; U, V, Fr = Frs))
    cimg = MultiDomainImage(
        IntensityMap(rand(16, 16), g), BSplinePulse{3}(),
        PolySpectral((1.0,), 230.0e9)
    )
    gfour = FourierDualDomain(gcube, guv, NFFTAlg())
    vis = visibilitymap(cimg, gfour)
    @test length(vis) == 2
    for k in 1:2
        guv1 = UnstructuredDomain((; U = U[k:k], V = V[k:k], Fr = Frs[k:k]))
        gf1 = FourierDualDomain(gcube, guv1, NFFTAlg())
        vis1 = visibilitymap(cimg, gf1)
        @test vis[k] ≈ vis1[1]
    end
end

@testset "polarized multidomain images" begin
    ref = 230.0e9
    g = imagepixels(10.0, 10.0, 8, 8)
    I = rand(8, 8)
    Q = 0.1 .* rand(8, 8)
    U = 0.1 .* rand(8, 8)
    V = 0.05 .* rand(8, 8)
    pimg = StructArray{StokesParams{Float64}}((; I, Q, U, V))
    cimg = MultiDomainImage(IntensityMap(pimg, g), BSplinePulse{3}(), PolySpectral((1.0,), ref))
    @test ComradeBase.ispolarized(typeof(cimg)) == ComradeBase.IsPolarized()

    gcube = RectiGrid((; X = g.X, Y = g.Y, Fr = [ref, 2 * ref]))
    img = intensitymap(cimg, gcube)
    for s in (:I, :Q, :U, :V)
        ps = parent(stokes(img, s))
        @test ps[:, :, 2] ≈ 2 .* ps[:, :, 1]
    end

    # Stokes projection preserves the multidomain structure and matches the cube.
    cI = stokes(cimg, :I)
    @test cI.params isa MultiDomainParams
    @test parent(intensitymap(cI, gcube)) ≈ parent(stokes(img, :I))

    # `p0` offsets every Stokes component of a polarized base alike. At `Fr = ref` the
    # spectral factor is 1, so the offset is all that is left.
    off = MultiDomainParams(pimg, PolySpectral((1.0,), ref, 1.0))((; Fr = ref))
    for s in (:I, :Q, :U, :V)
        @test stokes(off, s) ≈ stokes(pimg, s) .+ 1
    end

    # A component gets its own spectrum from its own model, not from a polarized parameter.
    mk(a) = MultiDomainImage(IntensityMap(I, g), BSplinePulse{3}(), PolySpectral((a,), ref))
    pm = PolarizedModel(mk(1.0), mk(0.5), mk(-0.5), mk(0.0))
    pimgs = intensitymap(pm, gcube)
    for (s, a) in ((:I, 1.0), (:Q, 0.5), (:U, -0.5), (:V, 0.0))
        ps = parent(stokes(pimgs, s))
        @test ps[:, :, 2] ≈ 2.0^a .* ps[:, :, 1]
    end
end

@testset "an image needs an explicit base" begin
    ref = 230.0e9
    g = imagepixels(10.0, 10.0, 8, 8)

    # A lone spectral model carries no base image, so it cannot describe one.
    @test_throws ArgumentError ContinuousImage(
        PolySpectral((fill(1.0, 8, 8),), ref), g, BSplinePulse{3}()
    )
    @test_throws "has no base image" ContinuousImage(
        PolySpectral(1.0, ref), g, BSplinePulse{3}()
    )

    # A model transforms a base, so it has no value of its own anywhere — not just as an
    # image. Pairing it with a unit base gives the spectral factor.
    ps = PolySpectral(1.0, ref)
    @test_throws "has none of its own" ComradeBase.build_param(ps, (; Fr = 2 * ref))
    @test ComradeBase.build_param(MultiDomainParams(1.0, ps), (; Fr = 2 * ref)) ≈ 2.0

    # A chain whose base is not a spatial array is refused too.
    @test_throws "must be a chain whose base is the spatial image" ContinuousImage(
        MultiDomainParams(1.0, ps), g, BSplinePulse{3}()
    )
end

@testset "_paramcube broadcasts a partial model up to the cube" begin
    ref = 230.0e9
    g = imagepixels(10.0, 10.0, 8, 8)
    base = rand(8, 8)
    # A frequency-only model on a frequency+time grid produces a result smaller than the
    # full cube, so it must be broadcast up along `Ti`.
    c = MultiDomainImage(IntensityMap(base, g), BSplinePulse{3}(), PolySpectral(1.0, ref))
    g4 = RectiGrid((; X = g.X, Y = g.Y, Fr = [ref, 2 * ref], Ti = [0.0, 1.0, 2.0]))
    img = intensitymap(c, g4)
    @test size(img) == (8, 8, 2, 3)
    for k in 1:3
        @test parent(img)[:, :, 2, k] ≈ 2 .* parent(img)[:, :, 1, k]
        @test parent(img)[:, :, 1, k] ≈ parent(img)[:, :, 1, 1]
    end
end

@testset "ContinuousImage construction validation" begin
    g = imagepixels(10.0, 10.0, 8, 8)

    # Non-grid grid arguments and mismatched sizes fail at construction.
    @test_throws MethodError ContinuousImage(rand(4, 4), "not a grid", BSplinePulse{3}())
    @test_throws "does not match the grid size" ContinuousImage(
        rand(4, 4), g, BSplinePulse{3}()
    )

    # stokes on a convolved polarized image rebuilds with the convolved kernel.
    I = rand(8, 8)
    pimg = StructArray{StokesParams{Float64}}((; I, Q = 0.1I, U = 0.1I, V = 0.05I))
    pci = ContinuousImage(IntensityMap(pimg, g), BSplinePulse{3}())
    cc = convolved(pci, Gaussian())
    ccI = ComradeBase.stokes(cc, :I)
    @test ccI isa ContinuousImage
    @test ccI.kernel === cc.kernel

    # MultiDomainImage takes a 2D spatial base: a cube has no matching method.
    gcube = RectiGrid((; X = g.X, Y = g.Y, Fr = [230.0e9, 345.0e9]))
    @test_throws MethodError MultiDomainImage(
        IntensityMap(rand(8, 8, 2), gcube), BSplinePulse{3}(), PolySpectral((1.0,), 230.0e9)
    )
    @test_throws "2D spatial grid" MultiDomainImage(
        ContinuousImage(IntensityMap(rand(8, 8, 2), gcube), BSplinePulse{3}()),
        PolySpectral((1.0,), 230.0e9)
    )
end

@testset "spatialdims" begin
    x = range(-5.0, 5.0; length = 8)
    gok = RectiGrid((; X = x, Y = x, Fr = [1.0e9, 2.0e9]))
    @test map(DD.name, DD.dims(spatialdims(gok))) == (:X, :Y)
    @test spatialdims(IntensityMap(rand(8, 8, 2), gok)) == spatialdims(gok)

    g2 = imagepixels(10.0, 10.0, 8, 8)
    @test spatialdims(g2) == g2
end

@testset "ContinuousImage show" begin
    g = imagepixels(10.0, 10.0, 8, 8)
    ci = ContinuousImage(IntensityMap(rand(8, 8), g), BSplinePulse{3}())
    s = sprint(show, ci)
    @test occursin("ContinuousImage", s)
    @test occursin("BSplinePulse", s)
    @test occursin("(8, 8)", s)
    @test !occursin("RectiGrid", s)
    @test length(s) < 120

    md = MultiDomainImage(IntensityMap(rand(8, 8), g), BSplinePulse{3}(), PolySpectral((1.0,), 230.0e9))
    smd = sprint(show, md)
    @test occursin("MultiDomainParams", smd)
    @test occursin("BSplinePulse", smd)
    @test length(smd) < 120
end

@testset "PolySpectral and MultiDomainParams show" begin
    ref = 230.0e9
    @test sprint(show, PolySpectral((1.5, 0.5), ref)) == "PolySpectral((1.5, 0.5), 2.3e11)"
    # a zero offset is the default and is left off
    @test sprint(show, PolySpectral(1.5, ref)) == "PolySpectral((1.5,), 2.3e11)"
    @test occursin(", 1.0)", sprint(show, PolySpectral(1.5, ref, 1.0)))

    # array-valued coefficients and bases print as a summary, never inline
    sarr = sprint(show, PolySpectral((fill(1.0, 64, 64),), ref))
    @test occursin("64×64", sarr)
    @test length(sarr) < 120

    smd = sprint(show, MultiDomainParams(rand(64, 64), PolySpectral(1.5, ref)))
    @test startswith(smd, "MultiDomainParams(")
    @test occursin("64×64", smd)
    @test occursin("PolySpectral", smd)
    @test length(smd) < 120
end
