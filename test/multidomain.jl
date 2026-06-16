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
        ts = PolySpectral(1.0, 1.0, 230.0, -1.0)
        @test ts((; Fr = 230.0)) ≈ 0.0
        @test ts((; Fr = 345.0)) ≈ 0.5

        ts2 = PolySpectral(1.0, (0.0, 1.0), 230.0)
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
            ts = PolySpectral(1.0, 1.0, 230.0e9)
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
            tev = PolySpectral(RM, 2.0, 230.0e9, -RM) # zeropoint the RM at 230 GHz
            m1 = modify(mb, Rotate(tev))
            mn = modify(mbn, Rotate(tev))
            test_modifier(m1, mb, modify(mb, Rotate(RM * (345 / 230)^2 - RM)), gfr)
            test_modifier(mn, mbn, modify(mbn, Rotate(RM * (345 / 230)^2 - RM)), gfr)
        end

        @testset "Shift" begin
            mb = Gaussian()
            mbn = ExtendedRing(8.0)
            ts = PolySpectral(1.0, 1.0, 230.0e9, -1.0)
            m1 = modify(mb, Shift(ts, 0.0))
            mn = modify(mbn, Shift(ts, 0.0))
            test_modifier(m1, mb, modify(mb, Shift(0.5, 0.0)), gfr)
            test_modifier(mn, mbn, modify(mbn, Shift(0.5, 0.0)), gfr)
        end

        @testset "Renormalize" begin
            mb = Gaussian()
            mbn = ExtendedRing(8.0)
            ts = PolySpectral(1.0, 1.0, 230.0e9)
            m1 = ts * mb
            mn = ts * mbn
            test_modifier(m1, mb, 1.5 * mb, gfr)
            test_modifier(mn, mbn, 1.5 * mbn, gfr)
        end

        @testset "Multi modifiers" begin
            mb = Gaussian()
            mbn = ExtendedRing(8.0)
            tss = PolySpectral(1.0, 1.0, 345.0e9)
            tsx = PolySpectral(1.0, 1.0, 230.0e9, -1.0)
            tsr = PolySpectral(1.0, 1.0, 345.0e9, -1.0)

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

        ts = PolySpectral(1.0, 1.0, 230.0e9)
        m1 = modify(Gaussian(), Stretch(ts))
        m2 = ExtendedRing(8.0)
        ts3 = PolySpectral(8.0, 1.0, 230.0e9)
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
            m2 = modify(Gaussian(), Stretch(PolySpectral(1.0, 1.0, 230.0e9)))

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
            m2 = modify(Gaussian(), Stretch(PolySpectral(1.0, 1.0, 230.0e9)))

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

    @testset "PolySpectral Array" begin # test implementation of build_param and build_param!
        g = imagepixels(10.0, 10.0, 64, 64)
        base = rand(64, 64)
        indices = (ones(64, 64), zeros(64, 64))
        ps = PolySpectral(base, indices, 230.0e9)
        @test ComradeBase.build_param(ps, (; Fr = 230.0e9)) ≈ base
        @test ComradeBase.build_param(ps, (; Fr = 230.0e9 * 2)) ≈ base .* 2.0
        @test ComradeBase.build_param(ps, (; Fr = 230.0e9 / 2)) ≈ base .* inv(2)

        bimg = IntensityMap(base,g)
        bimg_orig = copy(bimg)
        @test VLBISkyModels.build_param!(copy(bimg_orig), ps, (; Fr = 230.0e9)) ≈ bimg_orig
        @test VLBISkyModels.build_param!(copy(bimg_orig), ps, (; Fr = 230.0e9 * 2)) ≈ bimg_orig .* 2.0
        @test VLBISkyModels.build_param!(copy(bimg_orig), ps, (; Fr = 230.0e9 / 2)) ≈ bimg_orig .* inv(2)
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

        ps = PolySpectral(base, (α, β), ref, p0)

        @test ps isa MultiDomainParams
        @test ps.params ≈ base
        @test length(ps.models) == 1
        @test isnothing(first(ps.models).params)

        model = first(ps.models)
        @test model isa PolySpectral
        @test isnothing(model.params)
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
        base = reshape(collect(1.0:32 * 32), 32, 32)
        base_orig = copy(base)

        α = reshape(collect(range(-1.0, 2.0; length = 32 * 32)), 32, 32)
        β = reshape(collect(range(-0.25, 0.25; length = 32 * 32)), 32, 32)
        p0 = reshape(collect(range(0.1, 1.0; length = 32 * 32)), 32, 32)

        # Test param-provided PolySpectral path.
        ps_param = PolySpectral(base, (α, β), ref, p0)
        ps_noparam = PolySpectral((α, β), ref, p0)

        p = (; Fr = 2 * ref)

        x = log(p.Fr / ref)
        arg = α .* x .+ β .* x^2

        expected_param = base .* exp.(arg) .+ p0
        expected_noparam = exp.(arg) .+ p0

        # Test 2 -> 3 argument build_param conversion.
        @test ps_param(p) ≈ ComradeBase.build_param(ps_param, p)
        @test ps_noparam(p) ≈ ComradeBase.build_param(ps_noparam, p)

        @test ps_param(p) ≈ expected_param
        @test ps_noparam(p) ≈ expected_noparam

        # Direct 3-argument non-mutating paths.
        @test ComradeBase.build_param(nothing, ps_noparam, p) ≈ expected_noparam
        @test ComradeBase.build_param(base, ps_noparam, p) ≈ expected_param
        @test base ≈ base_orig

        # Direct 3-argument mutating path.
        out = copy(base)
        ret = VLBISkyModels.build_param!(out, ps_noparam, p)

        @test ret === out
        @test out ≈ expected_param
        @test base ≈ base_orig

        # Mutating path should transform the current first argument.
        current = fill(42.0, size(base))
        current_orig = copy(current)

        ret_current = VLBISkyModels.build_param!(current, ps_noparam, p)

        @test ret_current === current
        @test current ≈ current_orig .* exp.(arg) .+ p0

        # Test scalar param path.
        ps_scalar = PolySpectral(1.0, 1.0, ref, 1.0)
        scalar_expected = 1.0 * exp(1.0 * log(2.0)) + 1.0

        @test ComradeBase.build_param(ps_scalar, p) ≈ scalar_expected
        @test ComradeBase.build_param(1.0, ps_scalar, p) ≈ scalar_expected
        @test VLBISkyModels.build_param!(1.0, ps_scalar, p) ≈ scalar_expected
        @test ps_scalar(p) ≈ scalar_expected
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

        # Non-mutating recursion should apply m1, then m2.
        out = ComradeBase.build_param(md, p)

        @test out ≈ expected_forward
        @test !(out ≈ expected_reverse)

        # Mutating recursion should transform the current value passed in.
        buf = copy(base)
        ret = VLBISkyModels.build_param!(buf, md, p)

        @test ret === buf
        @test buf ≈ expected_forward
        @test !(buf ≈ expected_reverse)

        # Prove build_param! is mutating the first argument.
        buf2 = fill(42.0, size(base))
        VLBISkyModels.build_param!(buf2, md, p)

        expected_from_current = 3 .* (2 .* fill(42.0, size(base)) .+ 1) .+ 10
        @test buf2 ≈ expected_from_current
    end

    @testset "MultiDomainImage and imagepixels constructors" begin
        ref = 230.0e9

        @testset "MultiDomainImage wraps model and domains" begin
            dom = PolySpectral((1.0,), ref)
            md = MultiDomainImage(Gaussian(), dom)

            @test md isa MultiDomainParams
            @test md.params isa Gaussian
            @test md.models == (dom,)
        end

        @testset "imagepixels with one extra dimension" begin
            fr = Fr([230.0e9, 345.0e9])
            g = imagepixels(10.0, 20.0, 4, 5, fr)

            @test length(g.X) == 4
            @test length(g.Y) == 5
            @test length(g.Fr) == 2
            @test collect(g.Fr) == [230.0e9, 345.0e9]
        end

        @testset "imagepixels with two extra dimensions preserves order" begin
            fr = Fr([230.0e9, 345.0e9])
            ti = Ti([1.0, 2.0, 3.0])

            g_fr_ti = imagepixels(10.0, 20.0, 4, 5, fr, ti)
            g_ti_fr = imagepixels(10.0, 20.0, 4, 5, ti, fr)

            @test length(g_fr_ti.Fr) == 2
            @test length(g_fr_ti.Ti) == 3
            @test length(g_ti_fr.Ti) == 3
            @test length(g_ti_fr.Fr) == 2

            @test dims(g_fr_ti)[3] != dims(g_ti_fr)[3]
            @test dims(g_fr_ti)[4] != dims(g_ti_fr)[4]
        end

        @testset "imagepixels rejects nonpositive image sizes" begin
            fr = Fr([230.0e9, 345.0e9])

            @test_throws AssertionError imagepixels(10.0, 20.0, 0, 5, fr)
            @test_throws AssertionError imagepixels(10.0, 20.0, 4, 0, fr)
        end
    end
end