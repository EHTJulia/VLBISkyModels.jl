function domains4d(Nx, Nt, Nf)
    X = Y = range(-12, 12; length=Nx)
    Ti = sort(10 * rand(Nt))
    Fr = sort(1e11 * rand(Nf))

    imgdomain = RectiGrid((; X, Y, Ti, Fr))

    dx, dy = pixelsizes(imgdomain)
    U_vals = fftshift(fftfreq(500, 1 / dx))
    V_vals = fftshift(fftfreq(500, 1 / dy))

    # Repeat U and V to match Ti dimensions
    U_repeated = repeat(vec(U_vals); outer=(length(Ti)))
    V_repeated = repeat(vec(V_vals); outer=(length(Ti)))
    Ti_repeated = repeat(Ti; inner=(Int(length(vec(U_vals)))))

    # Repeat U and V and Ti to match Fr dimensions
    U_repeated = repeat(U_repeated; outer=(length(Fr)))
    V_repeated = repeat(V_repeated; outer=(length(Fr)))
    Ti_repeated = repeat(Ti_repeated; outer=(length(Fr)))
    Fr_repeated = repeat(Fr; inner=(Int(length(U_repeated) / length(Fr))))

    visdomain = UnstructuredDomain((; U=U_repeated, V=V_repeated, Ti=Ti_repeated,
                                    Fr=Fr_repeated))
    p = FourierDualDomain(imgdomain, visdomain, NFFTAlg())

    return p
end

function domains4d_swap(Nx, Nf, Nt)
    X = Y = range(-12, 12; length=Nx)
    Ti = sort(10 * rand(Nt))
    Fr = sort(1e11 * rand(Nf))

    imgdomain = RectiGrid((; X, Y, Fr, Ti))

    dx, dy = pixelsizes(imgdomain)
    U_vals = fftshift(fftfreq(500, 1 / dx))
    V_vals = fftshift(fftfreq(500, 1 / dy))

    # Repeat U and V to match Fr dimensions
    U_repeated = repeat(vec(U_vals); outer=(length(Fr)))
    V_repeated = repeat(vec(V_vals); outer=(length(Fr)))
    Fr_repeated = repeat(Fr; inner=(Int(length(vec(U_vals)))))

    # Repeat U and V and Fr to match Ti dimensions
    U_repeated = repeat(U_repeated; outer=(length(Ti)))
    V_repeated = repeat(V_repeated; outer=(length(Ti)))
    Fr_repeated = repeat(Fr_repeated; outer=(length(Ti)))
    Ti_repeated = repeat(Ti; inner=(Int(length(U_repeated) / length(Ti))))

    visdomain = UnstructuredDomain((; U=U_repeated, V=V_repeated, Fr=Fr_repeated,
                                    Ti=Ti_repeated))
    p = FourierDualDomain(imgdomain, visdomain, NFFTAlg())

    return p
end

function domains3d_time(Nx, Nt)
    X = Y = range(-12, 12; length=Nx)
    Ti = sort(10 * rand(Nt))

    imgdomain = RectiGrid((; X, Y, Ti))

    dx, dy = pixelsizes(imgdomain)
    U_vals = fftshift(fftfreq(500, 1 / dx))
    V_vals = fftshift(fftfreq(500, 1 / dy))

    # Repeat U and V to match Fr dimensions
    U_repeated = repeat(vec(U_vals); outer=(length(Ti)))
    V_repeated = repeat(vec(V_vals); outer=(length(Ti)))
    Ti_repeated = repeat(Ti; inner=(Int(length(vec(U_vals)))))

    visdomain = UnstructuredDomain((; U=U_repeated, V=V_repeated, Ti=Ti_repeated))
    p = FourierDualDomain(imgdomain, visdomain, NFFTAlg())

    return p
end

function domains3d_freq(Nx, Nf)
    X = Y = range(-12, 12; length=Nx)
    Fr = sort(1e11 * rand(Nf))

    imgdomain = RectiGrid((; X, Y, Fr))

    dx, dy = pixelsizes(imgdomain)
    U_vals = fftshift(fftfreq(500, 1 / dx))
    V_vals = fftshift(fftfreq(500, 1 / dy))

    # Repeat U and V to match Fr dimensions
    U_repeated = repeat(vec(U_vals); outer=(length(Fr)))
    V_repeated = repeat(vec(V_vals); outer=(length(Fr)))
    Fr_repeated = repeat(Fr; inner=(Int(length(vec(U_vals)))))

    visdomain = UnstructuredDomain((; U=U_repeated, V=V_repeated, Fr=Fr_repeated))
    p = FourierDualDomain(imgdomain, visdomain, NFFTAlg())

    return p
end

function domain2d(Nx)
    X = Y = range(-12, 12; length=Nx)

    imgdomain = RectiGrid((; X, Y))

    dx, dy = pixelsizes(imgdomain)
    U_vals = fftshift(fftfreq(500, 1 / dx))
    V_vals = fftshift(fftfreq(500, 1 / dy))

    visdomain = UnstructuredDomain((; U=vec(U_vals), V=vec(V_vals)))
    p = FourierDualDomain(imgdomain, visdomain, NFFTAlg())

    return p
end

# Function to calculate visibilities
function foo4D(x, p)
    cimg = ContinuousImage(IntensityMap(x, VLBISkyModels.imgdomain(p)), BSplinePulse{3}())
    vis = VLBISkyModels.visibilitymap_numeric(cimg, p)
    return sum(abs2, vis)
end

# Test function to check autodiff
function check4dautodiff(Nx, Nt, Nf, x, dx)
    p = domains4d(Nx, Nt, Nf)
    Enzyme.API.runtimeActivity!(true)
    df = Enzyme.autodiff(Reverse, foo4D, Active, Duplicated(x, dx), Const(p))
    return df, dx
end

# Function to test gradient against finite differences
function test4Dgrad(dx, x, p)
    finite_dx = grad(central_fdm(5, 1), x -> foo4D(x, p), x)[1]
    @test isapprox(dx, finite_dx, atol=1e-2)
end

# Check autodiff with Enzyme and compare grad
@testset "Enzyme autodiff test for 4D NFFT" begin
    # Example usage in test cases
    Nx, Nt, Nf = 24, 2, 2
    x = randn(Nx, Nx, Nt, Nf)
    dx = zeros(Nx, Nx, Nt, Nf)

    df, dx = check4dautodiff(Nx, Nt, Nf, x, dx)
    test4Dgrad(dx, x, domains4d(Nx, Nt, Nf))
end

# Time complexity tests
function testtimecomplexity(Nx, Nt1, Nt2, Nf1, Nf2)
    p1 = domains4d(Nx, Nt1, Nf1)
    cimg1 = ContinuousImage(IntensityMap(randn(Nx, Nx, 1, 1), VLBISkyModels.imgdomain(p1)),
                            BSplinePulse{3}())
    t1 = @benchmark VLBISkyModels.visibilitymap_numeric($cimg1, $p1)
    median_t1 = median(t1).time / 1e6

    p2 = domains4d(Nx, Nt2, Nf2)
    cimg2 = ContinuousImage(IntensityMap(randn(Nx, Nx, 2, 2), VLBISkyModels.imgdomain(p2)),
                            BSplinePulse{3}())
    t2 = @benchmark VLBISkyModels.visibilitymap_numeric($cimg2, $p2)
    median_t2 = median(t2).time / 1e6

    return median_t2 / median_t1
end

@testset "Check time complexity for time and freq FT" begin
    @test isapprox(testtimecomplexity(24, 1, 2, 1, 2), 4.0, atol=1e-1)
    @test isapprox(testtimecomplexity(24, 1, 2, 1, 1), 2.0, atol=1e-1)
    @test isapprox(testtimecomplexity(24, 1, 1, 1, 2), 2.0, atol=1e-1)
end

function rotating4dgaussian(p)
    # Elliptical gaussians rotating with a constant stretch and varying rotation
    gaussians = [modify(Gaussian(), Stretch(2, 1),
                        Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Ti) + π / 4),
                        Renormalize(1.0)) for (i, t) in enumerate(p.imgdomain.Ti)]
    intensity_maps = [intensitymap(mpr,
                                   RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y, Ti=[t],
                                              Fr=p.imgdomain.Fr)))
                      for (t, mpr) in zip(p.imgdomain.Ti, gaussians)]
    combined_img = cat(intensity_maps...; dims=3)  # Concatenate along the third dimension (Ti)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function rotating4dgaussian_shift(p)
    # Elliptical gaussians rotating with a constant stretch and varying rotation
    gaussians = [modify(Gaussian(), Stretch(2, 1), Shift(2.0, 1.0),
                        Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Ti) + π / 4),
                        Renormalize(1.0)) for (i, t) in enumerate(p.imgdomain.Ti)]
    intensity_maps = [intensitymap(mpr,
                                   RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y, Ti=[t],
                                              Fr=p.imgdomain.Fr)))
                      for (t, mpr) in zip(p.imgdomain.Ti, gaussians)]
    combined_img = cat(intensity_maps...; dims=3)  # Concatenate along the third dimension (Ti)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function test4dgaussiansft(Nx, Nt)
    p = domains4d(Nx, Nt, 1)
    cimg, gaussians = rotating4dgaussian(p)
    vis_numeric = VLBISkyModels.visibilitymap_numeric(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, t) in enumerate(p.imgdomain.Ti)
        imgdomain_analytic = RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y, Ti=[t],
                                        Fr=[p.imgdomain.Fr[1]]))
        visdomain_analytic = p.visdomain[Ti=t, Fr=p.imgdomain.Fr[1]]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, NFFTAlg())
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap_analytic(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol=1e-4)
end

function test4dgaussiansft_shift(Nx, Nt)
    p = domains4d(Nx, Nt, 1)
    cimg, gaussians = rotating4dgaussian_shift(p)
    vis_numeric = VLBISkyModels.visibilitymap_numeric(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, t) in enumerate(p.imgdomain.Ti)
        imgdomain_analytic = RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y, Ti=[t],
                                        Fr=[p.imgdomain.Fr[1]]))
        visdomain_analytic = p.visdomain[Ti=t, Fr=p.imgdomain.Fr[1]]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, NFFTAlg())
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap_analytic(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol=1e-4)
end

function rotating4dgaussian_swap(p)
    # Elliptical gaussians rotating with a constant stretch and varying rotation
    gaussians = [modify(Gaussian(), Stretch(2, 1),
                        Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Ti) + π / 4),
                        Renormalize(1.0)) for (i, t) in enumerate(p.imgdomain.Ti)]
    intensity_maps = [intensitymap(mpr,
                                   RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y,
                                              Fr=p.imgdomain.Fr, Ti=[t])))
                      for (t, mpr) in zip(p.imgdomain.Ti, gaussians)]
    combined_img = cat(intensity_maps...; dims=4)  # Concatenate along the fourth dimension (Ti)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function test4dgaussiansft_swap(Nx, Nt)
    p = domains4d_swap(Nx, 1, Nt)
    cimg, gaussians = rotating4dgaussian_swap(p)
    vis_numeric = VLBISkyModels.visibilitymap_numeric(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, t) in enumerate(p.imgdomain.Ti)
        imgdomain_analytic = RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y,
                                        Fr=[p.imgdomain.Fr[1]], Ti=[t]))
        visdomain_analytic = p.visdomain[Fr=p.imgdomain.Fr[1], Ti=t]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, NFFTAlg())
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap_analytic(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol=1e-4)
end

function rotating3dgaussian(p)
    # Elliptical gaussians rotating with a constant stretch and varying rotation
    gaussians = [modify(Gaussian(), Stretch(2, 1),
                        Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Ti) + π / 4),
                        Renormalize(1.0)) for (i, t) in enumerate(p.imgdomain.Ti)]
    intensity_maps = [intensitymap(mpr,
                                   RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y, Ti=[t])))
                      for (t, mpr) in zip(p.imgdomain.Ti, gaussians)]
    combined_img = cat(intensity_maps...; dims=3)  # Concatenate along the third dimension (Ti)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function test3dgaussians(Nx, Nt)
    p = domains3d_time(Nx, Nt)
    cimg, gaussians = rotating3dgaussian(p)
    vis_numeric = VLBISkyModels.visibilitymap_numeric(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, t) in enumerate(p.imgdomain.Ti)
        imgdomain_analytic = RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y, Ti=[t]))
        visdomain_analytic = p.visdomain[Ti=t]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, NFFTAlg())
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap_analytic(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol=1e-4)
end

function freqgaussians(p)
    gaussians = [modify(Gaussian(), Stretch(2, 1),
                        Rotate((i - 1) * 0.5 * π / length(p.imgdomain.Fr) + π / 4),
                        Renormalize(1.0)) for (i, fr) in enumerate(p.imgdomain.Fr)]
    intensity_maps = [intensitymap(mpr,
                                   RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y, Fr=[fr])))
                      for (fr, mpr) in zip(p.imgdomain.Fr, gaussians)]
    combined_img = cat(intensity_maps...; dims=3)  # Concatenate along the third dimension (Fr)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())
    return cimg, gaussians
end

function test3dgaussians_freq(Nx, Nf)
    p = domains3d_freq(Nx, Nf)
    cimg, gaussians = freqgaussians(p)

    vis_numeric = VLBISkyModels.visibilitymap_numeric(cimg, p)
    vis_analytic = similar(vis_numeric, 0)

    for (i, fr) in enumerate(p.imgdomain.Fr)
        imgdomain_analytic = RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y, Fr=[fr]))
        visdomain_analytic = p.visdomain[Fr=fr]
        p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, NFFTAlg())
        gaussian = gaussians[i]
        vis_analytic_t = VLBISkyModels.visibilitymap_analytic(gaussian, p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol=1e-4)
end

function test2dgaussian(Nx)
    p = domain2d(Nx)

    gaussian = modify(Gaussian(), Stretch(2, 1), Rotate(π / 4), Renormalize(1.0))
    intensity_map = intensitymap(gaussian, RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y)))
    cimg = ContinuousImage(intensity_map, BSplinePulse{3}())

    vis_numeric = VLBISkyModels.visibilitymap_numeric(cimg, p)

    imgdomain_analytic = RectiGrid((; X=p.imgdomain.X, Y=p.imgdomain.Y))
    visdomain_analytic = p.visdomain
    p_analytic = FourierDualDomain(imgdomain_analytic, visdomain_analytic, NFFTAlg())
    vis_analytic = VLBISkyModels.visibilitymap_analytic(gaussian, p_analytic)

    return isapprox(maximum(abs, vis_numeric - vis_analytic), 0; atol=1e-4)
end

# Todo 
# - Make the functions generic so that they work with any models

@testset "3D/4D ContinuousImage FT Correctness" begin
    @test test4dgaussiansft(1024, 10)
    @test test4dgaussiansft_shift(1024, 10)
    @test test4dgaussiansft_swap(1024, 10)
    @test test3dgaussians(1024, 10)
    @test test3dgaussians_freq(1024, 4)
    @test test2dgaussian(1024)
end
