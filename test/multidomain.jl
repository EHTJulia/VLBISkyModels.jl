function nuftdomains4d(Nx, Nt, Nf, fov)
    X = Y = range(-μas2rad(fov), μas2rad(fov), length=Nx)
    Ti = 10 * rand(Nt) |> sort 
    Fr = 1e11 * rand(Nf) |> sort 

    imgdomain = RectiGrid((;X, Y, Ti, Fr))
    U_vals = range(-1/μas2rad(fov), 1/μas2rad(fov), length=length(X))
    U_vals = U_vals' .* ones(length(Y))
    V_vals = range(-1/μas2rad(fov), 1/μas2rad(fov), length=length(Y))
    V_vals = V_vals' .* ones(length(X))
    U_vals = U_vals'

    # Repeat U and V to match Ti dimensions
    U_repeated = repeat(vec(U_vals), outer=(length(Ti)))
    V_repeated = repeat(vec(V_vals), outer=(length(Ti)))
    Ti_repeated = repeat(Ti, inner=(Int(length(vec(U_vals)))))

    # Repeat U and V and Ti to match Fr dimensions
    U_repeated = repeat(U_repeated, outer=(length(Fr)))
    V_repeated = repeat(V_repeated, outer=(length(Fr)))
    Ti_repeated = repeat(Ti_repeated, outer=(length(Fr)))
    Fr_repeated = repeat(Fr, inner=(Int(length(U_repeated)/length(Fr))))

    visdomain = UnstructuredDomain((;U=U_repeated, V=V_repeated, Ti=Ti_repeated, Fr=Fr_repeated))
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
function check4dautodiff(Nx, Nt, Nf, fov, x, dx)
    p = nuftdomains4d(Nx, Nt, Nf, fov)
    Enzyme.API.runtimeActivity!(true)
    df = Enzyme.autodiff(Reverse, foo4D, Active, Duplicated(x, dx), Const(p))
    return df,dx
end

# Function to test gradient against finite differences
function test4Dgrad(dx, x, p)
    finite_dx = grad(central_fdm(5, 1), x -> foo4D(x, p), x)[1]
    @test isapprox(dx, finite_dx, atol=1e-2)
end

# Check autodiff with Enzyme and compare grad
@testset "Enzyme autodiff test for 4D NFFT" begin
    # Example usage in test cases
    Nx, Nt, Nf, fov = 24, 2, 2, 100
    x = randn(Nx, Nx, Nt, Nf)
    dx = zeros(Nx, Nx, Nt, Nf)

    df, dx = check4dautodiff(Nx, Nt, Nf, fov, x, dx)
    test4Dgrad(dx, x, nuftdomains4d(Nx, Nt, Nf, fov))
end


# Time complexity tests
function testtimecomplexity(Nx, Nt1, Nt2, Nf1, Nf2, fov)
    p1 = nuftdomains4d(Nx,Nt1,Nf1,fov)
    cimg1 = ContinuousImage(IntensityMap(randn(Nx, Nx, 1,1), VLBISkyModels.imgdomain(p1)), BSplinePulse{3}())
    t1 = @benchmark VLBISkyModels.visibilitymap_numeric($cimg1, $p1)
    median_t1 = median(t1).time / 1e6

    p2 = nuftdomains4d(Nx,Nt2,Nf2,fov)
    cimg2 = ContinuousImage(IntensityMap(randn(Nx, Nx, 2, 2), VLBISkyModels.imgdomain(p2)), BSplinePulse{3}())
    t2 = @benchmark VLBISkyModels.visibilitymap_numeric($cimg2, $p2)
    median_t2 = median(t2).time / 1e6

    return median_t2/median_t1
end

@testset "Check time complexity for time and freq FT" begin
    @test isapprox(testtimecomplexity(24,1,2,1,2,100), 4.0, atol=1e-1)
    @test isapprox(testtimecomplexity(24,1,2,1,1,100), 2.0, atol=1e-1)
    @test isapprox(testtimecomplexity(24,1,1,1,2,100), 2.0, atol=1e-1)
end

"""
function rotating4dgaussian(sizex, sizey, p)
    # Elliptical Gaussians Rotating
    fwhmfac = 2*sqrt(2*log(2))
    stretch_x = μas2rad(sizex)
    stretch_y = μas2rad(sizey)

    # Create the Gaussians with constant stretch and varying rotation
    gaussians = [modify(Gaussian(), Stretch(stretch_x./fwhmfac, stretch_y./fwhmfac), Rotate((i-1) * 0.5*π / length(p.imgdomain.Ti) + π/4), Renormalize(1.0)) for (i,t) in enumerate(p.imgdomain.Ti)]
    intensity_maps = [intensitymap(mpr, RectiGrid((;X=p.imgdomain.X, Y=p.imgdomain.Y, Ti=[t], Fr=p.imgdomain.Fr))) for (t, mpr) in zip(p.imgdomain.Ti, gaussians)]
    combined_img = cat(intensity_maps..., dims=3)  # Concatenate along the third dimension (Ti)
    cimg = ContinuousImage(combined_img, BSplinePulse{3}())

    return cimg, gaussians
end


function test4dgaussiansft(sizex, sizey, Nx, Nt, fov)
    p=nuftdomains4d(Nx, Nt, 1, fov)
    cimg, gaussians = rotating4dgaussian(sizex, sizey, p)
    vis_numeric=VLBISkyModels.visibilitymap_numeric(cimg,p)

    vis_analytic=similar(vis_numeric, 0)

    for (i,t) in enumerate(p.imgdomain.Ti)
        imgdomain_analytic=RectiGrid((;X=p.imgdomain.X, Y=p.imgdomain.Y,Ti=[t], Fr=[p.imgdomain.Fr[1]]))
        visdomain_analytic=p.visdomain[Ti=t, Fr=p.imgdomain.Fr[1]]
        p_analytic=FourierDualDomain(imgdomain_analytic, visdomain_analytic, NFFTAlg())
        gaussian=gaussians[i]
        vis_analytic_t=VLBISkyModels.visibilitymap_analytic(gaussian,p_analytic)
        append!(vis_analytic, vis_analytic_t)
    end

    return vis_numeric, vis_analytic, gaussians, p #isapprox(vis_numeric, vis_analytic, atol=1e-3)
end

vis_numeric, vis_analytic, gaussians, p = test4dgaussiansft(50, 20, 1024, 10, 500)

vis_analytic = reshape(vis_analytic, 1024, 1024, 10, 1)

amp=abs.(vis_analytic)
phase=rad2deg.(angle.(vis_analytic))
fig = CM.Figure(size=(1400, 420));
# Image
selected_img = intensitymap(gaussians[1], RectiGrid((;X=p.imgdomain.X, Y=p.imgdomain.Y)))#combined_img[:, :, 1, 1]
ax = CM.Axis(fig[1, 1], aspect=CM.DataAspect(), xreversed=true, title = "Time: 1, Frequency: 1"); #xreversed=true
CM.image!(ax, selected_img, colormap=:afmhot);
CM.hidedecorations!(ax)
# Amplitude
ax = CM.Axis(fig[1, 2], xreversed=true, title = "Amplitude (T=1, F=1)")
heatmapp = CM.heatmap!(ax, amp[:, :, 1, 1], colormap=:plasma, colorrange=(0, 1))
CM.Colorbar(fig[1, 3], heatmapp, width=20)
CM.hidedecorations!(ax)
# Phase
ax = CM.Axis(fig[1, 4], xreversed=true, title = "Phase (T=1, F=1)")
heatmapp = CM.heatmap!(ax, phase[:, :, 1, 1], colormap=:hsv, colorrange=(-180, 180))
CM.Colorbar(fig[1, 5], heatmapp, width=20)
CM.hidedecorations!(ax)
CM.save("analytic.png", fig)
"""

#isapprox(VLBISkyModels.visibilitymap_numeric(Gaussian(),p), VLBISkyModels.visibilitymap_analytic(Gaussian(),p), atol=1e-8)