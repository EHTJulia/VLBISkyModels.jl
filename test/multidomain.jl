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

# Run tests
@testset "Enzyme autodiff test for 4D NFFT" begin
    # Example usage in test cases
    Nx, Nt, Nf, fov = 24, 2, 2, 100
    x = randn(Nx, Nx, Nt, Nf)
    dx = zeros(Nx, Nx, Nt, Nf)

    df, dx = check4dautodiff(Nx, Nt, Nf, fov, x, dx)
    test4Dgrad(dx, x, nuftdomains4d(Nx, Nt, Nf, fov))
end

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