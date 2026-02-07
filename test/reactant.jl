using Reactant
using NFFT

@testset "Reactant" begin
    
    gim = imagepixels(10.0, 10.0, 128, 128)
    gimr = @jit(identity(gim))

    rast = rand(128, 128)
    rastr = Reactant.to_rarray(rast)

    mr = ContinuousImage(rastr, gimr, BSplinePulse{3}())
    m = ContinuousImage(rast, gim, BSplinePulse{3}())

    u = randn(64)/5.0
    v = randn(64)/5.0
    guv = UnstructuredDomain((U = u, V = v))

    gfn = FourierDualDomain(gim, guv, NFFTAlg())
    gfr = FourierDualDomain(gimr, guv, VLBISkyModels.ReactantAlg())

    vnf = visibilitymap(m, gfn)
    vrf = @jit visibilitymap(mr, gfr)

    @test parent(vrf) ≈ vnf
end