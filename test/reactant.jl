using Reactant
using LinearAlgebra
using Random
using VLBISkyModels
using VLBISkyModels: NFFT
using Test

function _setup_nfft_pair(::Type{T}, D::Int, N::NTuple{Ndim, Int}, J::Int; seed::Int = 42) where {T, Ndim}
    D == Ndim || throw(ArgumentError("D=$D does not match length(N)=$Ndim"))
    rng = MersenneTwister(seed + 17D + J)
    k = rand(rng, T, D, J) .- T(0.5)
    kr = Reactant.to_rarray(k)

    g = imagepixels(1.0, 1.0, N...)
    guv = UnstructuredDomain((U = k[1, :], V = k[2, :]))
    p_ref = VLBISkyModels.plan_nuft_spatial(NFFTAlg(), g, guv)

    guvr = @jit UnstructuredDomain((U = kr[1, :], V = kr[2, :]))
    p_react = VLBISkyModels.plan_nuft_spatial(VLBISkyModels.ReactantNUFFTAlg(T), g, guvr)

    return rng, p_ref, p_react
end

@testset "Reactant" begin
    @testset "VisibilityMap Parity" begin
        gim = imagepixels(10.0, 10.0, 128, 128)
        gimr = @jit(identity(gim))

        rast = rand(128, 128)
        rastr = Reactant.to_rarray(rast)

        mr = ContinuousImage(rastr, gimr, DeltaPulse())
        m = ContinuousImage(rast, gim, DeltaPulse())

        u = randn(10^2) / 5.0
        v = randn(10^2) / 5.0
        guv = UnstructuredDomain((U = u, V = v))

        gfn = FourierDualDomain(gim, guv, NFFTAlg())
        gfr = FourierDualDomain(gimr, Reactant.to_rarray(guv), VLBISkyModels.ReactantNUFFTAlg(Float64; eps=1e-9))

        vnf = visibilitymap(m, gfn)
        vrf = @jit visibilitymap(mr, gfr)

        @test parent(vrf) ≈ vnf
    end
end
