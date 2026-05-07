using Reactant
using LinearAlgebra
using Random
using VLBISkyModels
using VLBISkyModels: NFFT

function _setup_nfft_pair(::Type{T}, D::Int, N::NTuple{Ndim, Int}, J::Int; seed::Int = 42) where {T, Ndim}
    D == Ndim || throw(ArgumentError("D=$D does not match length(N)=$Ndim"))
    rng = MersenneTwister(seed + 17D + J)
    k = rand(rng, T, D, J) .- T(0.5)

    p_ref = NFFT.plan_nfft(
        NFFT.NFFTBackend(),
        copy(k),
        N;
        reltol = 1.0e-7,
        precompute = NFFT.TENSOR,
        blocking = true,
        storeDeconvolutionIdx = true,
    )

    p_react = NFFT.plan_nfft(
        NFFT.NFFTBackend(),
        Reactant.RArray,
        k,
        N;
        reltol = 1.0e-7,
    )

    return rng, p_ref, p_react
end

@testset "Reactant" begin
    @testset "VisibilityMap Parity" begin
        gim = imagepixels(10.0, 10.0, 128, 128)
        gimr = @jit(identity(gim))

        rast = rand(128, 128)
        rastr = Reactant.to_rarray(rast)

        mr = ContinuousImage(rastr, gimr, BSplinePulse{3}())
        m = ContinuousImage(rast, gim, BSplinePulse{3}())

        u = randn(10^6) / 5.0
        v = randn(10^6) / 5.0
        guv = UnstructuredDomain((U = u, V = v))

        gfn = FourierDualDomain(gim, guv, NFFTAlg())
        gfr = FourierDualDomain(gimr, guv, VLBISkyModels.ReactantAlg())

        vnf = visibilitymap(m, gfn)
        vrf = @jit visibilitymap(mr, gfr)

        @test parent(vrf) ≈ vnf
    end

    @testset "Direct NFFT Forward/Adjoint Parity" begin
        configs = (
            (D = 1, N = (96,), J = 1800),
            (D = 2, N = (72, 56), J = 4000),
            (D = 3, N = (28, 24, 20), J = 2400),
        )

        for cfg in configs
            D, N, J = cfg.D, cfg.N, cfg.J
            rng, p_ref, p_react = _setup_nfft_pair(Float64, D, N, J; seed = 2026)

            x = rand(rng, Float64, N...)
            xr = Reactant.to_rarray(x)
            y = randn(rng, ComplexF64, J)
            yr = Reactant.to_rarray(y)

            forward_react(inp) = p_react * inp
            adjoint_react(inp) = p_react' * inp

            y_ref = p_ref * complex.(x)
            y_react = @jit forward_react(xr)
            @test parent(y_react) ≈ y_ref atol = 5.0e-7 rtol = 5.0e-6

            x_ref = p_ref' * y
            x_react = @jit adjoint_react(yr)
            @test parent(x_react) ≈ x_ref atol = 5.0e-7 rtol = 5.0e-6
        end
    end

    @testset "Single-Shot Harness" begin
        D = 2
        N = (128, 128)
        J = 25_000

        rng, p_ref, p_react = _setup_nfft_pair(Float64, D, N, J; seed = 77)
        x = rand(rng, Float64, N...)
        xr = Reactant.to_rarray(x)

        forward_react(inp) = p_react * inp

        # Warmup compile
        fr = @compile sync = true forward_react(xr)
        fr(xr)
        t_react = @elapsed begin
            fr(xr)
        end

        t_ref = @elapsed begin
            _ = p_ref * complex.(x)
        end

        @info "Reactant NFFT harness" t_ref t_react
    end
end
