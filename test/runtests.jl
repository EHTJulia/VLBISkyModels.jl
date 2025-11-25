using VLBISkyModels
using ChainRulesTestUtils
using ChainRulesCore
using FiniteDifferences
using FFTW
using JET
using Plots
using Statistics
using Test
using Serialization
using StructArrays
import DimensionalData as DD
import CairoMakie as CM
using ForwardDiff
using Enzyme
using LinearAlgebra
using Downloads
using BenchmarkTools
using EnzymeTestUtils
using FINUFFT
using NonuniformFFTs

function FiniteDifferences.to_vec(k::IntensityMap)
    v, b = to_vec(DD.data(k))
    back(x) = DD.rebuild(k, b(x))
    return v, back
end

function FiniteDifferences.to_vec(k::UnstructuredMap)
    v, b = to_vec(parent(k))
    back(x) = UnstructuredMap(b(x), axisdims(k))
    return v, back
end

function testgrad(f, x; atol = 1.0e-8, rtol = 1.0e-5)
    dx = Enzyme.make_zero(x)
    autodiff(set_runtime_activity(Enzyme.Reverse), Const(f), Active, Duplicated(x, dx))
    fdm = central_fdm(5, 1)
    gf = grad(fdm, f, x)[begin]
    return @test isapprox(dx, gf; atol, rtol)
end

function testmodel(
        m::VLBISkyModels.AbstractModel, npix = 256, atol = 1.0e-4, maxu = 1.0;
        radmul = 3.0
    )
    GC.gc()
    @info "Testing $(m)"
    # Plots.plot(m)
    g = imagepixels(
        radmul * VLBISkyModels.radialextent(m),
        radmul * VLBISkyModels.radialextent(m),
        npix, npix
    )
    gth = imagepixels(
        radmul * VLBISkyModels.radialextent(m),
        radmul * VLBISkyModels.radialextent(m),
        npix, npix; executor = ThreadsEx()
    )
    # CM.image(g.X, g.Y, m)
    img = intensitymap(m, g)
    imgt = intensitymap(m, gth)
    @test isapprox(maximum(img .- imgt), 0.0, atol = 1.0e-8)
    # Plots.plot(img)
    img2 = similar(img)
    intensitymap!(img2, m)
    @test eltype(img) === Float64
    @test isapprox(flux(m), flux(img), atol = atol)
    @test isapprox(maximum(parent(img) .- parent(img2)), 0, atol = 1.0e-8)
    dx, dy = pixelsizes(img)
    dx = max(VLBISkyModels.radialextent(m) / 128, dx)
    u1 = fftshift(fftfreq(size(img, 1), 1 / dx)) ./ 40
    u2 = range(-1 / (4 * dx), 1 / (4 * dx); length = size(img, 1) ÷ 2) * maxu
    if maximum(u2) > maximum(u1)
        u = u1
    else
        u = u2
    end
    uu = vec(u .* ones(length(u))')
    vv = vec(ones(length(u)) .* u')
    guv = UnstructuredDomain((U = uu, V = vv))
    gff = FourierDualDomain(g, guv, NFFTAlg())
    # Plots.closeall()
    mnon = ContinuousImage(img, DeltaPulse())
    van = visibilitymap(m, guv)
    vnu = visibilitymap(mnon, gff)
    @test isapprox(maximum(abs, van .- vnu), 0.0, atol = atol)
    img = nothing
    imgt = nothing
    u = nothing
    mnon = nothing
    van = nothing
    vnu = nothing
    gff = nothing
    mnon = nothing
    uu = nothing
    vv = nothing
    guv = nothing
    gff = nothing
    u1 = nothing
    u2 = nothing
    img2 = nothing

    return GC.gc()
end

function testft(m, npix = 256, atol = 1.0e-4)
    mn = VLBISkyModels.NonAnalyticTest(m)
    uu = push!(0.25 * randn(1000), 0.0)
    vv = push!(0.25 * randn(1000), 0.0)
    gim = imagepixels(
        2 * VLBISkyModels.radialextent(m), 2 * VLBISkyModels.radialextent(m),
        npix, npix
    )
    guv = UnstructuredDomain((U = uu, V = vv))
    img = intensitymap(m, gim)
    gnf = FourierDualDomain(gim, guv, NFFTAlg())
    gff = FourierDualDomain(gim, guv, FFTAlg())
    gdf = FourierDualDomain(gim, guv, DFTAlg())
    gfi = FourierDualDomain(gim, guv, FINUFFTAlg())
    gnu = FourierDualDomain(gim, guv, NonuniformFFTsAlg())

    va = visibilitymap(m, guv)

    vff = visibilitymap(mn, gff)
    vnf = visibilitymap(mn, gnf)
    vdf = visibilitymap(mn, gdf)
    vfi = visibilitymap(mn, gfi)
    vnu = visibilitymap(mn, gnu)

    @test isapprox(maximum(abs, va - vff), 0, atol = atol * 15)
    @test isapprox(maximum(abs, va - vnf), 0, atol = atol)
    @test isapprox(maximum(abs, va - vdf), 0, atol = atol)
    @test isapprox(maximum(abs, va - vfi), 0, atol = atol)
    @test isapprox(maximum(abs, va - vnu), 0, atol = atol)
    img = nothing
    gff = nothing
    gnf = nothing
    gdf = nothing
    gnu = nothing
    GC.gc()
    return nothing
end

function testft_nonan(mn, npix = 256, atol = 1.0e-4)
    uu = push!(0.25 * randn(25), 0.0)
    vv = push!(0.25 * randn(25), 0.0)
    gim = imagepixels(
        3 * VLBISkyModels.radialextent(mn),
        3 * VLBISkyModels.radialextent(mn),
        npix, npix
    )
    guv = UnstructuredDomain((U = uu, V = vv))
    img = intensitymap(mn, gim)
    gnf = FourierDualDomain(gim, guv, NFFTAlg())
    gff = FourierDualDomain(gim, guv, FFTAlg())
    gdf = FourierDualDomain(gim, guv, DFTAlg())
    gfi = FourierDualDomain(gim, guv, FINUFFTAlg())
    gnu = FourierDualDomain(gim, guv, NonuniformFFTsAlg())

    vff = visibilitymap(mn, gff)
    vnf = visibilitymap(mn, gnf)
    vdf = visibilitymap(mn, gdf)
    vfi = visibilitymap(mn, gfi)
    vnu = visibilitymap(mn, gnu)

    @test isapprox(maximum(abs, vnf - vff), 0, atol = atol * 15)
    @test isapprox(maximum(abs, vnf - vdf), 0, atol = atol)
    @test isapprox(maximum(abs, vnf - vfi), 0, atol = atol)
    @test isapprox(maximum(abs, vnf - vnu), 0, atol = atol)
    img = nothing
    gff = nothing
    gnf = nothing
    gdf = nothing
    gnu = nothing
    GC.gc()
    return nothing
end

function testft_cimg(m, atol = 1.0e-4)
    dx, dy = pixelsizes(m.img)
    u = fftshift(fftfreq(500, 1 / dx))
    v = fftshift(fftfreq(500, 1 / dy))
    gim = axisdims(parent(m))
    guv = UnstructuredDomain((U = u, V = v))
    gnf = FourierDualDomain(gim, guv, NFFTAlg())
    gff = FourierDualDomain(gim, guv, FFTAlg(; padfac = 20))
    gdf = FourierDualDomain(gim, guv, DFTAlg())
    gfi = FourierDualDomain(gim, guv, FINUFFTAlg())
    gnu = FourierDualDomain(gim, guv, NonuniformFFTsAlg())

    vff = visibilitymap(m, gff)
    vnf = visibilitymap(m, gnf)
    vdf = visibilitymap(m, gdf)
    vfi = visibilitymap(m, gfi)
    vnu = visibilitymap(m, gnu)

    @test isapprox(maximum(abs, vdf .- vnf), 0, atol = atol)
    @test isapprox(maximum(abs, vff .- vdf), 0, atol = atol * 10)
    @test isapprox(maximum(abs, vfi .- vnf), 0, atol = atol * 10)
    @test isapprox(maximum(abs, vnu .- vnf), 0, atol = atol * 10)

    gff = nothing
    gnf = nothing
    gdf = nothing
    gnu = nothing
    GC.gc()
    return nothing
end

function test_opt(m::M) where {M}
    if ComradeBase.imanalytic(M) == ComradeBase.IsAnalytic()
        @test_opt ComradeBase.intensity_point(m, (X = 0.0, Y = 0.0, Fr = 230.0e9, Ti = 0.0))
    end

    return if ComradeBase.visanalytic(M) == ComradeBase.IsAnalytic()
        @test_opt ComradeBase.visibility_point(m, (U = 0.0, V = 0.0, Fr = 230.0e9, Ti = 0.0))
    end
end

@testset "VLBISkyModels.jl" begin
    include("models.jl")
    include("continuous_image.jl")
    include("templates.jl")
    include("polarized.jl")
    include("multidomain.jl")
    include("utility.jl")
    include("viz.jl")
    include("io.jl")
    include("stokesintensitymap.jl")
    include("rules.jl")
    include("rotgrid.jl")
end
