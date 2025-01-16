function test_methods(m::Multifrequency{B}) where {B}
    @test visanalytic(m) == NotAnalytic()
    @test imanalysic(m) == imanalytic(B)
    @test radialextent(m) == radialextent(B)
    @test flux(m) = flux(B)
    GC.gc()
    return nothing
end

#function test_ContinuousImageTaylorSpectral()
#    return TaylorSpectral(index::NTuple{N}, freq0::Real)
#end

function gaussmodel(θ)
    (; f, σG) = θ
    g = f * stretched(Gaussian(), σG, σG)
    return g
end

# unit test consisting of a constant spectral index gaussian

@testset "Multifrequency Gaussian: Taylor Expansion" begin
    ν0 = 8e9 # reference frequency (Hz)
    ν = 12e9 # target frequency (Hz)
    νlist = [ν0, ν]

    # testing mfimagepixels & generating a multifrequency image grid
    g = imagepixels(μas2rad(1000.0), μas2rad(1000.0), 256, 256)
    mfgrid = mfimagepixels(μas2rad(1000.0), μas2rad(1000.0), 256, 256, νlist) # build multifrequency image grid

    @test dims(g)[1].val ≈ dims(mfgrid)[1].val
    @test dims(g)[2].val ≈ dims(mfgrid)[2].val
    @test dims(mfgrid)[3].val == νlist

    # 8 GHz Gaussian with total flux = 1.2 Jy
    θ1 = (f=1.2, σG=μas2rad(100))
    gauss1 = intensitymap(gaussmodel(θ1), g)
    gaussmodel1 = ContinuousImage(gauss1, BSplinePulse{3}())

    # 12 GHz Gaussian with total flux = 1.6 Jy
    θ2 = (f=1.6, σG=μas2rad(100))
    gauss2truth = intensitymap(gaussmodel(θ2), g)
    gaussmodel2truth = ContinuousImage(gauss2truth, BSplinePulse{3}())

    # testing spectral index map: constant spectral index and 0 spectral curvature
    α0 = log(1.6 / 1.2) / log(12 / 8)
    α = fill(α0, size(gauss1)) # spectral index map
    β0 = 0.0
    β = fill(β0, size(gauss1)) # spectral curvature map

    spec1 = TaylorSpectral((α, β), ν0)
    spec2 = TaylorSpectral((α0, β0), ν0)

    @test VLBISkyModels.order(spec1) == 2
    @test VLBISkyModels.order(spec2) == 2

    # create a multifrequency object
    mfgauss1 = Multifrequency(gaussmodel1, ν0, spec1)
    mfgauss2 = Multifrequency(gaussmodel1, ν0, spec2)

    # test intensity_point
    p = (; X=0, Y=0, Fr=ν)
    @test VLBISkyModels.intensity_point(mfgauss1, p) ==
          VLBISkyModels.intensity_point(gaussmodel2truth, p)
    @test VLBISkyModels.intensity_point(mfgauss2, p) ==
          VLBISkyModels.intensity_point(gaussmodel2truth, p)

    # test generatemodel
    # generate model at new frequency & compare with ground truth
    @test parent(generatemodel(mfgauss1, ν)) ≈ gauss2truth
    @test parent(generatemodel(mfgauss2, ν)) ≈ gauss2truth

    # test visibilitymap
    # generate 100 visibilities: half at 8 GHz and half at 12 GHz
    u = range(1, 10, 50) * 1e7 # random visibilities
    v = range(1, 10, 50) * 1e7
    f = similar(u) # each uv point has a frequency associated with it
    f[1:25] .= νlist[1]
    f[26:50] .= νlist[2]

    fdd8 = FourierDualDomain(axisdims(gauss1), UnstructuredDomain((U=u[1:25], V=v[1:25])),
                             NFFTAlg())
    fdd12 = FourierDualDomain(axisdims(gauss2truth),
                              UnstructuredDomain((U=u[26:50], V=v[26:50])), NFFTAlg())
    mffdd = FourierDualDomain(RectiGrid((X=g.X, Y=g.Y, Fr=νlist)),
                              UnstructuredDomain((U=u, V=v, Fr=f)), NFFTAlg())

    # comparing multifrequency to single frequency results
    @test visibilitymap(mfgauss1, mffdd)[1:25] ≈ visibilitymap(gaussmodel1, fdd8)
    @test visibilitymap(mfgauss1, mffdd)[26:50] ≈ visibilitymap(gaussmodel2truth, fdd12)
end
