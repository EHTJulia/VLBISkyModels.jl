export FFTAlg




# These are all overloads to allow us to forward propogate ForwardDiff dual numbers through
# an abstract FFT.
AbstractFFTs.complexfloat(x::AbstractArray{<:ForwardDiff.Dual}) = float.(ForwardDiff.value.(x) .+ 0im)

# Make a plan with Dual numbers
AbstractFFTs.plan_fft(x::AbstractArray{<:ForwardDiff.Dual}, region=1:ndims(x)) = plan_fft(zeros(ComplexF64, size(x)), region)
AbstractFFTs.plan_fft(x::AbstractArray{<:Complex{<:ForwardDiff.Dual}}, region=1:ndims(x)) = plan_fft(zeros(ComplexF64, size(x)), region)

# Allow things to work with Complex Dual numbers.
ForwardDiff.value(x::Complex{<:ForwardDiff.Dual}) = Complex(x.re.value, x.im.value)
ForwardDiff.partials(x::Complex{<:ForwardDiff.Dual}, n::Int) = Complex(ForwardDiff.partials(x.re, n), ForwardDiff.partials(x.img, n))
ForwardDiff.npartials(x::Complex{<:ForwardDiff.Dual}) = ForwardDiff.npartials(x.re)

# internal function that creates the interpolator objector to evaluate the FT.
function create_interpolator(U, V, vis::AbstractArray{<:Complex}, pulse)
    # Construct the interpolator
    #itp = interpolate(vis, BSpline(Cubic(Line(OnGrid()))))
    #etp = extrapolate(itp, zero(eltype(vis)))
    #scale(etp, u, v)
    p1 = BicubicInterpolator(U, V, real(vis), NoBoundaries())
    p2 = BicubicInterpolator(U, V, imag(vis), NoBoundaries())
    function (u,v)
        pl = visibility_point(pulse, u, v, zero(u), zero(u))
        return pl*(p1(u,v) + 1im*p2(u,v))
    end
end

function create_interpolator(U, V, vis::StructArray{<:StokesParams}, pulse)
    # Construct the interpolator
    pI_real = BicubicInterpolator(U, V, real(vis.I), NoBoundaries())
    pI_imag = BicubicInterpolator(U, V, real(vis.I), NoBoundaries())

    pQ_real = BicubicInterpolator(U, V, real(vis.Q), NoBoundaries())
    pQ_imag = BicubicInterpolator(U, V, real(vis.Q), NoBoundaries())

    pU_real = BicubicInterpolator(U, V, real(vis.U), NoBoundaries())
    pU_imag = BicubicInterpolator(U, V, real(vis.U), NoBoundaries())

    pV_real = BicubicInterpolator(U, V, real(vis.V), NoBoundaries())
    pV_imag = BicubicInterpolator(U, V, real(vis.V), NoBoundaries())


    function (u,v)
        pl = visibility_point(pulse, u, v, zero(u), zero(u))
        return StokesParams(
            pI_real(u,v)*pl + 1im*pI_imag(u,v)*pl,
            pQ_real(u,v)*pl + 1im*pQ_imag(u,v)*pl,
            pU_real(u,v)*pl + 1im*pU_imag(u,v)*pl,
            pV_real(u,v)*pl + 1im*pV_imag(u,v)*pl,
        )
    end
end

struct InterpolatedModel{M, S} <: AbstractModel
    model::M
    sitp::S
end

function Base.show(io::IO, m::InterpolatedModel)
    print(io, "InterpolatedModel(", m.model, ")")
end

function InterpolatedModel(model, cache::FFTCache)
    img = intensitymap(model, cache.grid)
    pimg = padimage(img, cache.alg)
    vis = applyfft(cache.plan, pimg)
    (;X, Y) = cache.grid
    (;U, V) = cache.gridUV
    vispc = phasecenter(vis, X, Y, U, V)
    pulse = cache.pulse
    sitp = create_interpolator(U, V, vispc, stretched(pulse, step(X), step(Y)))
    return InterpolatedModel{typeof(model), typeof(sitp)}(model, sitp)
end


@inline visanalytic(::Type{<:InterpolatedModel}) = IsAnalytic()
@inline imanalytic(::Type{<:InterpolatedModel})  = IsAnalytic()
@inline ispolarized(::Type{<:InterpolatedModel{M}}) where {M} = ispolarized(M)

intensity_point(m::InterpolatedModel, p) = intensity_point(m.model, p)
visibility_point(m::InterpolatedModel, u, v, t, f) = m.sitp(u, v)

#=
This is so ForwardDiff works with FFTW. I am very harsh on the `x` type because of type piracy.
=#
function Base.:*(p::AbstractFFTs.Plan, x::PaddedView{<:ForwardDiff.Dual{T,V,P},N, I,<:IntensityMapTypes}) where {T,V,P,N,I}
    M = typeof(ForwardDiff.value(first(x)))
    cache = Matrix{M}(undef, size(x)...)
    cache .= ForwardDiff.value.(x)
    xtil = p * cache
    ndxs = ForwardDiff.npartials(first(x))
    dxtils = ntuple(ndxs) do n
        cache .= ForwardDiff.partials.(x, n)
        p * cache
    end
    out = similar(cache, Complex{ForwardDiff.Dual{T,V,P}})
    for i in eachindex(out)
        dual = getindex.(dxtils, i)
        prim = xtil[i]
        red = ForwardDiff.Dual{T,V,P}(real(prim), ForwardDiff.Partials(real.(dual)))
        imd = ForwardDiff.Dual{T,V,P}(imag(prim), ForwardDiff.Partials(imag.(dual)))
        out[i] = Complex(red, imd)
    end
    return out
end

function padimage(img::IntensityMap, alg::FFTAlg)
    padfac = alg.padfac
    ny,nx = size(img)
    nnx = nextprod((2,3,5,7), padfac*nx)
    nny = nextprod((2,3,5,7), padfac*ny)
    PaddedView(zero(eltype(img)), img, (nny, nnx))
end

function padimage(img::Union{StokesIntensityMap, IntensityMap{<:StokesParams}}, alg::FFTAlg)
    pI = padimage(stokes(img, :I), alg)
    pQ = padimage(stokes(img, :Q), alg)
    pU = padimage(stokes(img, :U), alg)
    pV = padimage(stokes(img, :V), alg)
    return StructArray{eltype(img)}((I=pI, Q=pQ, U=pU, V=pV))
end


# phasecenter the FFT.
using FastBroadcast
@fastmath function ComradeBase.phasecenter(vis, X, Y, U, V)
    x0 = first(X)
    y0 = first(Y)
    return @.. thread=true conj.(vis).*cispi.(2 * (U.*x0 .+ V'.*y0))
end


function applyfft(plan, img::AbstractArray{<:Number})
    return fftshift(plan*img)
end

function applyfft(plan, img::AbstractArray{<:StokesParams})
    visI = applyfft(plan, stokes(img, :I))
    visQ = applyfft(plan, stokes(img, :Q))
    visU = applyfft(plan, stokes(img, :U))
    visV = applyfft(plan, stokes(img, :V))
    return StructArray{StokesParams{eltype(visI)}}((I=visI, Q=visQ, U=visU, V=visV))
end

FFTW.plan_fft(A::AbstractArray{<:StokesParams}, args...) = plan_fft(stokes(A, :I), args...)

function create_cache(alg::FFTAlg, grid::AbstractGrid, pulse::Pulse=DeltaPulse())
    pimg = padimage(IntensityMap(zeros(eltype(grid.X), size(grid)), grid), alg)
    # Do the plan and then fft
    plan = plan_fft(pimg)

    #Construct the uv grid
    (;X, Y) = grid
    griduv = uviterator(size(pimg, 1), step(X), size(pimg, 2), step(Y))
    return FFTCache(alg, plan, pulse, grid, griduv)
end


"""
    uviterator(nx, dx, ny dy)

Construct the u,v iterators for the Fourier transform of the image
with pixel sizes `dx, dy` and number of pixels `nx, ny`

If you are extending Fourier transform stuff please use these functions
to ensure that the centroid is being properly computed.
"""
function uviterator(nx, dx, ny, dy)
    U = fftshift(fftfreq(nx, inv(dx)))
    V = fftshift(fftfreq(ny, inv(dy)))
    return (;U, V)
end


@fastmath function phasedecenter!(vis, X, Y, U, V)
    x0 = first(X)
    y0 = first(Y)
    @.. thread=true vis = conj(vis*cispi(-2 * (U*x0 + V'*y0)))
    return vis
end

function Serialization.serialize(s::Serialization.AbstractSerializer, cache::FFTCache)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.alg)
    Serialization.serialize(s, cache.pulse)
    Serialization.serialize(s, cache.grid)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:FFTCache})
    alg = Serialization.deserialize(s)
    pulse = Serialization.deserialize(s)
    grid = Serialization.deserialize(s)
    return create_cache(alg, grid, pulse)
end
