export FFTAlg

"""
    $(TYPEDEF)

Use an FFT to compute the approximate numerical visibilitymap of a model.
For a DTFT see [`DFTAlg`](@ref DFTAlg) or for an NFFT [`NFFTAlg`](@ref NFFTAlg)

# Fields
$(FIELDS)

"""
Base.@kwdef struct FFTAlg{T} <: FourierTransform
    """
    The amount to pad the image by.
    Note we actually round up to the nearest factor
    of 2, but this will be improved in the future to use
    small primes
    """
    padfac::Int = 2
    """
    FFTW flags or wisdom for the transformation. The default is `FFTW.ESTIMATE`,
    but you can use `FFTW.MEASURE` for better performance if you plan on evaluating
    the sample FFT multiple times.
    """
    flags::T = FFTW.ESTIMATE
end

function build_padded_uvgrid(grid::AbstractRectiGrid, alg::FFTAlg)
    padfac = alg.padfac
    (;X, Y) = grid
    ny,nx = size(grid)
    nnx = nextprod((2,3,5,7), padfac*nx)
    nny = nextprod((2,3,5,7), padfac*ny)
    u, v = uviterator(nnx, step(X), nny, step(Y))
    return (U=u, V=v)
end

"""
    FourierDualDomain(imgdomain::AbstractRectiGrid, alg::FFTAlg)

Constructs a FourierDualDomain that uses the FFT algorithm to compute the transformation.
For this no visibilty domain is specified since we assume it is the default grid from the
FFT with padding specified in [`FFTAlg`](@ref).

## Arguments
- `imgdomain`: The image domain grid
- `alg`: The FFT algorithm to use
"""
function FourierDualDomain(imgdomain::AbstractRectiGrid, alg::FFTAlg)
    # construct the uvgrid for the padded image
    griduv = uvgrid(imgdomain)
    plan_forward, plan_reverse = create_plans(alg, imgdomain, griduv)
    return FourierDualDomain(imgdomain, griduv, alg, plan_forward, plan_reverse)
end

function visibilitymap_numeric(m::AbstractModel, grid::FourierDualDomain{GI, GV, <:FFTAlg}) where {GI, GV}
    minterp = InterpolatedModel(m, grid)
    return visibilitymap(minterp, visdomain(grid))
end


"""
    $(TYPEDEF)
The cache used when the `FFT` algorithm is used to compute
visibilties. This is an internal type and is not part of the public API
"""
struct FFTPlan{A<:FFTAlg,P} <: AbstractPlan
    alg::A # FFT algorithm
    plan::P # FFT plan or matrix
end

function create_forward_plan(alg::FFTAlg, imgdomain::AbstractRectiGrid, ::AbstractSingleDomain)
    pimg = padimage(ComradeBase.allocate_map(Array{eltype(imgdomain)}, imgdomain), alg)
    plan = plan_fft(pimg; flags = alg.flags)
    return FFTPlan(alg, plan)
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


FFTW.plan_fft(A::AbstractArray{<:StokesParams}, args...) = plan_fft(stokes(A, :I), args...)


function inverse_plan(plan::FFTPlan)
    a = zeros(eltype(plan.plan), size(plan.plan))
    ip = plan_ifft(a; flags = plan.alg.flags)
    return FFTPlan(plan.alg, ip)
end

function applyft(plan::FFTPlan, img::AbstractArray{<:Number})
    pimg = padimage(img, plan.alg)
    return fftshift(plan.plan*pimg)
end

function applyft(plan::FFTPlan, img::AbstractArray{<:StokesParams})
    visI = applyfft(plan.plan, stokes(img, :I))
    visQ = applyfft(plan.plan, stokes(img, :Q))
    visU = applyfft(plan.plan, stokes(img, :U))
    visV = applyfft(plan.plan, stokes(img, :V))
    return StructArray{StokesParams{eltype(visI)}}((I=visI, Q=visQ, U=visU, V=visV))
end

@fastmath function phasedecenter!(vis, grid, griduv)
    (;X, Y) = grid
    (;U, V) = griduv
    x0 = first(X)
    y0 = first(Y)
    @. vis = conj(vis*cispi(-2 * (U*x0 + V'*y0)))
    return vis
end

@fastmath function phasecenter!(vis, grid, griduv)
    (;X, Y) = grid
    (;U, V) = griduv
    x0 = first(X)
    y0 = first(Y)
    @. vis = conj(vis)*cispi(2 * (U*x0 + V'*y0))
    return vis
end





###################################################################
###################################################################
### FFTW ForwardDiff overloads
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
