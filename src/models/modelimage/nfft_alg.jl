export NFFTAlg

using NFFT


"""
    NFFTAlg(u::AbstractArray, v::AbstractArray; kwargs...)

Create an algorithm object using the non-unform Fourier transform object from uv positions
`u`, `v`. This will extract the uv positions from the observation to allow for a more efficient
FT cache.

The optional arguments are: `padfac` specifies how much to pad the image by, and `m`
is an internal variable for `NFFT.jl`.
"""
function NFFTAlg(u::AbstractArray, v::AbstractArray; kwargs...)
    uv = Matrix{eltype(u)}(undef, 2, length(u))
    uv[1,:] .= u
    uv[2,:] .= v
    return ObservedNUFT(NFFTAlg(;kwargs...), uv)
end

# pad from the center of the position.
function padimage(alg::NFFTAlg, img::SpatialIntensityMap)
    padfac = alg.padfac
    # if no padding exit now
    (padfac == 1) && return img

    ny,nx = size(img)
    nnx = nextprod((2,3,5,7), padfac*nx)
    nny = nextprod((2,3,5,7), padfac*ny)
    nsx = nnx÷2-nx÷2
    nsy = nny÷2-ny÷2
    pimg =  PaddedView(zero(eltype(img)), img.img,
                      (1:nnx, 1:nny),
                      (nsx+1:nsx+nx, nsy+1:nsy+ny)
                     )
    dx, dy = pixelsizes(img)
    return SpatialIntensityMap(collect(pimg), dx*size(pimg,2), dy*size(pimg, 1))
end

function plan_nuft(alg::ObservedNUFT{<:NFFTAlg}, img::IntensityMapTypes{T,2}) where {T}
    uv2 = similar(alg.uv)
    dpx = pixelsizes(img)
    dx = dpx.X
    dy = dpx.Y
    uv2[1,:] .= alg.uv[1,:]*dx
    uv2[2,:] .= alg.uv[2,:]*dy
    balg = alg.alg
    (;m, σ, window, precompute, blocking, sortNodes, storeDeconvolutionIdx, fftflags) = balg
    plan = plan_nfft(uv2, size(img); m, σ, window, precompute, blocking, sortNodes, storeDeconvolutionIdx, fftflags)
    return plan
end

function make_phases(alg::ObservedNUFT{<:NFFTAlg}, img::IntensityMapTypes, pulse::Pulse=DeltaPulse())
    dx, dy = pixelsizes(img)
    x0, y0 = phasecenter(img)
    u = @view alg.uv[1,:]
    v = @view alg.uv[2,:]
    # Correct for the nFFT phase center and the img phase center
    return cispi.((u.*(dx - 2*x0) .+ v.*(dy - 2*y0))).*visibility_point.(Ref(stretched(pulse, dx, dy)), u, v, zero(dx), zero(dy))
end

@inline function create_cache(alg::ObservedNUFT{<:NFFTAlg}, plan, phases, img::IntensityMapTypes, pulse=DeltaPulse())
    #timg = #SpatialIntensityMap(transpose(img.im), img.fovx, img.fovy)
    return NUFTCache(alg, plan, phases, pulse, img)
end

# Allow NFFT to work with ForwardDiff.
function _frule_vis(m::ModelImage{M,<:SpatialIntensityMap{<:ForwardDiff.Dual{T,V,P}},<:NUFTCache{O}}) where {M,T,V,P,A<:NFFTAlg,O<:ObservedNUFT{A}}
    p = m.cache.plan
    # Compute the fft
    bimg = parent(m.cache.img)
    buffer = ForwardDiff.value.(bimg)
    xtil = p*complex.(buffer)
    out = similar(buffer, Complex{ForwardDiff.Dual{T,V,P}})
    # Now take the deriv of nuft
    ndxs = ForwardDiff.npartials(first(m.cache.img))
    dxtils = ntuple(ndxs) do n
        buffer .= ForwardDiff.partials.(m.cache.img, n)
        p * complex.(buffer)
    end
    out = similar(xtil, Complex{ForwardDiff.Dual{T,V,P}})
    for i in eachindex(out)
        dual = getindex.(dxtils, i)
        prim = xtil[i]
        red = ForwardDiff.Dual{T,V,P}(real(prim), ForwardDiff.Partials(real.(dual)))
        imd = ForwardDiff.Dual{T,V,P}(imag(prim), ForwardDiff.Partials(imag.(dual)))
        out[i] = Complex(red, imd)
    end
    return out
end

function visibilities_numeric(m::ModelImage{M,<:SpatialIntensityMap{<:ForwardDiff.Dual{T,V,P}},<:NUFTCache{O}},
    u::AbstractArray,
    v::AbstractArray,
    time,
    freq) where {M,T,V,P,A<:NFFTAlg,O<:ObservedNUFT{A}}
    checkuv(m.cache.alg.uv, u, v)
    # Now reconstruct everything

    vis = _frule_vis(m)
    return conj.(vis).*m.cache.phases
end


nuft(A::NFFTPlan, b::AbstractArray{<:Real}) = nuft(A, complex(b))

function nuft(A::NFFTPlan, b::AbstractArray{<:Complex})
    out = similar(b, eltype(A), size(A)[1])
    _nuft!(out, A, b)
    return out
end

function _nuft!(out, A, b)
    mul!(out, A, b)
    return nothing
end

function ChainRulesCore.rrule(::typeof(nuft), A::NFFTPlan, b)
    pr = ChainRulesCore.ProjectTo(b)
    vis = nuft(A, b)
    function nuft_pullback(Δy)
        Δf = NoTangent()
        dy = similar(vis)
        dy .= unthunk(Δy)
        ΔA = @thunk(pr(A'*dy))
        return Δf, NoTangent(), ΔA
    end
    return vis, nuft_pullback
end

using EnzymeCore: EnzymeRules, Const, Active, Duplicated
#using EnzymeRules: ConfigWidth, needs_prima
function EnzymeRules.augmented_primal(config, ::Const{typeof(_nuft!)}, ::Type{<:Const}, out, A::Const, b)
    # It seems that if we don't change out.val the entire primal is skipped
    println("In augmented primal")
    _nuft!(out.val, A.val, b.val)

    cache_A = (EnzymeRules.overwritten(config)[3]) ? copy(A.val) : nothing

    return EnzymeRules.AugmentedReturn(nothing, nothing, cache_A)
end

function EnzymeRules.reverse(config::EnzymeRules.ConfigWidth{1}, ::Const{typeof(_nuft!)}, ::Type{<:Const}, tape, out::Duplicated, A::Const, b::Duplicated)
    # #TODO do I need to accumulate?
    cache_A = tape
    if !(EnzymeRules.overwritten(config)[3])
        cache_A = A.val
    end
    println("In reverse")
    dbs = if EnzymeRules.width(config) == 1
        (b.dval,)
    else
        b.dval
    end

    douts = if EnzymeRules.width(config) == 1
        (out.dval,)
    else
        out.dval
    end
    for (db, dout) in zip(dbs, douts)
        db .+= cache_A'*dout
        println("db: ", db)
        println("dout: ", dout)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end
