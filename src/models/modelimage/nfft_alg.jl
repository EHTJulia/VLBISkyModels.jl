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

function plan_nuft(alg::ObservedNUFT{<:NFFTAlg}, grid::AbstractGrid)
    uv2 = similar(alg.uv)
    dpx = pixelsizes(grid)
    dx = dpx.X
    dy = dpx.Y
    uv2[1,:] .= alg.uv[1,:]*dx
    uv2[2,:] .= alg.uv[2,:]*dy
    balg = alg.alg
    (;m, σ, window, precompute, blocking, sortNodes, storeDeconvolutionIdx, fftflags) = balg
    plan = plan_nfft(uv2, size(grid); m, σ, window, precompute, blocking, sortNodes, storeDeconvolutionIdx, fftflags)
    return plan
end

function make_phases(alg::ObservedNUFT{<:NFFTAlg}, grid::AbstractGrid, pulse::Pulse=DeltaPulse())
    dx, dy = pixelsizes(grid)
    x0, y0 = phasecenter(grid)
    u = @view alg.uv[1,:]
    v = @view alg.uv[2,:]
    # Correct for the nFFT phase center and the img phase center
    return cispi.((u.*(dx - 2*x0) .+ v.*(dy - 2*y0))).*visibility_point.(Ref(stretched(pulse, dx, dy)), u, v, zero(dx), zero(dy))
end

@inline function create_cache(alg::ObservedNUFT{<:NFFTAlg}, plan, phases, grid::AbstractGrid, pulse=DeltaPulse())
    return NUFTCache(alg, plan, phases, pulse, grid)
end

# Allow NFFT to work with ForwardDiff.


function _nuft(A::NFFTPlan, b::AbstractArray{<:Real})
    bc = complex(b)
    out = similar(b, eltype(A), size(A)[1])
    _nuft!(out, A, bc)
    return out
end


function _nuft(A::NFFTPlan, b::AbstractArray{<:ForwardDiff.Dual})
    return _frule_nuft(A, b)
end

function _frule_nuft(p::NFFTPlan, b::AbstractArray{<:ForwardDiff.Dual})
    # Compute the fft
    buffer = ForwardDiff.value.(b)
    xtil = p*complex.(buffer)
    out = similar(buffer, Complex{ForwardDiff.Dual{T,V,P}})
    # Now take the deriv of nuft
    ndxs = ForwardDiff.npartials(first(m.image))
    dxtils = ntuple(ndxs) do n
        buffer .= ForwardDiff.partials.(m.image, n)
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



function _nuft!(out, A, b)
    mul!(out, A, b)
    return nothing
end

function ChainRulesCore.rrule(::typeof(_nuft), A::NFFTPlan, b)
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
    _nuft!(out.val, A.val, b.val)
    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config::EnzymeRules.ConfigWidth{1}, ::Const{typeof(_nuft!)}, ::Type{<:Const}, tape, out::Duplicated, A::Const, b::Duplicated)

    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    # cache_A = tape
    # if !(EnzymeRules.overwritten(config)[3])
    #     cache_A = A.val
    # end

    # This is so Enzyme batch mode works
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
        # TODO open PR on NFFT so we can do this in place.
        db .+= A.val'*dout
        dout .= 0
    end
    return (nothing, nothing, nothing)
end
