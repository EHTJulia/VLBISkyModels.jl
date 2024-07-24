export NFFTAlg

"""
    NFFTAlg
Uses a non-uniform FFT to compute the visibilitymap.
You can optionally pass uv which are the uv positions you will
compute the NFFT at. This can allow for the NFFT plan to be cached improving
performance

# Fields
$(FIELDS)

"""
Base.@kwdef struct NFFTAlg{T,N,F} <: NUFT
    """
    Amount to pad the image
    """
    padfac::Int = 1
    """
    Kernel size parameters. This controls the accuracy of NFFT you do not usually need to change this
    """
    m::Int = 4
    """
    Over sampling factor. This controls the accuracy of NFFT you do not usually need to change this.
    """
    σ::T = 2.0
    """
    Window function for the NFFT. You do not usually need to change this
    """
    window::Symbol = :kaiser_bessel
    """
    NFFT interpolation algorithm.
    """
    precompute::N = NFFT.POLYNOMIAL
    """
    Flag block partioning should be used to speed up computation
    """
    blocking::Bool = true
    """
    Flag if the node should be sorted in a lexicographic way
    """
    sortNodes::Bool = false
    """
    Flag if the deconvolve indices should be stored, Currently required for GPU
    """
    storeDeconvolutionIdx::Bool = true
    """
    Flag passed to inner AbstractFFT. The fastest FFTW is FFTW.MEASURE but takes the longest
    to precompute
    """
    fftflags::F = FFTW.MEASURE
end

function applyft(p::AbstractNUFTPlan, img::Union{AbstractArray,StokesIntensityMap})
    vis = nuft(getplan(p), img)
    vis .= vis .* getphases(p)
    return vis .* getphases(p)
end

function plan_nuft(alg::NFFTAlg, imagegrid::AbstractRectiGrid,
                   visdomain::UnstructuredDomain)
    visp = domainpoints(visdomain)
    uv2 = similar(visp.U, (2, length(visdomain)))
    dpx = pixelsizes(imagegrid)
    dx = dpx.X
    dy = dpx.Y
    # Here we flip the sign because the NFFT uses the -2pi convention
    uv2[1, :] .= -visp.U * dx
    uv2[2, :] .= -visp.V * dy
    (; m, σ, window, precompute, blocking, sortNodes, storeDeconvolutionIdx, fftflags) = alg
    plan = plan_nfft(uv2, size(imagegrid); m, σ, window, precompute, blocking, sortNodes,
                     storeDeconvolutionIdx, fftflags)
    return plan
end

function make_phases(::NFFTAlg, imgdomain::AbstractRectiGrid, visdomain::UnstructuredDomain)
    dx, dy = pixelsizes(imgdomain)
    x0, y0 = phasecenter(imgdomain)
    visp = domainpoints(visdomain)
    u = visp.U
    v = visp.V
    # Correct for the nFFT phase center and the img phase center
    return cispi.((u .* (dx - 2 * x0) .+ v .* (dy - 2 * y0)))
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

function _frule_nuft(p::NFFTPlan, b::AbstractArray{<:ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # Compute the fft
    buffer = ForwardDiff.value.(b)
    xtil = p * complex.(buffer)
    out = similar(buffer, Complex{ForwardDiff.Dual{T,V,P}})
    # Now take the deriv of nuft
    ndxs = ForwardDiff.npartials(first(b))
    dxtils = ntuple(ndxs) do n
        buffer .= ForwardDiff.partials.(b, n)
        return p * complex.(buffer)
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
        ΔA = @thunk(pr(A' * dy))
        return Δf, NoTangent(), ΔA
    end
    return vis, nuft_pullback
end

using EnzymeCore: EnzymeRules, Const, Active, Duplicated
#using EnzymeRules: ConfigWidth, needs_prima
function EnzymeRules.augmented_primal(config, ::Const{typeof(_nuft!)}, ::Type{<:Const}, out,
                                      A::Const, b)
    _nuft!(out.val, A.val, b.val)
    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

function EnzymeRules.reverse(config::EnzymeRules.ConfigWidth{1}, ::Const{typeof(_nuft!)},
                             ::Type{<:Const}, tape, out::Duplicated, A::Const,
                             b::Duplicated)

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
        db .+= A.val' * dout
        dout .= 0
    end
    return (nothing, nothing, nothing)
end
