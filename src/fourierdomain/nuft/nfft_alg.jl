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
    relative tolerance of the NFFT
    """
    reltol::T = 1e-9
    """
    NFFT interpolation algorithm.
    """
    precompute::N = NFFT.TENSOR
    """
    Flag passed to inner AbstractFFT. The fastest FFTW is FFTW.MEASURE but takes the longest
    to precompute
    """
    fftflags::F = FFTW.MEASURE
end

function applyft(p::AbstractNUFTPlan, img::AbstractArray)
    vis = nuft(getplan(p), img)
    vis .= vis .* getphases(p)
    return vis
end

# This a new function is overloaded to handle when NUFTPlan has plans
# as dictionaries in the case of Ti or Fr case

@inline function applyft(p::NUFTPlan{<:FourierTransform,<:AbstractDict},
                         img::AbstractArray)
    vis_list = similar(baseimage(img), Complex{eltype(img)}, p.totalvis)
    plans = p.plan
    iminds, visinds = p.indices
    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        visind = visinds[i]
        # TODO
        # If visinds are consecutive then we can use the in-place _nuft!:
        # _nuft!(visind, plans[imind], @view(img[:, :, imind...])  
        vis_inner = nuft(plans[imind], @view(img[:, :, imind]))
        # After the todo this wont be required
        vis_list[visind] .= vis_inner
    end

    vis_list .= vis_list .* p.phases
    return vis_list
end

function plan_nuft_spatial(alg::NFFTAlg, imagegrid::AbstractRectiGrid,
                           visdomain::UnstructuredDomain)
    visp = domainpoints(visdomain)
    uv2 = similar(visp.U, (2, length(visdomain)))
    dpx = pixelsizes(imagegrid)
    dx = dpx.X
    dy = dpx.Y
    # Here we flip the sign because the NFFT uses the -2pi convention
    uv2[1, :] .= -visp.U .* dx
    uv2[2, :] .= -visp.V .* dy
    (; reltol, precompute, fftflags) = alg
    plan = plan_nfft(uv2, size(imagegrid)[1:2]; reltol, precompute, fftflags)
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
    out = similar(b, eltype(A), size(A)[1])
    _nuft!(out, A, b)
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

# function ChainRulesCore.rrule(::typeof(_nuft), A::NFFTPlan, b)
#     pr = ChainRulesCore.ProjectTo(b)
#     vis = nuft(A, b)
#     function nuft_pullback(Δy)
#         Δf = NoTangent()
#         dy = similar(vis)
#         dy .= unthunk(Δy)
#         ΔA = @thunk(pr(A' * dy))
#         return Δf, NoTangent(), ΔA
#     end
#     return vis, nuft_pullback
# end

#using EnzymeRules: ConfigWidth, needs_prima
function EnzymeRules.augmented_primal(config, ::Const{typeof(_nuft!)}, ::Type{<:Const}, out,
                                      A::Const, b)
    _nuft!(out.val, A.val, b.val)
    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    return EnzymeRules.AugmentedReturn(nothing, nothing, nothing)
end

@noinline function EnzymeRules.reverse(config::EnzymeRules.ConfigWidth{1}, ::Const{typeof(_nuft!)},
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
        db .+= real.(A.val' * dout)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end
