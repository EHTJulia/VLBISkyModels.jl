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

"""
This function helps us to lookup UnstructuredDomain at a particular Ti or Fr 
visdomain[Ti=T,Fr=F] or visdomain[Ti=T] or visdomain[Fr=F] calls work. 
Note: I cannot figure out how to write this function without specifying 
nothing to Ti or Fr. I tried kwargs... as well.
"""
function Base.getindex(domain::UnstructuredDomain; Ti=nothing, Fr=nothing)
    points = domainpoints(domain)
    indices = if Ti !== nothing && Fr !== nothing
        findall(p -> (p.Ti == Ti) && (p.Fr == Fr), points)
    elseif Ti !== nothing
        findall(p -> (p.Ti == Ti), points)
    elseif Fr !== nothing
        findall(p -> (p.Fr == Fr), points)
    else
        1:length(points)
    end
    return UnstructuredDomain(points[indices], executor(domain), header(domain))
end

# I am not sure if we need the same for UnstructuredMap as well.
function Base.getindex(domain::UnstructuredMap; Ti=nothing, Fr=nothing)
    points = domainpoints(domain)
    indices = if Ti !== nothing && Fr !== nothing
        findall(p -> (p.Ti == Ti) && (p.Fr == Fr), points)
    elseif Ti !== nothing
        findall(p -> (p.Ti == Ti), points)
    elseif Fr !== nothing
        findall(p -> (p.Fr == Fr), points)
    else
        1:length(points)
    end
    return UnstructuredMap(points[indices], executor(domain), header(domain))
end

function applyft(p::AbstractNUFTPlan, img::Union{AbstractArray,StokesIntensityMap})
    vis = nuft(getplan(p), img)
    vis .= vis .* getphases(p)
    return vis
end

"""
This a new function is overloaded to handle when NUFTPlan has plans
as dictionaries in the case of Ti or Fr case
"""
@inline function applyft(p::NUFTPlan{<:FourierTransform,<:AbstractDict},
                         img::Union{AbstractArray,StokesIntensityMap})
    vis_list = zeros(ComplexF64, p.totalvis)
    pimg = baseimage(img)
    plans = p.plan
    iminds, visinds = p.indices

    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        visind = visinds[i]
        vis_inner = nuft(plans[imind], @view(pimg[:, :, imind...]))
        vis_list[visind] .= vis_inner
    end

    vis_list .= vis_list .* p.phases
    return vis_list
end

"""
plan_nuft for only spatial part, no Ti or Fr
"""
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

"""
plan_nuft_spatial functions mapped to times Ti and frequencies Fr 
"""
function plan_nuft(alg::NFFTAlg, imagegrid::AbstractRectiGrid,
                   visdomain::UnstructuredDomain, indices::Tuple)
    # check_image_uv(imagegrid, visdomain) # Check if Ti or Fr in visdomain are subset of imgdomain Ti or Fr if present
    points = domainpoints(visdomain)
    iminds, visinds = indices

    uv = UnstructuredDomain(points[visinds[1]], executor(visdomain), header(visdomain))
    tplan = plan_nuft_spatial(alg, imagegrid, uv)
    plans = Dict{typeof(iminds[1]),typeof(tplan)}()

    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        visind = visinds[i]
        uv = UnstructuredDomain(points[visind], executor(visdomain), header(visdomain))
        plans[imind...] = plan_nuft_spatial(alg, imagegrid, uv)
    end

    return plans
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
        db .+= real.(A.val' * dout)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end
