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

# This a new function is overloaded to handle when NUFTPlan has plans
# as dictionaries in the case of Ti or Fr case

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

function _nuft(A::NUFTPlan, b::AbstractArray{<:ForwardDiff.Dual})
    return _frule_nuft(A, b)
end

function _frule_nuft(A::NUFTPlan, b::AbstractArray{<:ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # Compute the fft
    p = getplan(A)
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

# We split on a strided array since NFFT.jl only works on those
# and for StridedArrays we can potentially save an allocation
@inline function _nuft!(out::StridedArray, A::NFFTPlan, b::StridedArray)
    _jlnuft!(out, A, b)
    return nothing
end

@inline function _nuft!(out::AbstractArray, A::NFFTPlan, b::AbstractArray)
    tmp = similar(out)
    _jlnuft!(tmp, A, b)
    out .= tmp
    return nothing
end

@inline function _jlnuft!(out, A, b)
    mul!(out, A, b)
    return nothing
end

function forward(
    config::EnzymeRules.FwdConfig,
    func::Const{typeof(_jlnuft!)}, 
    RT, 
    out::Annotation{<:AbstractArray{<:Complex}}, A::Const{<:NFFTPlan},
    b::Annotation{<:AbstractArray{<:Real}}
) 
    if EnzymeRules.needs_primal(config) && EnzymeRules.needs_shadow(config)
        func.val(out.dval, A, b.dval)
        if EnzymeRules.width(config) == 1
            return Duplicated(out.val, out.dval)
        else
            func.val.(out.dval, Ref(A), b.dval)
            return BatchDuplicated(
                func.val(out.val, A.val, b.val), 
                ntuple(
                    i -> out.dval[i], Val(EnzymeRules.width(config))
                )
            )
        end
    elseif EnzymeRules.needs_shadow(config)
        if EnzymeRules.width(config) == 1
            func.val(out.dval, A, b.dval)
            return out.dval
        else
            func.val.(out.dval, Ref(A), b.dval)
            return ntuple(i -> out.dval[i], Val(EnzymeRules.width(config)))
        end
    elseif EnzymeRules.needs_primal(config)
        return func.val(out.val, A.val, b.val)
    else
        return nothing
    end
end


function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfigWidth,
                                      ::Const{typeof(_jlnuft!)}, ::Type{<:Const},
                                      out::Annotation,
                                      A::Annotation{<:NFFTPlan},
                                      b::Annotation{<:AbstractArray{<:Real}})
    isa(A, Const) ||
        throw(ArgumentError("A must be a constant in NFFT. We don't support dynamic plans"))
    primal = EnzymeRules.needs_primal(config) ? out.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? out.dval : nothing
    cache_out = EnzymeRules.overwritten(config)[2] ? out : nothing
    cache_b = EnzymeRules.overwritten(config)[4] ? b : nothing
    tape = (cache_out, cache_b)
    _jlnuft!(out.val, A.val, b.val)
    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth,
                             ::Const{typeof(_jlnuft!)},
                             ::Type{RT}, tape,
                             out::Annotation, A::Annotation{<:NFFTPlan},
                             b::Annotation{<:AbstractArray{<:Real}}) where {RT}

    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    # cache_A = tape
    # if !(EnzymeRules.overwritten(config)[3])
    #     cache_A = A.val
    # end
    isa(A, Const) ||
        throw(ArgumentError("A must be a constant in NFFT. We don't support dynamic plans"))

    # There is no gradient to propagate so short
    if isa(out, Const)
        return (nothing, nothing, nothing)
    end

    outfwd = EnzymeRules.overwritten(config)[2] ? tape[1] : out
    bfwd = EnzymeRules.overwritten(config)[4] ? tape[2] : b

    # This is so Enzyme batch mode works
    dbs = if EnzymeRules.width(config) == 1
        (bfwd.dval,)
    else
        bfwd.dval
    end

    douts = if EnzymeRules.width(config) == 1
        (outfwd.dval,)
    else
        outfwd.dval
    end
    for (db, dout) in zip(dbs, douts)
        # TODO open PR on NFFT so we can do this in place.
        db .+= real.(A.val' * dout)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end


function EnzymeRules.reverse(config::EnzymeRules.RevConfigWidth,
                                       ::Const{typeof(_jlnuft!)},
                                       ::Type{<:Const}, tape,
                                       out::Duplicated, A::Const{<:NFFTPlan},
                                       b::Duplicated{<:AbstractArray{<:Real}})

    # I think we don't need to cache this since A just has in internal temporary buffer
    # that is used to store the results of things like the FFT.
    # cache_A = (EnzymeRules.overwritten(config)[3]) ? A.val : nothing
    # cache_A = tape
    # if !(EnzymeRules.overwritten(config)[3])
    #     cache_A = A.val
    # end

    outfwd = EnzymeRules.overwritten(config)[2] ? tape[1] : out
    bfwd = EnzymeRules.overwritten(config)[4] ? tape[2] : b

    # This is so Enzyme batch mode works
    dbs = if EnzymeRules.width(config) == 1
        (bfwd.dval,)
    else
        bfwd.dval
    end

    douts = if EnzymeRules.width(config) == 1
        (outfwd.dval,)
    else
        outfwd.dval
    end
    for (db, dout) in zip(dbs, douts)
        # TODO open PR on NFFT so we can do this in place.
        db .+= real.(A.val' * dout)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end
