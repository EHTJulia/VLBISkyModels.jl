module VLBISkyModelsFINUFFT
using VLBISkyModels
using ComradeBase: AbstractRectiGrid, UnstructuredDomain, domainpoints
using VLBISkyModels: FINUFFTAlg, FINUFFTPlan, AdjointFINPlan, _nuft!, _jlnuft!
using EnzymeCore: EnzymeRules
using EnzymeCore

using FINUFFT

function VLBISkyModels.plan_nuft_spatial(
        alg::FINUFFTAlg, imgdomain::AbstractRectiGrid,
        visdomain::UnstructuredDomain
    )
    # check_image_uv(imagegrid, visdomain)
    # Check if Ti or Fr in visdomain are subset of imgdomain Ti or Fr if present
    visp = domainpoints(visdomain)
    U = visp.U
    V = visp.V
    T = eltype(U)
    dx, dy = pixelsizes(imgdomain)
    rm = ComradeBase.rotmat(imgdomain)'
    # No sign flip because we will use the FINUFFT +1 sign convention
    u = convert(T, 2π) .* VLBISkyModels._rotatex.(U, V, Ref(rm)) .* dx
    v = convert(T, 2π) .* VLBISkyModels._rotatey.(U, V, Ref(rm)) .* dy
    fftw = alg.fftflags # Convert our plans to correct numeric type
    pfor = FINUFFT.finufft_makeplan(
        2, collect(size(imgdomain)[1:2]), +1, 1, alg.reltol;
        nthreads = alg.threads,
        fftw = fftw, dtype = T, upsampfac = 2.0
    )
    FINUFFT.finufft_setpts!(pfor, u, v)
    ccache = similar(U, Complex{T}, size(imgdomain)[1:2])
    p = FINUFFTPlan(size(u), size(imgdomain)[1:2], pfor, padj, ccache)

    finalizer(
        p -> begin
            # #println("Run FINUFFT finalizer")
            FINUFFT.finufft_destroy!(p.forward)
            FINUFFT.finufft_destroy!(p.adjoint)
        end, p
    )
    return p
end

function VLBISkyModels.make_phases(
        ::FINUFFTAlg, imgdomain::AbstractRectiGrid,
        visdomain::UnstructuredDomain
    )
    # These use the same phases to just use the same code since it doesn't depend on NFFTAlg at all.
    return VLBISkyModels.make_phases(NFFTAlg(), imgdomain, visdomain)
end

@inline function VLBISkyModels._nuft!(out::StridedArray, A::FINUFFTPlan, b::StridedArray)
    _jlnuft!(out, A, b)
    return nothing
end

@inline function VLBISkyModels._nuft!(out::AbstractArray, A::FINUFFTPlan, b::AbstractArray)
    tmp = similar(out)
    _jlnuft!(tmp, A, b)
    for i in eachindex(out, tmp)
        out[i] = tmp[i]
    end
    # out .= tmp
    return nothing
end

@noinline function getcache(A::FINUFFTPlan)
    return A.ccache
end

@noinline function getcache(A::AdjointFINPlan)
    return A.plan.ccache
end

EnzymeRules.inactive(::typeof(getcache), args...) = nothing
# EnzymeRules.inactive_type(::Type{<:FINUFFT.finufft_plan}) = true

function VLBISkyModels._jlnuft!(out, A::FINUFFTPlan, b::AbstractArray{<:Real})
    bc = getcache(A)
    bc .= b
    FINUFFT.finufft_exec!(A.forward, bc, out)
    return nothing
end

function VLBISkyModels._jlnuft!(out::AbstractArray{<:Real}, A::AdjointFINPlan, b::AbstractArray{<:Complex})
    bc = getcache(A)
    FINUFFT.finufft_exec!(A.forward, b, bc)
    out .= real.(bc)
    return nothing
end

function _jlnuftadd!(out, A::AdjointFINPlan, b::AbstractArray{<:Complex})
    bc = getcache(A)
    FINUFFT.finufft_exec!(A.plan.adjoint, b, bc)
    out .+= real.(bc)
    return nothing
end

function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(_jlnuft!)},
        ::Type{RT},
        out::Annotation{<:AbstractArray{<:Complex}},
        A::Const{<:FINUFFTPlan},
        b::Annotation{<:AbstractArray{<:Real}}
    ) where {RT}
    # Forward rule does not have to return any primal or shadow since the original function returned nothing
    _jlnuft(out.val, A.val, b.val)

    if EnzymeRules.width(config) == 1
        _jlnuft(out.dval, A.val, b.dval)
    else
        ntuple(EnzymeRules.width(config)) do i
            Base.@_inline_meta
            return _jlnuft(out.dval[i], A.val, b.dval[i])
        end
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_jlnuft!)}, ::Type{<:Const},
        out::Annotation,
        A::Annotation{<:FINUFFTPlan},
        b::Annotation{<:AbstractArray{<:Real}}
    )
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

function EnzymeRules.reverse(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_jlnuft!)},
        ::Type{RT}, tape,
        out::Annotation, A::Annotation{<:FINUFFTPlan},
        b::Annotation{<:AbstractArray{<:Real}}
    ) where {RT}

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
        _jlnuftadd!(db, adjoint(A.val), dout)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end

end
