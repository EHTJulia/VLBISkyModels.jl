module VLBISkyModelsNonuniformFFTs
using VLBISkyModels
using ComradeBase: AbstractRectiGrid, UnstructuredDomain, domainpoints
using VLBISkyModels: NonuniformFFTAlg, _nuft!, _jlnuft!
using EnzymeCore: EnzymeRules
using EnzymeCore
using NonuniformFFTs

const KA = NonuniformFFTs.KA

function VLBISkyModels.plan_nuft_spatial(
        alg::NonuniformFFTAlg, imgdomain::AbstractRectiGrid,
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


    (; padfac, m, sigma, fftflags) = alg
    backend = get_nuft_backend(imgdomain, visdomain)
    if m < 0
        m = _reltol_to_m(alg.reltol)
    end

    plan = PlanNUFFT(
        Complex{T}, size(imgdomain)[1:2];
        backend, m = HalfSupport(m), σ = sigma,
        fftshift = true, # To match FINUFFT and NFFT conventions
        fftw_flags = fftflags, sort_points = True() # always sort for now
    )

    set_points!(plan, (u, v))
    return plan
end

function get_nuft_backend(imgdomain, visdomain)
    if executor(visdomain) isa KA.Backend
        return executor(visdomain)
    else
        return CPU()
    end
end

# TODO fix PlanNUFT to not require the adjoint since it isn't needed and we can always wrap it.
# individually. Right now we commit minor type piracy.
Base.adjoint(plan::PlanNUFFT) = plan #plan already has the adjoint method built in

function _reltol_to_m(reltol)
    w = ceil(Int, log(10, 1 / reltol)) + 1
    m = (w) ÷ 2
    return m
end

function VLBISkyModels.make_phases(
        ::NonuniformFFTAlg, imgdomain::AbstractRectiGrid,
        visdomain::UnstructuredDomain
    )
    # These use the same phases to just use the same code since it doesn't depend on NFFTAlg at all.
    return VLBISkyModels.make_phases(NFFTAlg(), imgdomain, visdomain)
end

function VLBISkyModels._nuft(A::PlanNUFFT, b::AbstractArray{<:Real})
    out = similar(b, eltype(A), length(A.points[1]))
    _nuft!(out, A, b)
    return out
end


@inline function VLBISkyModels._jlnuft!(out, A::PlanNUFFT, b::AbstractArray{<:Complex})
    exec_type2!(out, A, b)
    return nothing
end

@inline function VLBISkyModels._jlnuft!(out, A::PlanNUFFT, b::AbstractArray{<:Real})
    exec_type2!(out, A, complex(b))
    return nothing
end


function EnzymeRules.forward(
        config::EnzymeRules.FwdConfig,
        func::Const{typeof(_jlnuft!)},
        ::Type{RT},
        out::Annotation{<:AbstractArray{<:Complex}},
        A::Const{<:PlanNUFFT},
        b::Annotation{<:AbstractArray{<:Real}}
    ) where {RT}
    # Forward rule does not have to return any primal or shadow since the original function returned nothing
    func.val(out.val, A.val, b.val)
    if EnzymeRules.width(config) == 1
        func.val(out.dval, A.val, b.dval)
    else
        ntuple(EnzymeRules.width(config)) do i
            Base.@_inline_meta
            return func.val(out.dval[i], A.val, b.dval[i])
        end
    end
    return nothing
end

function EnzymeRules.augmented_primal(
        config::EnzymeRules.RevConfigWidth,
        ::Const{typeof(_jlnuft!)}, ::Type{<:Const},
        out::Annotation,
        A::Annotation{<:PlanNUFFT},
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
        out::Annotation, A::Annotation{<:PlanNUFFT},
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
        # TODO open PR on NFFT so we can do this in place.
        tmp = similar(dout, Complex{eltype(dout)})
        exec_type1!(tmp, A.val, dout)
        db .+= real.(tmp)
        dout .= 0
    end
    return (nothing, nothing, nothing)
end


end
