module VLBISkyModelsNonuniformFFTs
using VLBISkyModels
using ComradeBase: AbstractRectiGrid, StructuredDomain
using VLBISkyModels: NonuniformFFTsAlg, _nuft!, _jlnuft!
using EnzymeCore: EnzymeRules
using EnzymeCore
using NonuniformFFTs

const KA = NonuniformFFTs.KA

function VLBISkyModels.plan_nuft_spatial(
        alg::NonuniformFFTsAlg, imgdomain::AbstractRectiGrid,
        visdomain::StructuredDomain
    )
    (; U, V) = visdomain
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
        complex(T), size(imgdomain)[1:2];
        backend, m = HalfSupport(m), σ = sigma,
        fftshift = true, # To match FINUFFT and NFFT conventions
        fftw_flags = fftflags, sort_points = True() # always sort for now
    )

    set_points!(plan, (u, v))
    return plan
end

function get_nuft_backend(imgdomain, visdomain)
    ex = executor(visdomain)
    if ex isa KA.Backend
        return ex
    else
        return CPU()
    end
end

EnzymeRules.inactive_type(::Type{<:PlanNUFFT}) = true


# TODO fix PlanNUFT to not require the adjoint since it isn't needed and we can always wrap it.
# individually. Right now we commit minor type piracy.
Base.adjoint(plan::PlanNUFFT) = plan #plan already has the adjoint method built in
VLBISkyModels.vissize(plan::PlanNUFFT) = length(first(plan.points))

function _reltol_to_m(reltol)
    w = ceil(Int, log(10, 1 / reltol)) + 1
    m = (w) ÷ 2
    return m
end

# @inline function VLBISkyModels._jlnuft!(out, A::PlanNUFFT, b::AbstractArray{<:Complex})
#     exec_type2!(out, A, b)
#     return nothing
# end

@inline function VLBISkyModels._jlnuft!(out, A::PlanNUFFT, b::AbstractArray{<:Real})
    exec_type2!(out, A, complex(b))
    return nothing
end

function VLBISkyModels._jlnuft_adjointadd!(dI, A::PlanNUFFT, dv::AbstractArray{<:Complex})
    tmp = similar(dI, eltype(dv))
    exec_type1!(tmp, A, dv)
    dI .+= real.(tmp)
    return nothing
end

end
