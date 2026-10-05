using VLBISkyModels
using Reactant: unwrapped_eltype
using ReactantCore

function VLBISkyModels._jlnuft!(out, A::NUFFTSetPts, b::Reactant.AnyTracedRArray{<:Real})
    VLBISkyModels._jlnuft!(out, A, complex.(b))
    return nothing
end

function VLBISkyModels._jlnuft!(out, A::NUFFTSetPts, b::Reactant.AnyTracedRArray{<:Complex})
    execute_nufft!(out, A, b)
    return nothing
end

function VLBISkyModels.plan_nuft_spatial(
        alg::VLBISkyModels.ReactantNUFFTAlg, imgdomain::ComradeBase.AbstractRectiGrid, visdomain::ComradeBase.StructuredDomain
    )
    (; U, V) = visdomain
    T = eltype(U)
    dx, dy = pixelsizes(imgdomain)
    rm = ComradeBase.rotmat(imgdomain)'
    # No sign flip because we will use the FINUFFT +1 sign convention
    pl = plan_nufft(unwrapped_eltype(U), 2, size(imgdomain)[1:2]; iflag = +1, opts = alg)
    if ReactantCore.within_compile()
        u = convert(T, 2π) .* VLBISkyModels._rotatex.(U, V, Ref(rm)) .* dx
        v = convert(T, 2π) .* VLBISkyModels._rotatey.(U, V, Ref(rm)) .* dy
        pls = set_nufft_points(pl, (u, v))
    else
        u = @jit convert(T, 2π) .* VLBISkyModels._rotatex.(U, V, Ref(rm)) .* dx
        v = @jit convert(T, 2π) .* VLBISkyModels._rotatey.(U, V, Ref(rm)) .* dy
        pls = @jit set_nufft_points(pl, (u, v))
    end
    return pls
end

Base.adjoint(plan::NUFFTSetPts) = plan # Not needed Reactant is too smart for this
VLBISkyModels.vissize(plan::NUFFTSetPts) = plan.M
