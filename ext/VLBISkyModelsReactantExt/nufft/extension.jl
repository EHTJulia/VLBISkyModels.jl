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
        alg::VLBISkyModels.ReactantNUFFTAlg, imgdomain::ComradeBase.AbstractRectiGrid, visdomain::UnstructuredDomain
    )
    visp = domainpoints(visdomain)
    U = visp.U
    V = visp.V
    T = eltype(U)
    dx, dy = pixelsizes(imgdomain)
    rm = ComradeBase.rotmat(imgdomain)'
    # No sign flip because we will use the FINUFFT +1 sign convention
    u = convert(T, 2π) .* VLBISkyModels._rotatex.(U, V, Ref(rm)) .* dx
    v = convert(T, 2π) .* VLBISkyModels._rotatey.(U, V, Ref(rm)) .* dy
    pl = plan_nufft(unwrapped_eltype(u), 2, size(imgdomain)[1:2]; iflag = +1, opts = alg)
    if ReactantCore.within_compile()
        pls = set_nufft_points(pl, (u, v))
    else
        pls = @jit set_nufft_points(pl, (u, v))
    end
    return pls
end

function VLBISkyModels.make_phases(
        ::ReactantNUFFTAlg, imgdomain::ComradeBase.AbstractRectiGrid,
        visdomain::UnstructuredDomain
    )
    # These use the same phases to just use the same code since it doesn't depend on NFFTAlg at all.
    return VLBISkyModels.make_phases(NFFTAlg(), imgdomain, visdomain)
end

Base.adjoint(plan::NUFFTSetPts) = plan # Not needed Reactant is too smart for this
VLBISkyModels.vissize(plan::NUFFTSetPts) = plan.M
