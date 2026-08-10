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

# Batched Stokes transform: stack I,Q,U,V into a trailing ntrans axis and run a
# single type-2 NUFFT rather than four independent transforms (one fused
# oversampled FFT + one stencil gather instead of four of each).
function VLBISkyModels.nuft(
        A::VLBISkyModels.NUFTPlan{<:Any, <:NUFFTSetPts},
        b::ComradeBase.IntensityMap{<:VLBISkyModels.StokesParams}
    )
    fk = cat(
        ComradeBase.baseimage(stokes(b, :I)),
        ComradeBase.baseimage(stokes(b, :Q)),
        ComradeBase.baseimage(stokes(b, :U)),
        ComradeBase.baseimage(stokes(b, :V));
        dims = 3
    )
    c = execute_nufft(VLBISkyModels.getplan(A), complex.(fk))
    return VLBISkyModels.StructArrays.StructArray{VLBISkyModels.StokesParams{eltype(c)}}(
        (I = c[:, 1], Q = c[:, 2], U = c[:, 3], V = c[:, 4])
    )
end

# Multidomain (Ti/Fr) polarized transform. Each subdomain has its own
# nonuniform point set, so the subdomains cannot share one ntrans axis;
# instead batch the 4 Stokes within each subdomain's plan (one transform per
# subdomain instead of four).
function VLBISkyModels.nuft(
        A::VLBISkyModels.NUFTPlan{<:Any, <:AbstractDict{<:Any, <:NUFFTSetPts}},
        b::ComradeBase.IntensityMap{<:VLBISkyModels.StokesParams}
    )
    bI = ComradeBase.baseimage(stokes(b, :I))
    bQ = ComradeBase.baseimage(stokes(b, :Q))
    bU = ComradeBase.baseimage(stokes(b, :U))
    bV = ComradeBase.baseimage(stokes(b, :V))
    CT = complex(unwrapped_eltype(bI))
    visI = similar(bI, CT, A.totalvis)
    visQ = similar(bI, CT, A.totalvis)
    visU = similar(bI, CT, A.totalvis)
    visV = similar(bI, CT, A.totalvis)
    plans = VLBISkyModels.getplan(A)
    iminds, visinds = VLBISkyModels.getindices(A)
    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        visind = visinds[i]
        length(visind) == 0 && continue
        fk = cat(
            @view(bI[:, :, imind]), @view(bQ[:, :, imind]),
            @view(bU[:, :, imind]), @view(bV[:, :, imind]);
            dims = 3
        )
        c = execute_nufft(plans[imind], complex.(fk))
        copyto!(@view(visI[visind]), c[:, 1])
        copyto!(@view(visQ[visind]), c[:, 2])
        copyto!(@view(visU[visind]), c[:, 3])
        copyto!(@view(visV[visind]), c[:, 4])
    end
    return VLBISkyModels.StructArrays.StructArray{VLBISkyModels.StokesParams{eltype(visI)}}(
        (I = visI, Q = visQ, U = visU, V = visV)
    )
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

function VLBISkyModels.make_phases(
        ::ReactantNUFFTAlg, imgdomain::ComradeBase.AbstractRectiGrid,
        visdomain::UnstructuredDomain
    )
    # These use the same phases to just use the same code since it doesn't depend on NFFTAlg at all.
    return VLBISkyModels.make_phases(NFFTAlg(), imgdomain, visdomain)
end

Base.adjoint(plan::NUFFTSetPts) = plan # Not needed Reactant is too smart for this
VLBISkyModels.vissize(plan::NUFFTSetPts) = plan.M
