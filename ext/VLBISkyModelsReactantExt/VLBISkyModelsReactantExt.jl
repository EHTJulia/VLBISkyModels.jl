module VLBISkyModelsReactantExt

using VLBISkyModels
using AbstractFFTs
using ComradeBase
using Reactant
using NFFT
using NFFT: AbstractNFFTs
using VLBISkyModels: ReactantAlg
using LinearAlgebra

include("nfft.jl")

function VLBISkyModels.FourierDualDomain(
        imgdomain::ComradeBase.AbstractRectiGrid, visdomain::ComradeBase.UnstructuredDomain,
        algorithm::ReactantAlg
    )
    plan_forward, plan_reverse = VLBISkyModels.create_plans(algorithm, imgdomain, visdomain)
    return FourierDualDomain(imgdomain, Reactant.to_rarray(visdomain), algorithm, plan_forward, plan_reverse)
end


function VLBISkyModels.plan_nuft_spatial(
        ::ReactantAlg,
        imgdomain::ComradeBase.AbstractRectiGrid,
        visdomain::ComradeBase.UnstructuredDomain,
    )
    visp = domainpoints(visdomain)
    uv2 = similar(visp.U, (2, length(visdomain)))
    dpx = pixelsizes(imgdomain)
    dx = dpx.X
    dy = dpx.Y
    rm = ComradeBase.rotmat(imgdomain)'
    # Here we flip the sign because the NFFT uses the -2pi convention
    uv2[1, :] .= -VLBISkyModels._rotatex.(visp.U, visp.V, Ref(rm)) .* dx
    uv2[2, :] .= -VLBISkyModels._rotatey.(visp.U, visp.V, Ref(rm)) .* dy
    return plan_nfft(NFFTBackend(), Reactant.RArray, uv2, size(imgdomain)[1:2])
end

function VLBISkyModels.make_phases(
        ::ReactantAlg,
        imgdomain::ComradeBase.AbstractRectiGrid,
        visdomain::ComradeBase.UnstructuredDomain,
    )
    return Reactant.to_rarray(VLBISkyModels.make_phases(NFFTAlg(), imgdomain, visdomain))
end

function VLBISkyModels._jlnuft!(out, A::Reactant_NFFTPlan, inp::Reactant.AnyTracedRArray)
    LinearAlgebra.mul!(out, A, inp)
    return nothing
end




end
