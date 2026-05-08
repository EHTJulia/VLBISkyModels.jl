module VLBISkyModelsReactantExt

using VLBISkyModels
using AbstractFFTs
using ComradeBase
using Reactant
using NFFT
using NFFT: AbstractNFFTs
using VLBISkyModels: ReactantNUFFTAlg
using LinearAlgebra

include("nufft/ReactantNUFFT.jl")

function VLBISkyModels.applyphases!(vis::Reactant.AnyTracedRArray, phases::AbstractArray)
    vout = vis .* phases
    return copyto!(vis, vout)
end

function VLBISkyModels.PolExp2Map!(
        a::Reactant.AnyTracedRArray,
        b::Reactant.AnyTracedRArray,
        c::Reactant.AnyTracedRArray,
        d::Reactant.AnyTracedRArray,
        grid::ComradeBase.AbstractRectiGrid
    )

    # TODO figure out why the regular looped version isn't getting
    # raised nicely? Looks like some dus is getting in the way?
    p = sqrt.(b .^ 2 .+ c .^ 2 .+ d .^ 2)
    pimgI = exp.(a) .* cosh.(p)
    tmp = exp.(a) .* sinh.(p) ./ p
    pimgQ = tmp .* b
    pimgU = tmp .* c
    pimgV = tmp .* d

    copyto!(a, pimgI)
    copyto!(b, pimgQ)
    copyto!(c, pimgU)
    copyto!(d, pimgV)

    return stokes_intensitymap(a, b, c, d, grid)
end



# function VLBISkyModels.FourierDualDomain(
#         imgdomain::ComradeBase.AbstractRectiGrid, visdomain::ComradeBase.UnstructuredDomain,
#         algorithm::ReactantNUFFTAlg
#     )
#     plan_forward, plan_reverse = VLBISkyModels.create_plans(algorithm, imgdomain, visdomain)
#     return FourierDualDomain(imgdomain, Reactant.to_rarray(visdomain), algorithm, plan_forward, plan_reverse)
# end

end
