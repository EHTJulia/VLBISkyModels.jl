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

# function VLBISkyModels.FourierDualDomain(
#         imgdomain::ComradeBase.AbstractRectiGrid, visdomain::ComradeBase.UnstructuredDomain,
#         algorithm::ReactantNUFFTAlg
#     )
#     plan_forward, plan_reverse = VLBISkyModels.create_plans(algorithm, imgdomain, visdomain)
#     return FourierDualDomain(imgdomain, Reactant.to_rarray(visdomain), algorithm, plan_forward, plan_reverse)
# end

end
