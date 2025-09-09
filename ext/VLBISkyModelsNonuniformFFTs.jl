module VLBISkyModelsNonuniformFFTs
using VLBISkyModels
using ComradeBase: AbstractRectiGrid, UnstructuredDomain, domainpoints
using VLBISkyModels: NonuniformFFTAlg, _nuft!, _jlnuft!
using EnzymeCore: EnzymeRules
using EnzymeCore
using NonuniformFFTs

function VLBISkyModels.plan_nuft_spatial(
        alg::NonuniFFTAlg, imgdomain::AbstractRectiGrid,
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


    (; backend, padfac, m, sigma, fftflags) = alg
    if isnothing(backend)
        backend = CPU()
    end

    plan = PlanNUFFT(Complex{T}, size(imgdomain)[1:2]; 
                     backend, m, sigma, 
                     fftw_flags=fftflags, sort_nodes=True()
                    )
    return plan
end

end