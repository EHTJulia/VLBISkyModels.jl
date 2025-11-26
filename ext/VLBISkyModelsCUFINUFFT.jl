module VLBISkyModelsCUFINUFFT

using VLBISkyModels
using ComradeBase: AbstractRectiGrid, UnstructuredDomain, domainpoints
using VLBISkyModels: FINUFFTAlg, FINUFFTPlan, AdjointFINPlan, _nuft!, _jlnuft!, getcache
using CUDA
using FINUFFT
const KA = CUDA.KernelAbstractions

using EnzymeCore: EnzymeRules
using EnzymeCore


function VLBISkyModels.make_plan_finufft(bI::CUDABackend, bv::CUDABackend, Ng, u::CuArray, v::CuArray, alg::FINUFFTAlg)
        pfor = FINUFFT.cufinufft_makeplan(
        2, collect(size(imgdomain)[1:2]), +1, 1, alg.reltol;
        dtype = T, upsampfac = 2.0
    )

    check_inputs(bI, bv, u, v)

    FINUFFT.cufinufft_setpts!(pfor, u, v)
    # Now we construct the adjoint plan as well
    padj = FINUFFT.cufinufft_makeplan(
        1, collect(size(imgdomain)[1:2]), -1, 1, alg.reltol;
        dtype = T, upsampfac = 2.0
    )
    FINUFFT.cufinufft_setpts!(padj, u, v)

    ccache = similar(U, Complex{T}, size(imgdomain)[1:2])
    p = FINUFFTPlan(size(u), size(imgdomain)[1:2], pfor, padj, ccache)

    function cudestroy(p::FINUFFTPlan)
        FINUFFT.cufinufft_destroy!(p.forward)
        FINUFFT.cufinufft_destroy!(p.adjoint)
    end

    return finalizer(
        cudestroy, p
    )
end

function check_inputs(bI, bv, u, v)
    return nothing
end

function check_inputs(::Any, ::CUDABackend, u, v)
    (u <: CuArray || v <: CuArray) && 
            throw(ArgumentError("U and V should not be on the GPU since the image will not be on the GPU."*
                                "If you want to use the GPU entirely for CUFINUFFT, please use CUDABackend"* 
                                "for both the image domain/grid and visibility domain." ))
    return nothing
end

function check_inputs(::CUDABackend, ::Any, u, v)
    throw(ArgumentError("Having an image on the GPU and visibilities on the CPU is not currently supported."*
                        "If you want to use the GPU entirely for CUFINUFFT, please use CUDABackend"* 
                        "for both the image domain/grid and visibility domain." ))
end

function check_inputs(::KA.GPU, ::KA.GPU, Ng, u, v)
    throw(ArgumentError("Using KernelAbstractions GPU backend is not supported for FINUFFT. Please use CUDABackend instead."))
end


const cufinplan = Base.get_extension(FINUFFT, :CUFINUFFTExt).cufinufft_plan

const GPUPlan = FINUFFTPlan{<:cufinplan}

function VLBISkyModels._jlnuft!(out, A::GPUPlan, b::AbstractArray{<:Real})
    bc = VLBISkyModels.getcache(A)
    bc .= b
    tmp = FINUFFT.cufinufft_exec(A.forward, bc)
    out .= tmp
    return nothing
end

function VLBISkyModels._jlnuft_adjointadd!(out, A::GPUPlan, b::AbstractArray{<:Complex})
    bc = getcache(A)
    FINUFFT.cufinufft_exec!(A.adjoint, b, bc)
    out .+= real.(bc)
    return nothing
end



end