module VLBISkyModelsCUFINUFFT

using VLBISkyModels
using ComradeBase: AbstractRectiGrid, UnstructuredDomain, domainpoints
using VLBISkyModels: FINUFFTAlg, FINUFFTPlan, AdjointFINPlan, _nuft!, _jlnuft!, getcache
using CUDA
using FINUFFT

using EnzymeCore: EnzymeRules
using EnzymeCore


function VLBISkyModels.plan_nuft_spatial(
        alg::FINUFFTAlg{true}, imgdomain::AbstractRectiGrid,
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

    pfor = FINUFFT.cufinufft_makeplan(
        2, collect(size(imgdomain)[1:2]), +1, 1, alg.reltol;
        dtype = T, upsampfac = 2.0
    )
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

const cufinplan = Base.get_extension(FINUFFT, :CUFINUFFTExt).cufinufft_plan

const GPUPlan = FINUFFTPlan{<:Any, <:Any, <:Any, <:cufinplan, <:cufinplan, <:Any}

function VLBISkyModels._jlnuft!(out, A::GPUPlan, b::AbstractArray{<:Real})
    bc = VLBISkyModels.getcache(A)
    bc .= b
    tmp = FINUFFT.cufinufft_exec(A.forward, bc)
    out .= tmp
    return nothing
end

function VLBISkyModels._jlnuft!(out, A::AdjointFINPlan{P}, b::AbstractArray{<:Complex}) where {P <: GPUPlan}
    # bc = getcache(A)
    tmp = FINUFFT.cufinufft_exec(A.plan.adjoint, b, bc)
    out .= real.(tmp)
    return nothing
end

function VLBISkyModels._adjjlnuftadd!(out, A::FINUFFTPlan{P}, b::AbstractArray{<:Complex}) where {P <: GPUPlan}
    # bc = getcache(A)
    tmp = FINUFFT.cufinufft_exec(A.plan.adjoint, b)
    out .+= real.(tmp)
    return nothing
end



end