module VLBISkyModelsReactantExt

using VLBISkyModels
using AbstractFFTs
using Reactant
using NFFT
using NFFT: AbstractNFFTs
using VLBISkyModels: ReactantAlg
using LinearAlgebra


struct ReactantNFFTPlan{T, D, K <: AbstractArray, arrTc, vecI, vecII, FP, BP, INV, SM} <:
    AbstractNFFTPlan{T, D, 1}
    N::NTuple{D, Int}
    NOut::NTuple{1, Int}
    J::Int
    k::K
    Ñ::NTuple{D, Int}
    dims::UnitRange{Int}
    forwardFFT::FP
    backwardFFT::BP
    tmpVec::arrTc
    tmpVecHat::arrTc
    deconvolveIdx::vecI
    windowLinInterp::vecII
    windowHatInvLUT::INV
    B::SM
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
    return ReactantNFFTPlan(uv2, size(imgdomain))
end

function VLBISkyModels.make_phases(
        ::ReactantAlg,
        imgdomain::ComradeBase.AbstractRectiGrid,
        visdomain::ComradeBase.UnstructuredDomain,
    )
    return Reactant.to_rarray(VLBISkyModels.make_phases(NFFTAlg(), imgdomain, visdomain))
end

function VLBISkyModels._jlnuft!(out, A::ReactantNFFTPlan, inp::Reactant.AnyTracedRArray)
    LinearAlgebra.mul!(out, A, inp)
    return nothing
end


Base.adjoint(p::ReactantNFFTPlan) = p


function AbstractNFFTs.plan_nfft(
        arr::Type{<:Reactant.AnyTracedRArray},
        k::AbstractMatrix,
        N::NTuple{D, Int},
        rest...;
        kargs...,
    ) where {D}
    p = ReactantNFFTPlan(arr, k, N; kargs...)
    return p
end

function ReactantNFFTPlan(
        k::AbstractArray{T}, N::NTuple{D, Int}; fftflags = nothing, kwargs...
    ) where {T, D}


    dims = 1:D
    CT = complex(T)
    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(k, N, dims; kwargs...)

    # Get the correct type
    FP = @jit plan_fft!(zeros(ComplexF64, 2, 2))
    BP = @jit plan_bfft!(zeros(ComplexF64, 2, 2))

    params.storeDeconvolutionIdx = true # GPU_NFFT only works this way
    params.precompute = NFFT.FULL # GPU_NFFT only works this way

    windowLinInterp, windowPolyInterp, windowHatInvLUT, deconvolveIdx, B = NFFT.precomputation(
        k, N[dims_], Ñ[dims_], params
    )

    U = params.storeDeconvolutionIdx ? N : ntuple(d -> 0, Val(D))

    tmpVec = Reactant.to_rarray(zeros(CT, Ñ))
    tmpVecHat = Reactant.to_rarray(zeros(CT, U))
    deconvIdx = Reactant.to_rarray(Int.(deconvolveIdx))
    winHatInvLUT = Reactant.to_rarray(complex(windowHatInvLUT[1]))
    B_ = (Reactant.to_rarray(complex.(Array(B))))

    return ReactantNFFTPlan{
        T,
        D,
        typeof(k),
        typeof(tmpVec),
        typeof(deconvIdx),
        typeof(windowLinInterp),
        typeof(FP),
        typeof(BP),
        typeof(winHatInvLUT),
        typeof(B_),
    }(
        N,
        NOut,
        J,
        k,
        Ñ,
        dims_,
        FP,
        BP,
        tmpVec,
        tmpVecHat,
        deconvIdx,
        windowLinInterp,
        winHatInvLUT,
        B_,
    )
end

AbstractNFFTs.size_in(p::ReactantNFFTPlan) = p.N
AbstractNFFTs.size_out(p::ReactantNFFTPlan) = p.NOut

function AbstractNFFTs.convolve!(
        p::ReactantNFFTPlan{T, D}, g::Reactant.AnyTracedRArray, fHat::Reactant.AnyTracedRArray
    ) where {D, T}
    mul!(fHat, transpose(p.B), vec(g))
    return nothing
end

function AbstractNFFTs.convolve_transpose!(
        p::ReactantNFFTPlan{T, D}, fHat::Reactant.AnyTracedRArray, g::Reactant.AnyTracedRArray
    ) where {D, T}
    mul!(vec(g), p.B, fHat)
    return nothing
end

function Base.:*(p::ReactantNFFTPlan{T}, f::Reactant.AnyTracedRArray; kargs...) where {T}
    fHat = similar(f, complex(T), size_out(p))
    mul!(fHat, p, f; kargs...)
    return fHat
end

function AbstractNFFTs.deconvolve!(
        p::ReactantNFFTPlan{T, D}, f::AbstractArray, g::AbstractArray
    ) where {D, T}
    tmp = f .* reshape(p.windowHatInvLUT, size(f))
    @allowscalar g[p.deconvolveIdx] = reshape(tmp, :)
    return nothing
end

"""  in-place NFFT on the GPU"""
function LinearAlgebra.mul!(
        fHat::Reactant.AnyTracedRArray,
        p::ReactantNFFTPlan{T, D},
        f::Reactant.AnyTracedRArray;
        verbose = false,
        timing::Union{Nothing, TimingStats} = nothing,
    ) where {T, D}
    NFFT.consistencyCheck(p, f, fHat)

    fill!(p.tmpVec, zero(Complex{T}))
    t1 = @elapsed @inbounds deconvolve!(p, f, p.tmpVec)
    fHat .= p.tmpVec[1:length(fHat)]
    p.forwardFFT * p.tmpVec
    return t3 = @elapsed @inbounds NFFT.convolve!(p, p.tmpVec, fHat)
end

function NFFT.nfft(k::AbstractMatrix, f::Reactant.AnyTracedRArray, args...; kwargs...)
    p = ReactantNFFTPlan(typeof(f), k, size(f))
    return p * f
end


end
