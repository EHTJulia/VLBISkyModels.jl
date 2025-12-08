module VLBISkyModelsFINUFFT
using VLBISkyModels
using ComradeBase: AbstractRectiGrid, UnstructuredDomain, domainpoints
using VLBISkyModels: FINUFFTAlg, FINUFFTPlan, AdjointFINPlan, _nuft!, _jlnuft!
using EnzymeCore: EnzymeRules
using EnzymeCore

using FINUFFT

function VLBISkyModels.plan_nuft_spatial(
        alg::FINUFFTAlg, imgdomain::AbstractRectiGrid,
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
    fftw = Cint(alg.fftflags) # Convert our plans to correct numeric type
    pfor = FINUFFT.finufft_makeplan(
        2, collect(size(imgdomain)[1:2]), +1, 1, alg.reltol;
        nthreads = alg.threads,
        fftw = fftw, dtype = T, upsampfac = 2.0
    )
    FINUFFT.finufft_setpts!(pfor, u, v)
    # Now we construct the adjoint plan as well
    padj = FINUFFT.finufft_makeplan(
        1, collect(size(imgdomain)[1:2]), -1, 1, alg.reltol;
        nthreads = alg.threads,
        fftw = fftw, dtype = T, upsampfac = 2.0
    )
    FINUFFT.finufft_setpts!(padj, u, v)
    ccache = similar(U, complex(T), size(imgdomain)[1:2])
    p = FINUFFTPlan(size(u), size(imgdomain)[1:2], pfor, padj, ccache)

    finalizer(
        p -> begin
            # #println("Run FINUFFT finalizer")
            FINUFFT.finufft_destroy!(p.forward)
            FINUFFT.finufft_destroy!(p.adjoint)
        end, p
    )
    return p
end

function VLBISkyModels.make_phases(
        ::FINUFFTAlg, imgdomain::AbstractRectiGrid,
        visdomain::UnstructuredDomain
    )
    # These use the same phases to just use the same code since it doesn't depend on NFFTAlg at all.
    return VLBISkyModels.make_phases(NFFTAlg(), imgdomain, visdomain)
end

@noinline function getcache(A::FINUFFTPlan)
    return A.ccache
end

@noinline function getcache(A::AdjointFINPlan)
    return A.plan.ccache
end

EnzymeRules.inactive(::typeof(getcache), args...) = nothing
EnzymeRules.inactive_type(::Type{<:FINUFFT.finufft_plan}) = true

function VLBISkyModels._jlnuft!(out, A::FINUFFTPlan, b::AbstractArray{<:Real})
    bc = getcache(A)
    bc .= b
    FINUFFT.finufft_exec!(A.forward, bc, out)
    return nothing
end

function VLBISkyModels._jlnuft!(out, A::AdjointFINPlan, b::AbstractArray{<:Complex})
    bc = getcache(A)
    FINUFFT.finufft_exec!(A.forward, b, bc)
    out .= real.(bc)
    return nothing
end

function VLBISkyModels._jlnuft_adjointadd!(out, A::FINUFFTPlan, b::AbstractArray{<:Complex})
    bc = getcache(A)
    FINUFFT.finufft_exec!(A.adjoint, b, bc)
    out .+= real.(bc)
    return nothing
end

end
