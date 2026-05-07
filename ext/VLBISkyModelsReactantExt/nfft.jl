"""
    Reactant_NFFTPlan

NFFT plan for Reactant/XLA. Nodes are bin-sorted on the host once for cache
locality; per-node window stencils and integer offsets are precomputed in a
threaded host loop and shipped to the device. The forward/adjoint kernels are
single-shot fused gather/scatter ops — the linear-index table and window-product
table are synthesized inside the JIT'd region from `stencilStarts` (D × J Int32)
and `windowTensor` (M × D × J Float64), so device constants stay small and the
HLO graph is independent of node count.
"""
mutable struct Reactant_NFFTPlan{T <: Number, D, M, K, WT, WS, DI, WH, DO, PS, PO, TG, TV} <: AbstractNFFTPlan{T, D, 1}
    N::NTuple{D, Int64}
    NOut::NTuple{1, Int64}
    J::Int64
    k::K
    Ñ::NTuple{D, Int64}
    dims::UnitRange{Int64}
    numStencil::Int64

    # Per-node interpolation metadata, sorted by spatial bin
    stencilStarts::WS    # (D, J), 0-based first stencil index per dim
    windowTensor::WT     # (M, D, J)

    # Deconvolution metadata
    deconvolveIdx::DI
    windowHatInvLUT::WH
    deconvolveOutIdx::DO

    # Bin-sort permutations
    nodePermSortedToOrig::PS
    nodePermOrigToSorted::PO

    # Reusable scratch
    tmpGrid::TG
    tmpVecHat::TV
end

@inline window_width(::Reactant_NFFTPlan{T, D, M}) where {T, D, M} = M

struct AdjointRPlan{P}
    plan::P
end
Base.adjoint(p::Reactant_NFFTPlan) = AdjointRPlan(p)

@inline _index_type_for_size(n::Integer) = n <= typemax(Int32) ? Int32 : Int64

@inline function _choose_bin_width(Ñd::Int, M::Int)
    base = max(M, 8)
    return min(128, max(base, Ñd ÷ 16))
end

@inline function _evalpoly_col(x::T, P::AbstractMatrix{T}, l::Int) where {T}
    acc = zero(T)
    @inbounds for g in size(P, 1):-1:1
        acc = muladd(acc, x, P[g, l])
    end
    return acc
end

function _build_bin_offsets_counts(nodeBinIds::Vector{Int64}, nBins::Int)
    counts = zeros(Int64, nBins)
    @inbounds for b in nodeBinIds
        counts[b] += 1
    end

    offsets = Vector{Int64}(undef, nBins + 1)
    offsets[1] = 1
    @inbounds for b in 1:nBins
        offsets[b + 1] = offsets[b] + counts[b]
    end

    return offsets, counts
end

"""
Create NFFT plan for Reactant arrays.
"""
function AbstractNFFTs.plan_nfft(
        ::NFFT.NFFTBackend,
        ::Type{<:Reactant.RArray},
        k::AbstractMatrix{T},
        N::NTuple{D, Int},
        rest...;
        timing::Union{Nothing, AbstractNFFTs.TimingStats} = nothing,
        kargs...,
    ) where {T, D}
    t = @elapsed begin
        p = Reactant_NFFTPlan(k, N, rest...; kargs...)
    end
    if timing !== nothing
        timing.pre = t
    end
    return p
end

function Reactant_NFFTPlan(
        k::AbstractMatrix{T},
        N::NTuple{D, Int};
        dims::Union{Integer, UnitRange{Int64}} = 1:D,
        fftflags = nothing,
        kwargs...,
    ) where {T, D}
    D > 3 && throw(ArgumentError("Reactant NFFT only supports D ≤ 3, got D=$D"))

    NFFT.checkNodes(k)

    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(
        k,
        N,
        dims;
        precompute = TENSOR,
        storeDeconvolutionIdx = true,
        blocking = false,
        kwargs...,
    )

    if length(dims_) != D
        error("Reactant NFFT does not support directional transforms yet!")
    end

    m = params.m
    M = 2m
    numStencil = M^D

    gridIndexType = _index_type_for_size(prod(Ñ))

    stencilStarts, windowTensor,
        nodePermSortedToOrig, nodePermOrigToSorted = precompute_window_reactant(
        k, Ñ, params, gridIndexType,
    )

    deconvolveIdx, windowHatInvLUT = precompute_deconvolve_reactant(N, Ñ, params, gridIndexType)
    deconvolveOutIdx = Reactant.to_rarray(collect(Int64, 1:length(windowHatInvLUT)))

    k_r = Reactant.to_rarray(collect(k))
    tmpGrid_r = Reactant.to_rarray(zeros(Complex{T}, Ñ))
    tmpVecHat_r = Reactant.to_rarray(zeros(Complex{T}, length(windowHatInvLUT)))

    return Reactant_NFFTPlan{
        T, D, M,
        typeof(k_r),
        typeof(windowTensor),
        typeof(stencilStarts),
        typeof(deconvolveIdx),
        typeof(windowHatInvLUT),
        typeof(deconvolveOutIdx),
        typeof(nodePermSortedToOrig),
        typeof(nodePermOrigToSorted),
        typeof(tmpGrid_r),
        typeof(tmpVecHat_r),
    }(
        N, NOut, J, k_r, Ñ, dims_, numStencil,
        stencilStarts, windowTensor,
        deconvolveIdx, windowHatInvLUT, deconvolveOutIdx,
        nodePermSortedToOrig, nodePermOrigToSorted,
        tmpGrid_r, tmpVecHat_r,
    )
end

AbstractNFFTs.size_in(p::Reactant_NFFTPlan) = p.N
AbstractNFFTs.size_out(p::Reactant_NFFTPlan) = p.NOut
AbstractNFFTs.size_out(p::AdjointRPlan) = AbstractNFFTs.size_in(p.plan)
AbstractNFFTs.size_in(p::AdjointRPlan) = AbstractNFFTs.size_out(p.plan)

function Base.show(io::IO, p::Reactant_NFFTPlan{T, D, M}) where {T, D, M}
    return print(io, "Reactant_NFFTPlan with $(p.J) nodes for $(D)D input $(p.N), window=$(M)")
end

#############################
# Reactant tracing support
#############################

@inline function _make_tracer_optional(seen, prev, path, mode; kwargs...)
    prev === nothing && return nothing
    return Reactant.make_tracer(seen, prev, path, mode; kwargs...)
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
        @nospecialize(RT::Type{<:Reactant_NFFTPlan{T, D, M, K, WT, WS, DI, WH, DO, PS, PO, TG, TV}}),
        seen,
        mode::Reactant.TraceMode,
        @nospecialize(track_numbers::Type),
        @nospecialize(ndevices),
        @nospecialize(runtime),
    ) where {T, D, M, K, WT, WS, DI, WH, DO, PS, PO, TG, TV}
    K2 = Reactant.traced_type_inner(K, seen, mode, track_numbers, ndevices, runtime)
    WT2 = Reactant.traced_type_inner(WT, seen, mode, track_numbers, ndevices, runtime)
    WS2 = Reactant.traced_type_inner(WS, seen, mode, track_numbers, ndevices, runtime)
    DI2 = Reactant.traced_type_inner(DI, seen, mode, track_numbers, ndevices, runtime)
    WH2 = Reactant.traced_type_inner(WH, seen, mode, track_numbers, ndevices, runtime)
    DO2 = Reactant.traced_type_inner(DO, seen, mode, track_numbers, ndevices, runtime)
    PS2 = Reactant.traced_type_inner(PS, seen, mode, track_numbers, ndevices, runtime)
    PO2 = Reactant.traced_type_inner(PO, seen, mode, track_numbers, ndevices, runtime)
    TG2 = Reactant.traced_type_inner(TG, seen, mode, track_numbers, ndevices, runtime)
    TV2 = Reactant.traced_type_inner(TV, seen, mode, track_numbers, ndevices, runtime)

    return Reactant_NFFTPlan{T, D, M, K2, WT2, WS2, DI2, WH2, DO2, PS2, PO2, TG2, TV2}
end

Base.@nospecializeinfer function Reactant.make_tracer(
        seen,
        prev::Reactant_NFFTPlan{T, D, M},
        @nospecialize(path),
        mode;
        kwargs...,
    ) where {T, D, M}
    if mode == Reactant.TracedToTypes
        push!(path, Core.Typeof(prev))
        return nothing
    end

    if haskey(seen, prev)
        return seen[prev]
    end

    k_traced = _make_tracer_optional(seen, prev.k, (path..., :k), mode; kwargs...)
    wt_traced = _make_tracer_optional(seen, prev.windowTensor, (path..., :windowTensor), mode; kwargs...)
    ws_traced = _make_tracer_optional(seen, prev.stencilStarts, (path..., :stencilStarts), mode; kwargs...)
    di_traced = _make_tracer_optional(seen, prev.deconvolveIdx, (path..., :deconvolveIdx), mode; kwargs...)
    wh_traced = _make_tracer_optional(seen, prev.windowHatInvLUT, (path..., :windowHatInvLUT), mode; kwargs...)
    do_traced = _make_tracer_optional(seen, prev.deconvolveOutIdx, (path..., :deconvolveOutIdx), mode; kwargs...)
    ps_traced = _make_tracer_optional(seen, prev.nodePermSortedToOrig, (path..., :nodePermSortedToOrig), mode; kwargs...)
    po_traced = _make_tracer_optional(seen, prev.nodePermOrigToSorted, (path..., :nodePermOrigToSorted), mode; kwargs...)
    tg_traced = _make_tracer_optional(seen, prev.tmpGrid, (path..., :tmpGrid), mode; kwargs...)
    tv_traced = _make_tracer_optional(seen, prev.tmpVecHat, (path..., :tmpVecHat), mode; kwargs...)

    result = Reactant_NFFTPlan{
        T, D, M,
        typeof(k_traced),
        typeof(wt_traced),
        typeof(ws_traced),
        typeof(di_traced),
        typeof(wh_traced),
        typeof(do_traced),
        typeof(ps_traced),
        typeof(po_traced),
        typeof(tg_traced),
        typeof(tv_traced),
    }(
        prev.N, prev.NOut, prev.J, k_traced, prev.Ñ, prev.dims, prev.numStencil,
        ws_traced, wt_traced,
        di_traced, wh_traced, do_traced,
        ps_traced, po_traced,
        tg_traced, tv_traced,
    )

    seen[prev] = result
    return result
end

#############################
# Precomputation
#############################

"""
Precompute per-node window stencils, integer-offset starts, and bin-sort
permutations. The per-node loop is multithreaded; each `j` writes to a unique
sorted index `s = permOrigToSorted[j]`, so the writes are race-free.
"""
function precompute_window_reactant(
        k::AbstractMatrix{T},
        Ñ::NTuple{D, Int},
        params,
        ::Type{GI},
    ) where {T, D, GI <: Integer}
    m = params.m
    σ = params.σ
    J = size(k, 2)
    M = 2m

    win, _ = NFFT.getWindow(params.window)
    P = NFFT.precomputePolyInterp(win, m, σ, T)

    kShifted = collect(k)
    NFFT.shiftNodes!(kShifted)

    # Coarse spatial binning for cache locality
    binWidths = ntuple(d -> _choose_bin_width(Ñ[d], M), D)
    binShape = ntuple(d -> cld(Ñ[d], binWidths[d]), D)
    binStrides = ntuple(d -> d == 1 ? 1 : prod(binShape[1:(d - 1)]), D)
    nBins = prod(binShape)
    nodeBinIds = Vector{Int64}(undef, J)

    @inbounds for j in 1:J
        binId0 = 0
        for d in 1:D
            kscale = kShifted[d, j] * Ñ[d]
            center0 = mod(unsafe_trunc(Int, kscale), Ñ[d])
            binCoord = center0 ÷ binWidths[d]
            binId0 += binCoord * binStrides[d]
        end
        nodeBinIds[j] = binId0 + 1
    end

    binOffsets, _ = _build_bin_offsets_counts(nodeBinIds, nBins)

    # Stable counting sort by bin
    NI = _index_type_for_size(J)
    permSortedToOrig = Vector{NI}(undef, J)
    permOrigToSorted = Vector{NI}(undef, J)
    cursor = copy(binOffsets[1:(end - 1)])

    @inbounds for j in 1:J
        b = nodeBinIds[j]
        s = cursor[b]
        cursor[b] = s + 1
        permSortedToOrig[s] = NI(j)
        permOrigToSorted[j] = NI(s)
    end

    # Per-node window stencils, parallelized across nodes
    stencilStarts = Matrix{GI}(undef, D, J)
    windowTensorSorted = Array{T, 3}(undef, M, D, J)

    Threads.@threads for j in 1:J
        s = Int(permOrigToSorted[j])
        @inbounds for d in 1:D
            kscale = kShifted[d, j] * Ñ[d]
            off = unsafe_trunc(Int, kscale) - m + 1
            kfrac = kscale - off - m + 1 - T(0.5)

            stencilStarts[d, s] = GI(mod(off, Ñ[d]))
            for l in 1:M
                windowTensorSorted[l, d, s] = _evalpoly_col(kfrac, P, l)
            end
        end
    end

    return (
        Reactant.to_rarray(stencilStarts),
        Reactant.to_rarray(windowTensorSorted),
        Reactant.to_rarray(permSortedToOrig),
        Reactant.to_rarray(permOrigToSorted),
    )
end

"""
Precompute deconvolution lookup table and indices.
"""
function precompute_deconvolve_reactant(
        N::NTuple{D, Int},
        Ñ::NTuple{D, Int},
        params,
        ::Type{I},
    ) where {D, I <: Integer}
    T = eltype(params.σ)
    m = params.m
    σ = params.σ

    _, win_hat = NFFT.getWindow(params.window)

    windowHatInvLUT_sep = Vector{Vector{T}}(undef, D)
    NFFT.precomputeWindowHatInvLUT(windowHatInvLUT_sep, win_hat, N, Ñ, m, σ, T)

    windowHatInvLUT, deconvolveIdx = NFFT.precompWindowHatInvLUT(params, N, Ñ, windowHatInvLUT_sep)

    return Reactant.to_rarray(I.(deconvolveIdx)), Reactant.to_rarray(real.(windowHatInvLUT))
end

#############################
# In-kernel synthesis of linearIndices and windowProducts
#############################

@inline function _linear_indices_full(p::Reactant_NFFTPlan{T, 1, M}) where {T, M}
    GI = eltype(p.stencilStarts)
    J = p.J
    offs = reshape(GI.(0:(M - 1)), M, 1)
    n1 = GI(p.Ñ[1])
    s1 = p.stencilStarts[1, :]
    return mod.(reshape(s1, 1, J) .+ offs, n1) .+ one(GI)
end

@inline function _linear_indices_full(p::Reactant_NFFTPlan{T, 2, M}) where {T, M}
    GI = eltype(p.stencilStarts)
    J = p.J
    offs = reshape(GI.(0:(M - 1)), M, 1)
    n1 = GI(p.Ñ[1])
    n2 = GI(p.Ñ[2])
    s1 = p.stencilStarts[1, :]
    s2 = p.stencilStarts[2, :]
    z1 = mod.(reshape(s1, 1, J) .+ offs, n1)
    z2 = mod.(reshape(s2, 1, J) .+ offs, n2)
    return reshape(
        reshape(z1, M, 1, J) .+ n1 .* reshape(z2, 1, M, J),
        M * M, J,
    ) .+ one(GI)
end

@inline function _linear_indices_full(p::Reactant_NFFTPlan{T, 3, M}) where {T, M}
    GI = eltype(p.stencilStarts)
    J = p.J
    offs = reshape(GI.(0:(M - 1)), M, 1)
    n1 = GI(p.Ñ[1])
    n2 = GI(p.Ñ[2])
    n3 = GI(p.Ñ[3])
    s1 = p.stencilStarts[1, :]
    s2 = p.stencilStarts[2, :]
    s3 = p.stencilStarts[3, :]
    z1 = mod.(reshape(s1, 1, J) .+ offs, n1)
    z2 = mod.(reshape(s2, 1, J) .+ offs, n2)
    z3 = mod.(reshape(s3, 1, J) .+ offs, n3)
    n12 = n1 * n2
    return reshape(
        reshape(z1, M, 1, 1, J) .+ n1 .* reshape(z2, 1, M, 1, J) .+
            n12 .* reshape(z3, 1, 1, M, J),
        M * M * M, J,
    ) .+ one(GI)
end

@inline function _window_products_full(p::Reactant_NFFTPlan{T, 1, M}) where {T, M}
    return p.windowTensor[:, 1, :]
end

@inline function _window_products_full(p::Reactant_NFFTPlan{T, 2, M}) where {T, M}
    J = p.J
    w1 = p.windowTensor[:, 1, :]
    w2 = p.windowTensor[:, 2, :]
    return reshape(
        reshape(w1, M, 1, J) .* reshape(w2, 1, M, J),
        M * M, J,
    )
end

@inline function _window_products_full(p::Reactant_NFFTPlan{T, 3, M}) where {T, M}
    J = p.J
    w1 = p.windowTensor[:, 1, :]
    w2 = p.windowTensor[:, 2, :]
    w3 = p.windowTensor[:, 3, :]
    return reshape(
        reshape(w1, M, 1, 1, J) .* reshape(w2, 1, M, 1, J) .* reshape(w3, 1, 1, M, J),
        M * M * M, J,
    )
end

#############################
# Convolution (forward: g -> fHat)
#############################

function AbstractNFFTs.convolve!(
        p::Reactant_NFFTPlan{T, D},
        g::AbstractArray{<:Number, D},
        fHat::AbstractVector{<:Number},
    ) where {T, D}
    gFlat = vec(g)

    li = _linear_indices_full(p)
    wp = _window_products_full(p)
    sorted = vec(sum(gFlat[li] .* wp, dims = 1))
    copyto!(fHat, sorted[p.nodePermOrigToSorted])
    return fHat
end

#############################
# Convolution transpose (adjoint: fHat -> g)
#############################

function AbstractNFFTs.convolve_transpose!(
        p::Reactant_NFFTPlan{T, D},
        fHat::AbstractVector{<:Number},
        g::AbstractArray{<:Number, D},
    ) where {T, D}
    g .= zero(eltype(g))

    sortedFHat = fHat[p.nodePermSortedToOrig]

    li = _linear_indices_full(p)
    wp = _window_products_full(p)
    updates = vec(wp .* reshape(sortedFHat, 1, :))

    gFlat = Reactant.promote_to(Reactant.TracedRArray, vec(g))
    idx = Reactant.promote_to(Reactant.TracedRArray, reshape(Int64.(vec(li)), 1, :))
    upd = Reactant.promote_to(Reactant.TracedRArray, updates)

    gFlat = Reactant.Ops.scatter(
        +,
        [gFlat],
        idx,
        [upd];
        update_window_dims = Int64[],
        inserted_window_dims = Int64[1],
        input_batching_dims = Int64[],
        scatter_indices_batching_dims = Int64[],
        scatter_dims_to_operand_dims = Int64[1],
        index_vector_dim = 1,
        unique_indices = false,
        indices_are_sorted = false,
    )[1]

    copyto!(g, reshape(gFlat, size(g)))
    return g
end

#############################
# Deconvolution (f -> g)
#############################

function AbstractNFFTs.deconvolve!(
        p::Reactant_NFFTPlan{T, D},
        f::AbstractArray{<:Number, D},
        g::AbstractArray{<:Number, D},
    ) where {T, D}
    tmp = p.tmpVecHat

    copyto!(tmp, vec(f)[p.deconvolveOutIdx])
    tmp .*= p.windowHatInvLUT
    g[p.deconvolveIdx] = tmp
    return nothing
end

#############################
# Deconvolution transpose (g -> f)
#############################

function AbstractNFFTs.deconvolve_transpose!(
        p::Reactant_NFFTPlan{T, D},
        g::AbstractArray{<:Number, D},
        f::AbstractArray{<:Number, D},
    ) where {T, D}
    tmp = p.tmpVecHat

    copyto!(tmp, g[p.deconvolveIdx])
    tmp .*= p.windowHatInvLUT

    f .= zero(eltype(f))
    fFlat = Reactant.promote_to(Reactant.TracedRArray, vec(f))
    idx = Reactant.promote_to(Reactant.TracedRArray, reshape(p.deconvolveOutIdx, 1, :))
    upd = Reactant.promote_to(Reactant.TracedRArray, tmp)

    fFlat = Reactant.Ops.scatter(
        +,
        [fFlat],
        idx,
        [upd];
        update_window_dims = Int64[],
        inserted_window_dims = Int64[1],
        input_batching_dims = Int64[],
        scatter_indices_batching_dims = Int64[],
        scatter_dims_to_operand_dims = Int64[1],
        index_vector_dim = 1,
        unique_indices = true,
        indices_are_sorted = true,
    )[1]

    copyto!(f, reshape(fFlat, size(f)))
    return nothing
end

#############################
# mul! for forward and adjoint
#############################

function LinearAlgebra.mul!(
        fHat::Reactant.AnyTracedRVector,
        p::Reactant_NFFTPlan{T, D},
        f::Reactant.AnyTracedRArray;
        verbose = false,
        timing::Union{Nothing, AbstractNFFTs.TimingStats} = nothing,
    ) where {T, D}
    NFFT.consistencyCheck(p, f, fHat)

    g = p.tmpGrid
    g .= zero(eltype(g))

    t1 = @elapsed NFFT.deconvolve!(p, f, g)
    t2 = @elapsed fft!(g)
    t3 = @elapsed NFFT.convolve!(p, g, fHat)

    if verbose
        @info "Timing: deconv=$t1 fft=$t2 conv=$t3"
    end
    if timing !== nothing
        timing.conv = t3
        timing.fft = t2
        timing.deconv = t1
    end

    return fHat
end

function LinearAlgebra.mul!(
        f::Reactant.AnyTracedRArray,
        pl::AdjointRPlan,
        fHat::Reactant.AnyTracedRVector;
        verbose = false,
        timing::Union{Nothing, AbstractNFFTs.TimingStats} = nothing,
    )
    p = pl.plan
    NFFT.consistencyCheck(p, f, fHat)

    g = p.tmpGrid
    g .= zero(eltype(g))

    t1 = @elapsed NFFT.convolve_transpose!(p, fHat, g)
    t2 = @elapsed bfft!(g)
    t3 = @elapsed NFFT.deconvolve_transpose!(p, g, f)

    if verbose
        @info "Timing: conv=$t1 fft=$t2 deconv=$t3"
    end
    if timing !== nothing
        timing.conv_adjoint = t1
        timing.fft_adjoint = t2
        timing.deconv_adjoint = t3
    end

    return f
end

#############################
# * operator
#############################

function Base.:*(p::Reactant_NFFTPlan{T, D}, f::Reactant.AnyTracedRArray; kargs...) where {T, D}
    fHat = similar(f, complex(eltype(f)), size_out(p))
    mul!(fHat, p, f; kargs...)
    return fHat
end

function Base.:*(pl::AdjointRPlan, fHat::Reactant.AnyTracedRVector; kargs...)
    f = similar(fHat, complex(eltype(fHat)), size_out(pl))
    mul!(f, pl, fHat; kargs...)
    return f
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
