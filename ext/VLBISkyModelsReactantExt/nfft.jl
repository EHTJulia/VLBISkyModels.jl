"""
    Reactant_NFFTPlan

NFFT plan for Reactant/XLA with a cuFINUFFT-inspired structure:
1. One-time preprocessing of nodes into coarse spatial bins.
2. Reorder nodes by bin for locality.
3. Build chunk metadata (bin-aligned chunks).
4. Precompute interpolation stencils and kernel weights once.
5. Execute forward/adjoint in chunked gather/scatter phases.
"""
mutable struct Reactant_NFFTPlan{T<:Number,D,M,K,WI,WF,WP,DI,WH,DO,PS,PO,SI,SS,SP,TG,TV,TF} <: AbstractNFFTPlan{T,D,1}
    N::NTuple{D,Int64}
    NOut::NTuple{1,Int64}
    J::Int64
    k::K
    Ñ::NTuple{D,Int64}
    dims::UnitRange{Int64}
    numStencil::Int64

    # Interpolation metadata (node-sorted by bins)
    linearIndices::WI           # (M^D, J) or `nothing` (implicit index mode)
    stencilStarts::SI           # (D, J), 0-based first stencil index per dim
    windowTensor::WF            # (M, D, J)
    windowProducts::WP          # (M^D, J)

    # Deconvolution metadata
    deconvolveIdx::DI
    windowHatInvLUT::WH
    deconvolveOutIdx::DO

    # Node order permutations
    nodePermSortedToOrig::PS
    nodePermOrigToSorted::PO

    # Chunk-local sorted scatter metadata
    scatterIndicesSorted::SS    # concatenated sorted indices per chunk
    scatterPermLocal::SP        # local permutation per chunk

    # Structural metadata (host-side constants)
    chunkOffsets::Vector{Int64}       # node-space offsets, length nchunks+1
    chunkCounts::Vector{Int32}        # nodes per chunk
    scatterChunkOffsets::Vector{Int64} # stencil-update offsets, length nchunks+1
    binOffsets::Vector{Int64}
    binCounts::Vector{Int64}
    forwardGlobal::Bool
    adjointGlobal::Bool

    # Reusable scratch
    tmpGrid::TG
    tmpVecHat::TV
    tmpFHatSorted::TF
end

@inline window_width(::Reactant_NFFTPlan{T,D,M}) where {T,D,M} = M

struct AdjointRPlan{P}
    plan::P
end
Base.adjoint(p::Reactant_NFFTPlan) = AdjointRPlan(p)

@inline _index_type_for_size(n::Integer) = n <= typemax(Int32) ? Int32 : Int64

@inline function _choose_bin_width(Ñd::Int, M::Int)
    base = max(M, 8)
    return min(128, max(base, Ñd ÷ 16))
end

@inline function _choose_chunk_nodes(numStencil::Int, J::Int)
    J <= 0 && return 1
    # Balance graph size and temporary volume while forcing multi-chunk locality.
    by_updates = max(128, fld(3_000_000, max(numStencil, 1)))
    min_nodes = max(64, cld(J, 32)) # <= 32 chunks
    max_nodes = max(min_nodes, cld(J, 2)) # >= 2 chunks for larger J
    n = clamp(by_updates, min_nodes, max_nodes)
    n = min(J, max(1, ((n + 127) ÷ 128) * 128))
    return n
end

@inline function _core_index_bytes(::Type{GI}, numStencil::Int, J::Int) where {GI<:Integer}
    return Int64(sizeof(GI)) * Int64(numStencil) * Int64(J)
end

@inline function _use_implicit_indices(::Type{GI}, numStencil::Int, J::Int) where {GI<:Integer}
    # Dense (M^D, J) index metadata quickly dominates GPU memory; switch to
    # on-the-fly index synthesis once this table exceeds ~1.5 GiB.
    return _core_index_bytes(GI, numStencil, J) > 1536 * 1024^2
end

@inline function _use_precomputed_window_products(::Type{T}, numStencil::Int, J::Int) where {T}
    # Keep dense window product table when it is not too large; this is a major
    # runtime speedup for forward/adjoint gathers.
    bytes = Int64(sizeof(T)) * Int64(numStencil) * Int64(J)
    return bytes <= 2048 * 1024^2
end

@inline function _use_sorted_scatter_metadata(numStencil::Int, J::Int)
    # Sorted scatter metadata is useful but optional; cap it more aggressively.
    bytes = Int64(sizeof(Int64)) * 2 * Int64(numStencil) * Int64(J)
    return bytes <= 1536 * 1024^2
end

@inline function _choose_runtime_modes(
    p::Type{T},
    numStencil::Int,
    J::Int,
    hasDenseIndices::Bool,
    hasDenseWindowProducts::Bool,
    hasSortedScatter::Bool,
    nchunks::Int,
) where {T}
    n = Int64(numStencil) * Int64(J)
    bytesComplex = Int64(sizeof(Complex{T})) * n
    bytesWeight = Int64(sizeof(T)) * n
    bytesIdx64 = Int64(sizeof(Int64)) * n

    # Forward global path builds gathered .* weights and reduces.
    peakForward = bytesComplex + bytesWeight
    forwardGlobal = hasDenseIndices && peakForward <= 2 * 1024^3 && (nchunks <= 16 || n <= 40_000_000)

    # Adjoint global path does one large scatter of weighted updates.
    peakAdjoint = bytesComplex + bytesWeight + bytesIdx64
    adjointGlobal = hasDenseIndices && hasDenseWindowProducts &&
                    peakAdjoint <= 1536 * 1024^2 &&
                    (nchunks <= 8 || n <= 20_000_000)

    # Prefer chunked adjoint when sorted scatter metadata is available.
    hasSortedScatter && (adjointGlobal = false)

    return forwardGlobal, adjointGlobal
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

function _build_chunk_offsets(binOffsets::Vector{Int64}, binCounts::Vector{Int64}, J::Int, numStencil::Int)
    targetNodes = _choose_chunk_nodes(numStencil, J)
    maxChunks = 64

    offsets = Int64[1]
    acc = 0

    @inbounds for b in eachindex(binCounts)
        c = binCounts[b]
        c == 0 && continue

        bstart = binOffsets[b]
        if acc > 0 && acc + c > targetNodes && bstart > offsets[end] && (length(offsets) - 1) < (maxChunks - 1)
            push!(offsets, bstart)
            acc = 0
        end
        acc += c
    end

    if offsets[end] != J + 1
        push!(offsets, J + 1)
    end

    counts = Vector{Int32}(undef, length(offsets) - 1)
    @inbounds for i in eachindex(counts)
        counts[i] = Int32(offsets[i + 1] - offsets[i])
    end

    return offsets, counts
end

@inline function _evalpoly_col(x::T, P::AbstractMatrix{T}, l::Int) where {T}
    acc = zero(T)
    @inbounds for g in size(P, 1):-1:1
        acc = muladd(acc, x, P[g, l])
    end
    return acc
end

@inline function _fill_linear_indices!(
    linearSorted::AbstractMatrix{GI},
    idxBuf::NTuple{1,Vector{Int64}},
    strides,
    s::Int,
    M::Int,
) where {GI}
    idx1 = idxBuf[1]
    @inbounds for l1 in 1:M
        linearSorted[l1, s] = GI(1 + idx1[l1] * strides[1])
    end
    return nothing
end

@inline function _fill_linear_indices!(
    linearSorted::AbstractMatrix{GI},
    idxBuf::NTuple{2,Vector{Int64}},
    strides,
    s::Int,
    M::Int,
) where {GI}
    idx1 = idxBuf[1]
    idx2 = idxBuf[2]
    sidx = 1
    @inbounds for l2 in 1:M
        i2 = idx2[l2] * strides[2]
        for l1 in 1:M
            linearSorted[sidx, s] = GI(1 + idx1[l1] * strides[1] + i2)
            sidx += 1
        end
    end
    return nothing
end

@inline function _fill_linear_indices!(
    linearSorted::AbstractMatrix{GI},
    idxBuf::NTuple{3,Vector{Int64}},
    strides,
    s::Int,
    M::Int,
) where {GI}
    idx1 = idxBuf[1]
    idx2 = idxBuf[2]
    idx3 = idxBuf[3]
    sidx = 1
    @inbounds for l3 in 1:M
        i3 = idx3[l3] * strides[3]
        for l2 in 1:M
            i23 = idx2[l2] * strides[2] + i3
            for l1 in 1:M
                linearSorted[sidx, s] = GI(1 + idx1[l1] * strides[1] + i23)
                sidx += 1
            end
        end
    end
    return nothing
end

@inline function _window_products(windowTensor::Array{T,3}, ::Val{1}, M::Int, J::Int) where {T}
    return copy(view(windowTensor, :, 1, :))
end

@inline function _window_products(windowTensor::Array{T,3}, ::Val{2}, M::Int, J::Int) where {T}
    w1 = view(windowTensor, :, 1, :)
    w2 = view(windowTensor, :, 2, :)
    return reshape(
        reshape(w1, M, 1, J) .* reshape(w2, 1, M, J),
        M * M,
        J,
    )
end

@inline function _window_products(windowTensor::Array{T,3}, ::Val{3}, M::Int, J::Int) where {T}
    w1 = view(windowTensor, :, 1, :)
    w2 = view(windowTensor, :, 2, :)
    w3 = view(windowTensor, :, 3, :)
    return reshape(
        reshape(w1, M, 1, 1, J) .* reshape(w2, 1, M, 1, J) .* reshape(w3, 1, 1, M, J),
        M * M * M,
        J,
    )
end

function _build_chunk_scatter_metadata(
    linearSorted::Matrix{GI},
    chunkOffsets::Vector{Int64},
    ::Type{NI},
) where {GI<:Integer,NI<:Integer}
    numStencil, J = size(linearSorted)
    total = numStencil * J

    scatterIdxSorted = Vector{GI}(undef, total)
    scatterPermLocal = Vector{NI}(undef, total)
    scatterChunkOffsets = Vector{Int64}(undef, length(chunkOffsets))
    scatterChunkOffsets[1] = 1

    dst = 1
    flat = vec(linearSorted)

    @inbounds for c in 1:(length(chunkOffsets) - 1)
        jlo = chunkOffsets[c]
        jhi = chunkOffsets[c + 1] - 1

        if jhi < jlo
            scatterChunkOffsets[c + 1] = dst
            continue
        end

        lo = (jlo - 1) * numStencil + 1
        hi = jhi * numStencil
        localIdx = view(flat, lo:hi)

        perm = sortperm(localIdx)
        n = length(perm)

        scatterIdxSorted[dst:(dst + n - 1)] .= localIdx[perm]
        scatterPermLocal[dst:(dst + n - 1)] .= NI.(perm)

        dst += n
        scatterChunkOffsets[c + 1] = dst
    end

    return scatterIdxSorted, scatterPermLocal, scatterChunkOffsets
end

"""
Create NFFT plan for Reactant arrays.
"""
function AbstractNFFTs.plan_nfft(
    ::NFFT.NFFTBackend,
    ::Type{<:Reactant.RArray},
    k::AbstractMatrix{T},
    N::NTuple{D,Int},
    rest...;
    timing::Union{Nothing,AbstractNFFTs.TimingStats}=nothing,
    kargs...,
) where {T,D}
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
    N::NTuple{D,Int};
    dims::Union{Integer,UnitRange{Int64}}=1:D,
    fftflags=nothing,
    kwargs...,
) where {T,D}
    D > 3 && throw(ArgumentError("Reactant NFFT only supports D ≤ 3, got D=$D"))

    NFFT.checkNodes(k)

    # Force TENSOR precompute path for Reactant plan setup.
    params, N, NOut, J, Ñ, dims_ = NFFT.initParams(
        k,
        N,
        dims;
        precompute=TENSOR,
        storeDeconvolutionIdx=true,
        blocking=false,
        kwargs...,
    )

    if length(dims_) != D
        error("Reactant NFFT does not support directional transforms yet!")
    end

    m = params.m
    M = 2m
    numStencil = M^D

    gridIndexType = _index_type_for_size(prod(Ñ))
    nodeIndexType = _index_type_for_size(J)
    implicitIndices = _use_implicit_indices(gridIndexType, numStencil, J)
    storeWindowProducts = _use_precomputed_window_products(T, numStencil, J)
    storeSortedScatter = !implicitIndices && _use_sorted_scatter_metadata(numStencil, J)

    linearIndices, stencilStarts,
    windowTensor, windowProducts,
    nodePermSortedToOrig, nodePermOrigToSorted,
    scatterIndicesSorted, scatterPermLocal,
    chunkOffsets, chunkCounts, scatterChunkOffsets,
    binOffsets, binCounts = precompute_window_and_scatter_reactant(
        k,
        Ñ,
        params,
        gridIndexType,
        nodeIndexType,
        implicitIndices,
        storeWindowProducts,
        storeSortedScatter,
    )
    forwardGlobal, adjointGlobal = _choose_runtime_modes(
        T,
        numStencil,
        J,
        linearIndices !== nothing,
        windowProducts !== nothing,
        scatterIndicesSorted !== nothing,
        length(chunkCounts),
    )

    deconvolveIdx, windowHatInvLUT = precompute_deconvolve_reactant(N, Ñ, params, gridIndexType)
    deconvolveOutIdx = Reactant.to_rarray(collect(Int64, 1:length(windowHatInvLUT)))

    k_r = Reactant.to_rarray(collect(k))
    tmpGrid_r = Reactant.to_rarray(zeros(Complex{T}, Ñ))
    tmpVecHat_r = Reactant.to_rarray(zeros(Complex{T}, length(windowHatInvLUT)))
    tmpFHatSorted_r = Reactant.to_rarray(zeros(Complex{T}, J))

    return Reactant_NFFTPlan{
        T,
        D,
        M,
        typeof(k_r),
        typeof(linearIndices),
        typeof(windowTensor),
        typeof(windowProducts),
        typeof(deconvolveIdx),
        typeof(windowHatInvLUT),
        typeof(deconvolveOutIdx),
        typeof(nodePermSortedToOrig),
        typeof(nodePermOrigToSorted),
        typeof(stencilStarts),
        typeof(scatterIndicesSorted),
        typeof(scatterPermLocal),
        typeof(tmpGrid_r),
        typeof(tmpVecHat_r),
        typeof(tmpFHatSorted_r),
    }(
        N,
        NOut,
        J,
        k_r,
        Ñ,
        dims_,
        numStencil,
        linearIndices,
        stencilStarts,
        windowTensor,
        windowProducts,
        deconvolveIdx,
        windowHatInvLUT,
        deconvolveOutIdx,
        nodePermSortedToOrig,
        nodePermOrigToSorted,
        scatterIndicesSorted,
        scatterPermLocal,
        chunkOffsets,
        chunkCounts,
        scatterChunkOffsets,
        binOffsets,
        binCounts,
        forwardGlobal,
        adjointGlobal,
        tmpGrid_r,
        tmpVecHat_r,
        tmpFHatSorted_r,
    )
end

AbstractNFFTs.size_in(p::Reactant_NFFTPlan) = p.N
AbstractNFFTs.size_out(p::Reactant_NFFTPlan) = p.NOut
AbstractNFFTs.size_out(p::AdjointRPlan) = AbstractNFFTs.size_in(p.plan)
AbstractNFFTs.size_in(p::AdjointRPlan) = AbstractNFFTs.size_out(p.plan)

function Base.show(io::IO, p::Reactant_NFFTPlan{T,D,M}) where {T,D,M}
    nbins = length(p.binCounts)
    nchunks = length(p.chunkCounts)
    compact = p.windowProducts === nothing
    implicit = p.linearIndices === nothing
    print(io, "Reactant_NFFTPlan with $(p.J) nodes for $(D)D input $(p.N), window=$(M), bins=$(nbins), chunks=$(nchunks), compact=$(compact), implicit_indices=$(implicit), modes=(fwd=$(p.forwardGlobal ? "global" : "chunk"), adj=$(p.adjointGlobal ? "global" : "chunk"))")
end

#############################
# Reactant tracing support
#############################

@inline function _make_tracer_optional(seen, prev, path, mode; kwargs...)
    prev === nothing && return nothing
    return Reactant.make_tracer(seen, prev, path, mode; kwargs...)
end

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(RT::Type{<:Reactant_NFFTPlan{T,D,M,K,WI,WF,WP,DI,WH,DO,PS,PO,SI,SS,SP,TG,TV,TF}}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime),
) where {T,D,M,K,WI,WF,WP,DI,WH,DO,PS,PO,SI,SS,SP,TG,TV,TF}
    K2 = Reactant.traced_type_inner(K, seen, mode, track_numbers, ndevices, runtime)
    WI2 = Reactant.traced_type_inner(WI, seen, mode, track_numbers, ndevices, runtime)
    WF2 = Reactant.traced_type_inner(WF, seen, mode, track_numbers, ndevices, runtime)
    WP2 = Reactant.traced_type_inner(WP, seen, mode, track_numbers, ndevices, runtime)
    DI2 = Reactant.traced_type_inner(DI, seen, mode, track_numbers, ndevices, runtime)
    WH2 = Reactant.traced_type_inner(WH, seen, mode, track_numbers, ndevices, runtime)
    DO2 = Reactant.traced_type_inner(DO, seen, mode, track_numbers, ndevices, runtime)
    PS2 = Reactant.traced_type_inner(PS, seen, mode, track_numbers, ndevices, runtime)
    PO2 = Reactant.traced_type_inner(PO, seen, mode, track_numbers, ndevices, runtime)
    SI2 = Reactant.traced_type_inner(SI, seen, mode, track_numbers, ndevices, runtime)
    SS2 = Reactant.traced_type_inner(SS, seen, mode, track_numbers, ndevices, runtime)
    SP2 = Reactant.traced_type_inner(SP, seen, mode, track_numbers, ndevices, runtime)
    TG2 = Reactant.traced_type_inner(TG, seen, mode, track_numbers, ndevices, runtime)
    TV2 = Reactant.traced_type_inner(TV, seen, mode, track_numbers, ndevices, runtime)
    TF2 = Reactant.traced_type_inner(TF, seen, mode, track_numbers, ndevices, runtime)

    return Reactant_NFFTPlan{T,D,M,K2,WI2,WF2,WP2,DI2,WH2,DO2,PS2,PO2,SI2,SS2,SP2,TG2,TV2,TF2}
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    prev::Reactant_NFFTPlan{T,D,M},
    @nospecialize(path),
    mode;
    kwargs...,
) where {T,D,M}
    if mode == Reactant.TracedToTypes
        push!(path, Core.Typeof(prev))
        return nothing
    end

    if haskey(seen, prev)
        return seen[prev]
    end

    k_traced = _make_tracer_optional(seen, prev.k, (path..., :k), mode; kwargs...)
    li_traced = _make_tracer_optional(seen, prev.linearIndices, (path..., :linearIndices), mode; kwargs...)
    lf_traced = _make_tracer_optional(seen, prev.stencilStarts, (path..., :stencilStarts), mode; kwargs...)
    wt_traced = _make_tracer_optional(seen, prev.windowTensor, (path..., :windowTensor), mode; kwargs...)
    wp_traced = _make_tracer_optional(seen, prev.windowProducts, (path..., :windowProducts), mode; kwargs...)
    di_traced = _make_tracer_optional(seen, prev.deconvolveIdx, (path..., :deconvolveIdx), mode; kwargs...)
    wh_traced = _make_tracer_optional(seen, prev.windowHatInvLUT, (path..., :windowHatInvLUT), mode; kwargs...)
    do_traced = _make_tracer_optional(seen, prev.deconvolveOutIdx, (path..., :deconvolveOutIdx), mode; kwargs...)
    ps_traced = _make_tracer_optional(seen, prev.nodePermSortedToOrig, (path..., :nodePermSortedToOrig), mode; kwargs...)
    po_traced = _make_tracer_optional(seen, prev.nodePermOrigToSorted, (path..., :nodePermOrigToSorted), mode; kwargs...)
    ss_traced = _make_tracer_optional(seen, prev.scatterIndicesSorted, (path..., :scatterIndicesSorted), mode; kwargs...)
    sp_traced = _make_tracer_optional(seen, prev.scatterPermLocal, (path..., :scatterPermLocal), mode; kwargs...)
    tg_traced = _make_tracer_optional(seen, prev.tmpGrid, (path..., :tmpGrid), mode; kwargs...)
    tv_traced = _make_tracer_optional(seen, prev.tmpVecHat, (path..., :tmpVecHat), mode; kwargs...)
    tf_traced = _make_tracer_optional(seen, prev.tmpFHatSorted, (path..., :tmpFHatSorted), mode; kwargs...)

    result = Reactant_NFFTPlan{
        T,
        D,
        M,
        typeof(k_traced),
        typeof(li_traced),
        typeof(wt_traced),
        typeof(wp_traced),
        typeof(di_traced),
        typeof(wh_traced),
        typeof(do_traced),
        typeof(ps_traced),
        typeof(po_traced),
        typeof(lf_traced),
        typeof(ss_traced),
        typeof(sp_traced),
        typeof(tg_traced),
        typeof(tv_traced),
        typeof(tf_traced),
    }(
        prev.N,
        prev.NOut,
        prev.J,
        k_traced,
        prev.Ñ,
        prev.dims,
        prev.numStencil,
        li_traced,
        lf_traced,
        wt_traced,
        wp_traced,
        di_traced,
        wh_traced,
        do_traced,
        ps_traced,
        po_traced,
        ss_traced,
        sp_traced,
        prev.chunkOffsets,
        prev.chunkCounts,
        prev.scatterChunkOffsets,
        prev.binOffsets,
        prev.binCounts,
        prev.forwardGlobal,
        prev.adjointGlobal,
        tg_traced,
        tv_traced,
        tf_traced,
    )

    seen[prev] = result
    return result
end

#############################
# Precomputation
#############################

"""
Precompute node-sorted stencil indices, window coefficients, window products,
and chunk-local sorted scatter metadata.
"""
function precompute_window_and_scatter_reactant(
    k::AbstractMatrix{T},
    Ñ::NTuple{D,Int},
    params,
    ::Type{GI},
    ::Type{NI},
    implicitIndices::Bool,
    storeWindowProducts::Bool,
    storeSortedScatter::Bool,
) where {T,D,GI<:Integer,NI<:Integer}
    m = params.m
    σ = params.σ
    J = size(k, 2)
    M = 2m
    numStencil = M^D

    win, _ = NFFT.getWindow(params.window)
    P = NFFT.precomputePolyInterp(win, m, σ, T)

    kShifted = collect(k)
    NFFT.shiftNodes!(kShifted)

    strides = ntuple(d -> d == 1 ? 1 : prod(Ñ[1:(d - 1)]), D)

    # Coarse binning.
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

    binOffsets, binCounts = _build_bin_offsets_counts(nodeBinIds, nBins)

    # Stable counting sort by bin.
    permSortedToOrig = Vector{NI}(undef, J)
    permOrigToSorted = Vector{NI}(undef, J)
    cursor = copy(binOffsets[1:end-1])

    @inbounds for j in 1:J
        b = nodeBinIds[j]
        s = cursor[b]
        cursor[b] = s + 1
        permSortedToOrig[s] = NI(j)
        permOrigToSorted[j] = NI(s)
    end

    # Bin-aligned chunking.
    chunkOffsets, chunkCounts = _build_chunk_offsets(binOffsets, binCounts, J, numStencil)

    # Fill sorted interpolation tables directly (no unsorted temporary matrices).
    linearSorted = implicitIndices ? nothing : Matrix{GI}(undef, numStencil, J)
    stencilStarts = Matrix{GI}(undef, D, J)
    windowTensorSorted = Array{T,3}(undef, M, D, J)

    if D == 1
        idxBuf = (Vector{Int64}(undef, M),)
        @inbounds for j in 1:J
            s = Int(permOrigToSorted[j])

            kscale = kShifted[1, j] * Ñ[1]
            off = unsafe_trunc(Int, kscale) - m + 1
            k_ = kscale - off - m + 1 - T(0.5)

            for l in 1:M
                idxBuf[1][l] = mod(off + l - 1, Ñ[1])
                windowTensorSorted[l, 1, s] = _evalpoly_col(k_, P, l)
            end

            stencilStarts[1, s] = GI(idxBuf[1][1])
            if linearSorted !== nothing
                _fill_linear_indices!(linearSorted, idxBuf, strides, s, M)
            end
        end
    elseif D == 2
        idxBuf = (Vector{Int64}(undef, M), Vector{Int64}(undef, M))
        @inbounds for j in 1:J
            s = Int(permOrigToSorted[j])

            for d in 1:2
                kscale = kShifted[d, j] * Ñ[d]
                off = unsafe_trunc(Int, kscale) - m + 1
                k_ = kscale - off - m + 1 - T(0.5)

                for l in 1:M
                    idxBuf[d][l] = mod(off + l - 1, Ñ[d])
                    windowTensorSorted[l, d, s] = _evalpoly_col(k_, P, l)
                end
            end

            stencilStarts[1, s] = GI(idxBuf[1][1])
            stencilStarts[2, s] = GI(idxBuf[2][1])
            if linearSorted !== nothing
                _fill_linear_indices!(linearSorted, idxBuf, strides, s, M)
            end
        end
    else
        idxBuf = (Vector{Int64}(undef, M), Vector{Int64}(undef, M), Vector{Int64}(undef, M))
        @inbounds for j in 1:J
            s = Int(permOrigToSorted[j])

            for d in 1:3
                kscale = kShifted[d, j] * Ñ[d]
                off = unsafe_trunc(Int, kscale) - m + 1
                k_ = kscale - off - m + 1 - T(0.5)

                for l in 1:M
                    idxBuf[d][l] = mod(off + l - 1, Ñ[d])
                    windowTensorSorted[l, d, s] = _evalpoly_col(k_, P, l)
                end
            end

            stencilStarts[1, s] = GI(idxBuf[1][1])
            stencilStarts[2, s] = GI(idxBuf[2][1])
            stencilStarts[3, s] = GI(idxBuf[3][1])
            if linearSorted !== nothing
                _fill_linear_indices!(linearSorted, idxBuf, strides, s, M)
            end
        end
    end

    windowProducts = storeWindowProducts ? Reactant.to_rarray(_window_products(windowTensorSorted, Val(D), M, J)) : nothing

    scatterIdxSorted = nothing
    scatterPermLocal = nothing
    if storeSortedScatter && linearSorted !== nothing
        sIdx, sPerm, _ = _build_chunk_scatter_metadata(
            linearSorted,
            chunkOffsets,
            NI,
        )
        scatterIdxSorted = Reactant.to_rarray(Int64.(sIdx))
        scatterPermLocal = Reactant.to_rarray(Int64.(sPerm))
    end

    scatterChunkOffsets = Vector{Int64}(undef, length(chunkOffsets))
    @inbounds for c in 1:length(chunkOffsets)
        scatterChunkOffsets[c] = (chunkOffsets[c] - 1) * numStencil + 1
    end

    return (
        linearSorted === nothing ? nothing : Reactant.to_rarray(linearSorted),
        Reactant.to_rarray(stencilStarts),
        Reactant.to_rarray(windowTensorSorted),
        windowProducts,
        Reactant.to_rarray(permSortedToOrig),
        Reactant.to_rarray(permOrigToSorted),
        scatterIdxSorted,
        scatterPermLocal,
        chunkOffsets,
        chunkCounts,
        scatterChunkOffsets,
        binOffsets,
        binCounts,
    )
end

"""
Precompute deconvolution lookup table and indices.
"""
function precompute_deconvolve_reactant(
    N::NTuple{D,Int},
    Ñ::NTuple{D,Int},
    params,
    ::Type{I},
) where {D,I<:Integer}
    T = eltype(params.σ)
    m = params.m
    σ = params.σ

    _, win_hat = NFFT.getWindow(params.window)

    windowHatInvLUT_sep = Vector{Vector{T}}(undef, D)
    NFFT.precomputeWindowHatInvLUT(windowHatInvLUT_sep, win_hat, N, Ñ, m, σ, T)

    windowHatInvLUT, deconvolveIdx = NFFT.precompWindowHatInvLUT(params, N, Ñ, windowHatInvLUT_sep)

    return Reactant.to_rarray(I.(deconvolveIdx)), Reactant.to_rarray(real.(windowHatInvLUT))
end

@inline function _window_products_chunk(p::Reactant_NFFTPlan{T,1,M}, jlo::Int, jhi::Int) where {T,M}
    if p.windowProducts === nothing
        return p.windowTensor[:, 1, jlo:jhi]
    end
    return p.windowProducts[:, jlo:jhi]
end

@inline function _window_products_chunk(p::Reactant_NFFTPlan{T,2,M}, jlo::Int, jhi::Int) where {T,M}
    if p.windowProducts === nothing
        Jc = jhi - jlo + 1
        w1 = p.windowTensor[:, 1, jlo:jhi]
        w2 = p.windowTensor[:, 2, jlo:jhi]
        return reshape(
            reshape(w1, M, 1, Jc) .* reshape(w2, 1, M, Jc),
            M * M,
            Jc,
        )
    end
    return p.windowProducts[:, jlo:jhi]
end

@inline function _window_products_chunk(p::Reactant_NFFTPlan{T,3,M}, jlo::Int, jhi::Int) where {T,M}
    if p.windowProducts === nothing
        Jc = jhi - jlo + 1
        w1 = p.windowTensor[:, 1, jlo:jhi]
        w2 = p.windowTensor[:, 2, jlo:jhi]
        w3 = p.windowTensor[:, 3, jlo:jhi]
        return reshape(
            reshape(w1, M, 1, 1, Jc) .* reshape(w2, 1, M, 1, Jc) .* reshape(w3, 1, 1, M, Jc),
            M * M * M,
            Jc,
        )
    end
    return p.windowProducts[:, jlo:jhi]
end

@inline function _linear_indices_chunk(p::Reactant_NFFTPlan{T,1,M}, jlo::Int, jhi::Int) where {T,M}
    p.linearIndices !== nothing && return p.linearIndices[:, jlo:jhi]
    GI = eltype(p.stencilStarts)
    Jc = jhi - jlo + 1
    offs = reshape(GI.(0:(M - 1)), M, 1)
    n1 = GI(p.Ñ[1])
    s1 = p.stencilStarts[1, jlo:jhi]
    return mod.(reshape(s1, 1, Jc) .+ offs, n1) .+ one(GI)
end

@inline function _linear_indices_chunk(p::Reactant_NFFTPlan{T,2,M}, jlo::Int, jhi::Int) where {T,M}
    p.linearIndices !== nothing && return p.linearIndices[:, jlo:jhi]
    GI = eltype(p.stencilStarts)
    Jc = jhi - jlo + 1
    offs = reshape(GI.(0:(M - 1)), M, 1)
    n1 = GI(p.Ñ[1])
    n2 = GI(p.Ñ[2])
    s1 = p.stencilStarts[1, jlo:jhi]
    s2 = p.stencilStarts[2, jlo:jhi]
    z1 = mod.(reshape(s1, 1, Jc) .+ offs, n1)
    z2 = mod.(reshape(s2, 1, Jc) .+ offs, n2)
    return reshape(
        reshape(z1, M, 1, Jc) .+ n1 .* reshape(z2, 1, M, Jc),
        M * M,
        Jc,
    ) .+ one(GI)
end

@inline function _linear_indices_chunk(p::Reactant_NFFTPlan{T,3,M}, jlo::Int, jhi::Int) where {T,M}
    p.linearIndices !== nothing && return p.linearIndices[:, jlo:jhi]
    GI = eltype(p.stencilStarts)
    Jc = jhi - jlo + 1
    offs = reshape(GI.(0:(M - 1)), M, 1)
    n1 = GI(p.Ñ[1])
    n2 = GI(p.Ñ[2])
    n3 = GI(p.Ñ[3])
    s1 = p.stencilStarts[1, jlo:jhi]
    s2 = p.stencilStarts[2, jlo:jhi]
    s3 = p.stencilStarts[3, jlo:jhi]
    z1 = mod.(reshape(s1, 1, Jc) .+ offs, n1)
    z2 = mod.(reshape(s2, 1, Jc) .+ offs, n2)
    z3 = mod.(reshape(s3, 1, Jc) .+ offs, n3)
    n12 = n1 * n2
    return reshape(
        reshape(z1, M, 1, 1, Jc) .+ n1 .* reshape(z2, 1, M, 1, Jc) .+
        n12 .* reshape(z3, 1, 1, M, Jc),
        M * M * M,
        Jc,
    ) .+ one(GI)
end

@inline _prefer_global_forward(p::Reactant_NFFTPlan) = p.forwardGlobal
@inline _prefer_global_adjoint(p::Reactant_NFFTPlan) = p.adjointGlobal

#############################
# Convolution (forward: g -> fHat)
#############################

function AbstractNFFTs.convolve!(
    p::Reactant_NFFTPlan{T,D},
    g::AbstractArray{<:Number,D},
    fHat::AbstractVector{<:Number},
) where {T,D}
    gFlat = vec(g)

    if _prefer_global_forward(p)
        wp = _window_products_chunk(p, 1, p.J)
        li = _linear_indices_chunk(p, 1, p.J)
        gathered = gFlat[li]
        copyto!(p.tmpFHatSorted, vec(sum(gathered .* wp, dims=1)))
        copyto!(fHat, p.tmpFHatSorted[p.nodePermOrigToSorted])
        return fHat
    end

    @inbounds for c in 1:length(p.chunkCounts)
        jlo = p.chunkOffsets[c]
        jhi = p.chunkOffsets[c + 1] - 1
        jhi < jlo && continue

        li = _linear_indices_chunk(p, jlo, jhi)
        wp = _window_products_chunk(p, jlo, jhi)
        chunkVals = vec(sum(gFlat[li] .* wp, dims=1))
        p.tmpFHatSorted[jlo:jhi] .= chunkVals
    end

    copyto!(fHat, p.tmpFHatSorted[p.nodePermOrigToSorted])
    return fHat
end

#############################
# Convolution transpose (adjoint: fHat -> g)
#############################

function AbstractNFFTs.convolve_transpose!(
    p::Reactant_NFFTPlan{T,D},
    fHat::AbstractVector{<:Number},
    g::AbstractArray{<:Number,D},
) where {T,D}
    g .= zero(eltype(g))

    copyto!(p.tmpFHatSorted, fHat[p.nodePermSortedToOrig])

    gFlat = Reactant.promote_to(Reactant.TracedRArray, vec(g))

    if _prefer_global_adjoint(p)
        wp = _window_products_chunk(p, 1, p.J)
        updates = vec(wp .* reshape(p.tmpFHatSorted, 1, :))
        li = _linear_indices_chunk(p, 1, p.J)
        idx = Reactant.promote_to(Reactant.TracedRArray, reshape(Int64.(vec(li)), 1, :))
        upd = Reactant.promote_to(Reactant.TracedRArray, updates)

        gFlat = Reactant.Ops.scatter(
            +,
            [gFlat],
            idx,
            [upd];
            update_window_dims=Int64[],
            inserted_window_dims=Int64[1],
            input_batching_dims=Int64[],
            scatter_indices_batching_dims=Int64[],
            scatter_dims_to_operand_dims=Int64[1],
            index_vector_dim=1,
            unique_indices=false,
            indices_are_sorted=false,
        )[1]

        copyto!(g, reshape(gFlat, size(g)))
        return g
    end

    @inbounds for c in 1:length(p.chunkCounts)
        jlo = p.chunkOffsets[c]
        jhi = p.chunkOffsets[c + 1] - 1
        jhi < jlo && continue

        wp = _window_products_chunk(p, jlo, jhi)
        fh = reshape(p.tmpFHatSorted[jlo:jhi], 1, :)
        updatesLocal = vec(wp .* fh)

        if p.scatterIndicesSorted === nothing
            idxChunk = _linear_indices_chunk(p, jlo, jhi)
            idx = Reactant.promote_to(Reactant.TracedRArray, reshape(Int64.(vec(idxChunk)), 1, :))
            upd = Reactant.promote_to(Reactant.TracedRArray, updatesLocal)

            gFlat = Reactant.Ops.scatter(
                +,
                [gFlat],
                idx,
                [upd];
                update_window_dims=Int64[],
                inserted_window_dims=Int64[1],
                input_batching_dims=Int64[],
                scatter_indices_batching_dims=Int64[],
                scatter_dims_to_operand_dims=Int64[1],
                index_vector_dim=1,
                unique_indices=false,
                indices_are_sorted=false,
            )[1]
        else
            slo = p.scatterChunkOffsets[c]
            shi = p.scatterChunkOffsets[c + 1] - 1
            shi < slo && continue

            idxSorted = reshape(p.scatterIndicesSorted[slo:shi], 1, :)
            permLocal = p.scatterPermLocal[slo:shi]
            updatesSorted = updatesLocal[permLocal]

            idx = Reactant.promote_to(Reactant.TracedRArray, idxSorted)
            upd = Reactant.promote_to(Reactant.TracedRArray, updatesSorted)

            gFlat = Reactant.Ops.scatter(
                +,
                [gFlat],
                idx,
                [upd];
                update_window_dims=Int64[],
                inserted_window_dims=Int64[1],
                input_batching_dims=Int64[],
                scatter_indices_batching_dims=Int64[],
                scatter_dims_to_operand_dims=Int64[1],
                index_vector_dim=1,
                unique_indices=false,
                indices_are_sorted=true,
            )[1]
        end
    end

    copyto!(g, reshape(gFlat, size(g)))
    return g
end

#############################
# Deconvolution (f -> g)
#############################

function AbstractNFFTs.deconvolve!(
    p::Reactant_NFFTPlan{T,D},
    f::AbstractArray{<:Number,D},
    g::AbstractArray{<:Number,D},
) where {T,D}
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
    p::Reactant_NFFTPlan{T,D},
    g::AbstractArray{<:Number,D},
    f::AbstractArray{<:Number,D},
) where {T,D}
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
        update_window_dims=Int64[],
        inserted_window_dims=Int64[1],
        input_batching_dims=Int64[],
        scatter_indices_batching_dims=Int64[],
        scatter_dims_to_operand_dims=Int64[1],
        index_vector_dim=1,
        unique_indices=true,
        indices_are_sorted=true,
    )[1]

    copyto!(f, reshape(fFlat, size(f)))
    return nothing
end

#############################
# mul! for forward and adjoint
#############################

function LinearAlgebra.mul!(
    fHat::Reactant.AnyTracedRVector,
    p::Reactant_NFFTPlan{T,D},
    f::Reactant.AnyTracedRArray;
    verbose=false,
    timing::Union{Nothing,AbstractNFFTs.TimingStats}=nothing,
) where {T,D}
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
    verbose=false,
    timing::Union{Nothing,AbstractNFFTs.TimingStats}=nothing,
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

function Base.:*(p::Reactant_NFFTPlan{T,D}, f::Reactant.AnyTracedRArray; kargs...) where {T,D}
    fHat = similar(f, complex(eltype(f)), size_out(p))
    mul!(fHat, p, f; kargs...)
    return fHat
end

function Base.:*(pl::AdjointRPlan, fHat::Reactant.AnyTracedRVector; kargs...)
    f = similar(fHat, complex(eltype(fHat)), size_out(pl))
    mul!(f, pl, fHat; kargs...)
    return f
end
