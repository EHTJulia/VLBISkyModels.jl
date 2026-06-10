#==============================================================================
Shared spread/interpolation machinery for the type-1 and type-2 executes.

Everything here is regular Julia array code (broadcasts, getindex/setindex!,
reductions) that Reactant traces to StableHLO. The per-point stencil is the
same as FINUFFT's: each nonuniform point touches a w^D block of the
oversampled grid, with separable ES-kernel weights evaluated by Horner
polynomials (see kernel.jl).

Index convention: `base` is the 0-based grid coordinate of the leftmost
stencil cell per dim (may be negative / past the end before wrapping);
`frac ∈ [0, 1)` is the offset used for the Horner evaluation. Both are
stored in original point order by setpts.jl; the spread additionally applies
the bin-sort permutation chunk-wise (see `_chunk_stencil`).
==============================================================================#

# Horner weights for one dim and one chunk of points.
# `coefs :: (w, deg+1)` host matrix, `frac` traced (cs,) vector.
# Returns (cs, w): weights[j, k] = p_k(2*frac[j] - 1).
function _horner_weights(coefs::AbstractMatrix, frac::AbstractVector)
    t = 2 .* frac .- 1
    n = size(coefs, 2)
    acc = t .* transpose(coefs[:, n]) .+ transpose(coefs[:, n - 1])
    for p in (n - 2):-1:1
        acc = acc .* t .+ transpose(coefs[:, p])
    end
    return acc
end

# Wrapped 0-based grid coordinates of the stencil cells for one dim:
# (cs, w), entry [j, k] = mod(base[j] + k - 1, ngrid_d).
function _grid_offsets(base::AbstractVector, ngrid_d::Int, w::Int)
    return mod.(base .+ transpose(0:(w - 1)), ngrid_d)
end

# Combine per-dim wrapped offsets into 1-based linear indices over the
# flattened spatial grid. Shapes: (cs, w), (cs, w, w), (cs, w, w, w).
function _linear_indices(offs::NTuple{1, AbstractMatrix}, ::NTuple{1, Int})
    return offs[1] .+ 1
end

function _linear_indices(offs::NTuple{2, AbstractMatrix}, ngrid::NTuple{2, Int})
    cs, w = size(offs[1])
    return reshape(offs[1], cs, w, 1) .+
        ngrid[1] .* reshape(offs[2], cs, 1, w) .+ 1
end

function _linear_indices(offs::NTuple{3, AbstractMatrix}, ngrid::NTuple{3, Int})
    cs, w = size(offs[1])
    return reshape(offs[1], cs, w, 1, 1) .+
        ngrid[1] .* reshape(offs[2], cs, 1, w, 1) .+
        (ngrid[1] * ngrid[2]) .* reshape(offs[3], cs, 1, 1, w) .+ 1
end

# Separable tensor product of per-dim weights, matching the index shapes
# above: (cs, w), (cs, w, w), (cs, w, w, w).
_weight_product(wpd::NTuple{1, AbstractMatrix}) = wpd[1]

function _weight_product(wpd::NTuple{2, AbstractMatrix})
    cs, w = size(wpd[1])
    return reshape(wpd[1], cs, w, 1) .* reshape(wpd[2], cs, 1, w)
end

function _weight_product(wpd::NTuple{3, AbstractMatrix})
    cs, w = size(wpd[1])
    return reshape(wpd[1], cs, w, 1, 1) .*
        reshape(wpd[2], cs, 1, w, 1) .*
        reshape(wpd[3], cs, 1, 1, w)
end

# Contract gathered stencil values against the per-dim weights, last dim
# first, so the intermediates shrink: (cs, w, .., w) -> (cs,).
function _contract_weights(vals, wpd::NTuple{1, AbstractMatrix})
    return vec(sum(vals .* wpd[1]; dims = 2))
end

function _contract_weights(vals, wpd::NTuple{2, AbstractMatrix})
    cs, w = size(wpd[1])
    t2 = dropdims(sum(vals .* reshape(wpd[2], cs, 1, w); dims = 3); dims = 3)
    return vec(sum(t2 .* wpd[1]; dims = 2))
end

function _contract_weights(vals, wpd::NTuple{3, AbstractMatrix})
    cs, w = size(wpd[1])
    t3 = dropdims(sum(vals .* reshape(wpd[3], cs, 1, 1, w); dims = 4); dims = 4)
    t2 = dropdims(sum(t3 .* reshape(wpd[2], cs, 1, w); dims = 3); dims = 3)
    return vec(sum(t2 .* wpd[1]; dims = 2))
end

# Per-chunk stencil weights and linear indices, shared by spread and interp.
# `idx` selects the chunk's points: a plain `UnitRange` for original-order
# access (interp) or a traced `perm` slice for bin-sorted access (spread).
function _chunk_stencil(
        prep::NUFFTSetPts{T, D}, idx, coefs::AbstractMatrix,
    ) where {T, D}
    plan = prep.plan
    w = plan.nspread
    base_c = ntuple(d -> prep.base[d][idx], Val(D))
    frac_c = ntuple(d -> prep.frac[d][idx], Val(D))
    wpd = ntuple(d -> _horner_weights(coefs, frac_c[d]), Val(D))
    offs = ntuple(d -> _grid_offsets(base_c[d], plan.ngrid[d], w), Val(D))
    lin = _linear_indices(offs, plan.ngrid)
    return wpd, lin
end

# Static chunk ranges covering 1:M; the last chunk may be ragged.
function _chunk_ranges(M::Int, chunk_size::Int)
    cs = max(1, min(chunk_size, M))
    return [lo:min(lo + cs - 1, M) for lo in 1:cs:M]
end

#==============================================================================
Spread (type-1 kernel): scatter-add each point's weighted stencil into the
oversampled grid.

!!! warning "Performance blocker — sequential scatter"
    Reactant/StableHLO has no *parallel* scatter-add reachable through
    regular Julia array semantics:
      * `fw[lin] .+= upd` does not trace (scalar-indexing fallback), and
        Base semantics would drop duplicate-index contributions anyway;
      * the `@trace for` accumulation loop below traces correctly but
        lowers to a sequential `stablehlo.while` (one dynamic_update_slice
        per update) — the enzymexla loop-raising passes do not lift it to
        `stablehlo.scatter`.
    This makes standalone type-1 transforms O(M * w^D) *serial*. The type-2
    path (what VLBISkyModels uses) does not go through this function, and
    its Enzyme adjoint generates the parallel scatter internally.
==============================================================================#
function _scatter_add!(fw_vec::AbstractVector, lin::AbstractVector, upd::AbstractVector)
    Reactant.@trace track_numbers = false for n in 1:length(lin)
        @allowscalar fw_vec[lin[n]] += upd[n]
    end
    return fw_vec
end

# Spread all points of all transforms into a fresh oversampled grid.
# `cmat :: (M, ntrans)` complex strengths in original (unsorted) point order.
# Returns `fw :: (ngrid..., ntrans)` complex.
function _spread(prep::NUFFTSetPts{T, D}, cmat::AbstractMatrix, ntrans::Int) where {T, D}
    plan = prep.plan
    CT = complex(T)
    nflat = prod(plan.ngrid)

    fw_vec = similar(cmat, CT, nflat * ntrans)
    fill!(fw_vec, zero(CT))

    coefs = plan.horner_coefs
    for r in _chunk_ranges(prep.M, plan.chunk_size)
        perm_c = prep.perm[r]                              # bin-sorted point ids
        wpd, lin = _chunk_stencil(prep, perm_c, coefs)
        wprod = _weight_product(wpd)                       # (cs, w^D) real
        cc = cmat[perm_c, :]                               # (cs, ntrans) complex
        for t in 1:ntrans
            upd = wprod .* cc[:, t]                        # (cs, w^D)
            lin_t = lin .+ (t - 1) * nflat
            fw_vec = _scatter_add!(fw_vec, vec(lin_t), vec(upd))
        end
    end

    return reshape(fw_vec, (plan.ngrid..., ntrans))
end

#==============================================================================
Interpolate (type-2 kernel): gather each point's stencil from the (already
FFT'd and deconvolved) oversampled grid and contract against the weights.
The traced-integer-array getindex lowers to one `stablehlo.gather` per chunk.
==============================================================================#

# Interpolate all points for all transforms.
# `fw :: (ngrid..., ntrans)` complex; returns `c :: (M, ntrans)` complex.
#
# Points are read in *original* order (no bin-sort indirection), so the
# output is assembled by plain `hcat`/`vcat` of chunk results and needs no
# final permutation gather. This is deliberate twice over for Reactant
# 0.2.264:
#   * traced-index gathers whose complex operand is a concatenate or a
#     `setindex!`(dynamic_update_slice)-built array are miscompiled by the
#     scatter/gather optimization passes (silent zeros, or an invalid
#     `stablehlo.real` op) — so the output must not be gathered again;
#   * the chunk/transform loops are explicit `for` loops because the same
#     body written as `map(...) do` closures recurses infinitely between
#     `Base.mapreduce` and Reactant's `Base.mapreducedim!` at trace time
#     (StackOverflowError).
# The remaining stencil gather reads from the FFT output directly, which is
# unaffected by the gather miscompilation.
function _interp(prep::NUFFTSetPts{T, D}, fw::AbstractArray, ntrans::Int) where {T, D}
    plan = prep.plan
    nflat = prod(plan.ngrid)
    fw_vec = vec(fw)
    coefs = plan.horner_coefs

    chunk_rows = []
    for r in _chunk_ranges(prep.M, plan.chunk_size)
        wpd, lin = _chunk_stencil(prep, r, coefs)
        cols = []
        for t in 1:ntrans
            vals = fw_vec[lin .+ (t - 1) * nflat]          # (cs, w, .., w)
            push!(cols, _contract_weights(vals, wpd))      # (cs,)
        end
        row = ntrans == 1 ? reshape(cols[1], :, 1) : hcat(cols...)
        push!(chunk_rows, row)
    end
    return length(chunk_rows) == 1 ? chunk_rows[1] : vcat(chunk_rows...)
end
