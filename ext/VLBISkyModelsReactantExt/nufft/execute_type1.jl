#==============================================================================
Type-1 execute: spread → FFT → deconvolve + crop.

Spreading is chunked: each chunk processes `chunk_size` bin-sorted points,
materializing only a (chunk_size, w^D, ntrans) intermediate before the
scatter-add. This bounds peak memory and gives XLA scatter-add bin-localized
targets (less atomic contention on GPU, better cache reuse on CPU).

The body is plain Julia for everything except the scatter, which calls
`@opcall scatter` with custom `dimension_numbers` so the (Nf..., ntrans)
layout works without permuting the FFT axes.
==============================================================================#

# ---------- helpers ---------------------------------------------------------

# Per-chunk Horner weights for one dim:
#   t = 2*frac - 1 ∈ [-1, 1], shape (M,)
#   weights = horner_eval(coefs, t) :: (M, w)
@inline function _per_dim_weights(coefs::AbstractMatrix{T}, frac::AbstractVector) where {T}
    return horner_eval(coefs, T(2) .* frac .- T(1))
end

# Outer product of per-dim weight matrices (M, w_d) for d in 1..D.
# Returns (M, prod(w_d)) — for our case w_d = w for all d, so (M, w^D).
@inline function _outer_product_weights(wpd::NTuple{1, AbstractMatrix})
    return wpd[1]                                              # (M, w)
end
@inline function _outer_product_weights(wpd::NTuple{2, AbstractMatrix})
    M, w = size(wpd[1])
    return reshape(reshape(wpd[1], M, w, 1) .* reshape(wpd[2], M, 1, w), M, w * w)
end
@inline function _outer_product_weights(wpd::NTuple{3, AbstractMatrix})
    M, w = size(wpd[1])
    a = reshape(wpd[1], M, w, 1, 1)
    b = reshape(wpd[2], M, 1, w, 1)
    c = reshape(wpd[3], M, 1, 1, w)
    return reshape(a .* b .* c, M, w * w * w)
end

# Per-dim local stencil indices (M, w), 1-based Julia indices into the fw grid.
# `base` is 0-based physical position of the leftmost stencil cell (may be
# negative or > ngrid; periodic wrap is applied here). `offsets_row` is a
# precomputed (1, w) Int row, hoisted outside any `@trace for` to keep `w`
# from being lifted to a traced value.
@inline function _per_dim_indices(base::AbstractVector, ngrid::Integer, offsets_row::AbstractMatrix)
    M = length(base)
    return mod.(reshape(base, M, 1) .+ offsets_row, ngrid) .+ 1
end

# Stack per-dim (M, w) index matrices into a single (D, M*w^D) Int matrix.
# The (D, N) layout (rather than (N, D)) is what `stablehlo.scatter` with
# index_vector_dim=0 wants natively — using (N, D) forces XLA to insert a
# 49 MB-per-chunk transpose that consumed ~3 ms/call (50% of T1 e2e at
# the D=2 256² M=10⁶ target row).
@inline function _stack_scatter_indices(idxs::NTuple{1, AbstractMatrix})
    M, w = size(idxs[1])
    return reshape(idxs[1], 1, M * w)
end
# Build (D, M*w^D) directly via broadcast-add with zero, avoiding `repeat`
# (which materializes a (M, w, w) intermediate that XLA layout-flips
# 49 MB-per-chunk before the scatter — ~3 ms/call at the D=2 256² M=10⁶
# target row).
@inline function _stack_scatter_indices(idxs::NTuple{2, AbstractMatrix})
    M, w = size(idxs[1])
    z = zero(eltype(idxs[1]))
    a = reshape(idxs[1], M, w, 1) .+ z .* reshape(idxs[2], M, 1, w)   # (M, w, w)
    b = reshape(idxs[2], M, 1, w) .+ z .* reshape(idxs[1], M, w, 1)   # (M, w, w)
    return vcat(reshape(a, 1, M * w * w), reshape(b, 1, M * w * w))
end
@inline function _stack_scatter_indices(idxs::NTuple{3, AbstractMatrix})
    M, w = size(idxs[1])
    z = zero(eltype(idxs[1]))
    a = reshape(idxs[1], M, w, 1, 1) .+
        z .* reshape(idxs[2], M, 1, w, 1) .+
        z .* reshape(idxs[3], M, 1, 1, w)
    b = reshape(idxs[2], M, 1, w, 1) .+
        z .* reshape(idxs[1], M, w, 1, 1) .+
        z .* reshape(idxs[3], M, 1, 1, w)
    c = reshape(idxs[3], M, 1, 1, w) .+
        z .* reshape(idxs[1], M, w, 1, 1) .+
        z .* reshape(idxs[2], M, 1, w, 1)
    N = M * w * w * w
    return vcat(reshape(a, 1, N), reshape(b, 1, N), reshape(c, 1, N))
end

# ---------- the scatter call ------------------------------------------------
#
# Real-valued single-operand scatter. The spread path scatters real and imag
# parts separately into two `(ngrid..., ntrans)` real tensors, recombining
# via `complex.(re, im)` immediately before the FFT. Going through complex
# scatter directly forces XLA to insert a 32 MB layout-transpose between
# scatter and FFT (~187 µs / call on D=2 M=10⁶ N=1024² T1, DRAM-bound);
# the split keeps both tensors in the FFT-input layout.
#
# Note on `indices_are_sorted`: bin-sorted indices are clustered but not
# strictly lex-sorted. Asserting the hint is a no-op on XLA-CPU and ~7×
# SLOWER on XLA-GPU (it picks a sequential code path), so we don't pass it.
function _scatter_add_real!(
        fw, scatter_idx::AbstractMatrix, updates::AbstractMatrix, ::Val{D},
    ) where {D}
    T = Reactant.unwrapped_eltype(eltype(fw))
    res = Reactant.Ops.@opcall scatter(
        +,
        [Reactant.promote_to(Reactant.TracedRArray{T, D + 1}, fw)],
        Reactant.promote_to(Reactant.TracedRArray{Int, 2}, scatter_idx),
        [Reactant.promote_to(Reactant.TracedRArray{T, 2}, updates)];
        update_window_dims = Int64[2],                       # ntrans axis of updates
        inserted_window_dims = collect(Int64, 1:D),          # spatial dims of fw
        input_batching_dims = Int64[],
        scatter_indices_batching_dims = Int64[],
        scatter_dims_to_operand_dims = collect(Int64, 1:D),  # idx rows → spatial dims
        index_vector_dim = Int64(1),                          # (D, N) layout — see _stack_scatter_indices
    )
    return only(res)
end


# ---------- central-mode crop (with periodic wrap) --------------------------
#
# Read the 2^D corners of fw_hat (positive modes from front, negative modes
# from back of the oversampled grid) and place them into a pre-allocated
# central buffer in the centered-mode layout `[neg_half; non_neg_half]` along
# each dim. Plain Julia slice + setindex; Reactant lowers each iteration to
# `stablehlo.slice` + `stablehlo.dynamic_update_slice`.

function _central_view(fw_hat, nmodes::NTuple{D, Int}, ngrid::NTuple{D, Int}) where {D}
    if all(nmodes .== ngrid)
        return fw_hat
    end
    @assert ndims(fw_hat) == D + 1
    out = similar(fw_hat, (nmodes..., size(fw_hat, D + 1)))
    fill!(out, zero(eltype(fw_hat)))
    halves = nmodes .÷ 2
    n_poss = nmodes .- halves
    for sign_bits in 0:(1 << D - 1)
        is_neg = ntuple(d -> ((sign_bits >> (d - 1)) & 1) == 1, Val(D))
        any(d -> is_neg[d] && halves[d] == 0, 1:D) && continue
        src = ntuple(
            d -> is_neg[d] ?
                ((ngrid[d] - halves[d] + 1):ngrid[d]) :
                (1:n_poss[d]), Val(D)
        )
        dst = ntuple(
            d -> is_neg[d] ?
                (1:halves[d]) :
                ((halves[d] + 1):nmodes[d]), Val(D)
        )
        out[dst..., :] = fw_hat[src..., :]
    end
    return out
end


# ---------- phi_hat outer-product for deconvolution -------------------------
#
# Build a (nmodes_1, ..., nmodes_D, 1) tensor whose entries are the product
# of the per-dim phi_hat values, ready to broadcast-divide fw_central. The
# trailing dim is the (singleton) ntrans axis.
function _phi_hat_tensor(plan::NUFFTPlan{T, D}) where {T, D}
    factors = ntuple(D) do d
        shape = ntuple(i -> i == d ? plan.nmodes[d] : 1, Val(D + 1))
        return reshape(plan.phi_hat[d], shape...)
    end
    return reduce(.*, factors)
end

# ---------- main entry point ------------------------------------------------

"""
    execute_type1(prep, c) -> fk

Type-1 NUFFT: nonuniform → uniform.
- `c::AbstractArray{<:Complex}` of shape `(M,)` or `(M, ntrans)`.
- Returns `fk` of shape `nmodes...` (when input was `(M,)`) or
  `(nmodes..., ntrans)` (when input was `(M, ntrans)`).

Designed to be called inside `Reactant.@jit`.
"""
function execute_type1(prep::NUFFTSetPts{T, D}, c::AbstractArray) where {T, D}
    plan = prep.plan
    @assert nufft_type(plan) == 1 "Plan was not built for type-1"
    @assert size(c, 1) == prep.M "Strength count mismatch with prepared points"

    squeeze_out = ndims(c) == 1
    cmat = squeeze_out ? reshape(c, prep.M, 1) : c
    ntrans = size(cmat, 2)

    return _execute_type1_impl(prep, cmat, ntrans, squeeze_out)
end

function _execute_type1_impl(
        prep::NUFFTSetPts{T, D}, cmat::AbstractMatrix, ntrans::Int, squeeze_out::Bool
    ) where {T, D}
    plan = prep.plan
    w = plan.nspread
    ngrid = plan.ngrid
    nchunks = prep.nchunks
    cs = prep.chunk_size

    # 1. Chunked spread: bin-sorted points → fw via repeated scatter-add.
    #    The bin-sort permutation and the pad-mask are applied per-chunk
    #    inside `_spread_chunks` rather than via an upfront materialized
    #    `c_sorted = cmat[perm, :] .* mask` — keeping the gather in the same
    #    op group as the contribution build lets XLA fuse them and avoids
    #    a (M_pad, ntrans) intermediate that costs ~25–35% of e2e at M=10⁶.
    coefs = Reactant.promote_to(Reactant.TracedRArray{T, 2}, plan.horner_coefs)
    offsets_row = reshape(collect(0:(w - 1)), 1, w)         # static (1, w) Int row
    fw_re = similar(cmat, real(eltype(cmat)), ngrid..., ntrans)
    fw_im = similar(cmat, real(eltype(cmat)), ngrid..., ntrans)
    fw_re, fw_im = _spread_chunks(
        fw_re, fw_im, cmat, prep.perm, prep.mask,
        prep.base_sorted, prep.frac_sorted, coefs,
        ngrid, w, ntrans, nchunks, cs, offsets_row, Val(D),
    )
    fw = complex.(fw_re, fw_im)

    # 2. FFT (sign per iflag).
    fw_hat = plan.iflag < 0 ?
        AbstractFFTs.fft(fw, 1:D) :
        AbstractFFTs.bfft(fw, 1:D)

    # 3. Crop central modes (with periodic wrap).
    fw_central = _central_view(fw_hat, plan.nmodes, ngrid)

    # 4. Deconvolve by separable phi_hat product.
    phih = _phi_hat_tensor(plan)
    fk = fw_central ./ phih

    return squeeze_out ? dropdims(fk; dims = D + 1) : fk
end

# --- chunked spread ---------------------------------------------------------
#
# Each iteration computes Horner weights, multi-D stencil indices, and a
# contribution tensor of shape (chunk_size, w^D, ntrans), then scatter-adds
# it into the (real, imag) tensor pair. With bin-sorted ordering, each
# chunk's scatter targets are spatially localized.
#
# Static unroll — for our target sweep nchunks ≤ ~16 so the IR stays
# bounded. Larger M (10⁷) would benefit from a `@trace for` loop, but
# that path runs into loop-carried-state issues with the scatter pattern.
function _spread_chunks(
        fw_re, fw_im, cmat, perm, mask, base_full, frac_full, coefs,
        ngrid::NTuple{ND, Int}, w::Int, ntrans::Int,
        nchunks::Int, cs::Int, offsets_row::AbstractMatrix, dimval::Val{ND},
    ) where {ND}
    wD = w^ND
    for k in 1:nchunks
        j0 = (k - 1) * cs + 1
        bs = ntuple(d -> base_full[d][j0:(j0 + cs - 1)], dimval)
        fr = ntuple(d -> frac_full[d][j0:(j0 + cs - 1)], dimval)
        cc = _gather_chunk_strengths(cmat, perm, mask, j0, cs)

        wpd = ntuple(d -> _per_dim_weights(coefs, fr[d]), dimval)
        idxpd = ntuple(d -> _per_dim_indices(bs[d], ngrid[d], offsets_row), dimval)
        weights = _outer_product_weights(wpd)               # real (cs, wD)
        scatter_idx = _stack_scatter_indices(idxpd)         # (ND, cs*wD)

        # Splitting cc into real/imag keeps the kernel multiply real.
        cc_re = real.(cc)
        cc_im = imag.(cc)
        contrib_re = reshape(weights, cs, wD, 1) .* reshape(cc_re, cs, 1, ntrans)
        contrib_im = reshape(weights, cs, wD, 1) .* reshape(cc_im, cs, 1, ntrans)
        upd_re = reshape(contrib_re, cs * wD, ntrans)
        upd_im = reshape(contrib_im, cs * wD, ntrans)

        fw_re = _scatter_add_real!(fw_re, scatter_idx, upd_re, dimval)
        fw_im = _scatter_add_real!(fw_im, scatter_idx, upd_im, dimval)
    end
    return fw_re, fw_im
end

# Per-chunk strength gather + mask. Replaces an upfront
# `c_sorted = cmat[perm, :] .* mask` materialization that costs ~25–35%
# of e2e on M=10⁶ T1 rows.
@inline function _gather_chunk_strengths(cmat, perm, mask, j0, cs::Int)
    perm_slice = perm[j0:(j0 + cs - 1)]         # (cs,) Int — bin-sorted source rows
    mask_slice = mask[j0:(j0 + cs - 1)]         # (cs,)     — pad-row mask
    cc_un = cmat[perm_slice, :]                 # (cs, ntrans) — gather original strengths
    return cc_un .* reshape(mask_slice, cs, 1)
end
