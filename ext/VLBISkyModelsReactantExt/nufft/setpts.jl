#==============================================================================
set_nufft_points: bin-sort + per-dim base index / fractional offset.

Mirrors cuFINUFFT setpts: wrap to [0, 2pi), scale to the oversampled grid,
take floor/frac, build a bin index, `sortperm`, then `getindex` to permute
the per-dim arrays. Plain Julia traced through Reactant — no `@opcall`.
==============================================================================#

"""
    NUFFTSetPts{T,D}

Output of [`set_nufft_points`](@ref). Holds the bin-sorted per-point
metadata, padded to `M_pad = nchunks * chunk_size` so the chunked
spread/gather `@trace for` loop can use a static slice size.

Padded entries (j > M) are zeroed via `mask`; the padded `base`/`frac`
entries are valid sentinel values (base=1, frac=0) that scatter/gather
nothing meaningful but keep all indices in range.
"""
struct NUFFTSetPts{T<:Real,D,P,A,B,F,Mk}
    plan::P
    M::Int                 # original point count
    M_pad::Int             # = nchunks * chunk_size
    nchunks::Int
    chunk_size::Int        # static per-iter slice size
    perm::A                # ConcreteRArray{Int,1}, length M_pad (user → sorted, padded with 1's)
    invperm::A             # ConcreteRArray{Int,1}, length M     (sorted → user)
    base_sorted::B         # NTuple{D, ConcreteRArray{Int,1}}, length M_pad each
    frac_sorted::F         # NTuple{D, ConcreteRArray{T,1}},   length M_pad each
    mask::Mk               # ConcreteRArray{T,1}, length M_pad (1.0 for j≤M, 0.0 else)
end

ndims_(::NUFFTSetPts{<:Any,D}) where {D} = D
Base.eltype(::NUFFTSetPts{T}) where {T} = T

# --- Traced kernel ---------------------------------------------------------

# Compute the bin-sort permutation, sorted base/frac per dim, then PAD all
# per-point arrays out to M_pad so the execute path sees a static slice
# size. Padded entries are zeroed at execute via `mask`.
#
# All arithmetic is plain Julia, traced through Reactant.
function _setpts_traced(
    points::NTuple{D,AbstractVector},
    ngrid::NTuple{D,Int},
    nspread::Int,
    bin_dims::NTuple{D,Int},
    nbins::NTuple{D,Int},
    M_pad::Int,
) where {D}
    M = length(points[1])
    T = real(Reactant.unwrapped_eltype(eltype(points[1])))
    period = T(2 * pi)
    half_w_offset = nspread ÷ 2 - 1     # base = floor(s) - half_w_offset

    s = ntuple(d -> mod.(points[d], period) .* (T(ngrid[d]) / period), Val(D))
    base = ntuple(d -> floor.(Int, s[d]) .- half_w_offset, Val(D))
    frac = ntuple(d -> s[d] .- floor.(s[d]), Val(D))

    stride = 1
    bin_id = mod.(base[1], ngrid[1]) .÷ bin_dims[1]
    for d in 2:D
        stride *= nbins[d - 1]
        bin_id = bin_id .+ (mod.(base[d], ngrid[d]) .÷ bin_dims[d]) .* stride
    end
    perm = sortperm(bin_id)
    invp = sortperm(perm)
    base_s = ntuple(d -> base[d][perm], Val(D))
    frac_s = ntuple(d -> frac[d][perm], Val(D))

    # Pad to M_pad with sentinel values: perm→1, base→1, frac→0.
    # Padding entries' contributions are killed by `mask`.
    perm_p, base_p, frac_p, mask = _pad_for_chunks(perm, base_s, frac_s, M, M_pad, T, Val(D))
    return perm_p, invp, base_p, frac_p, mask
end

function _pad_for_chunks(
    perm, base_s::NTuple{D,Any}, frac_s::NTuple{D,Any},
    M::Int, M_pad::Int, ::Type{T}, ::Val{D},
) where {T,D}
    one_T = Reactant.promote_to(Reactant.TracedRNumber{T}, one(T))
    if M_pad == M
        return perm, base_s, frac_s, fill(one_T, M_pad)
    end
    npad = M_pad - M
    one_I  = Reactant.promote_to(Reactant.TracedRNumber{Int}, Int(1))
    zero_T = Reactant.promote_to(Reactant.TracedRNumber{T},     zero(T))
    pad_perm = fill(one_I,  npad)
    pad_int  = fill(one_I,  npad)
    pad_T    = fill(zero_T, npad)
    perm_p = vcat(perm, pad_perm)
    base_p = ntuple(d -> vcat(base_s[d], pad_int), Val(D))
    frac_p = ntuple(d -> vcat(frac_s[d], pad_T),   Val(D))
    mask = vcat(fill(one_T, M), fill(zero_T, npad))
    return perm_p, base_p, frac_p, mask
end

# --- User-facing entry point -----------------------------------------------

"""
    set_nufft_points(plan::NUFFTPlan{T,D}, x_1, ..., x_D) -> NUFFTSetPts
    set_nufft_points(plan, (x_1, ..., x_D)) -> NUFFTSetPts

Bind point coordinates to a plan. Each `x_d` is a length-`M` array of real
coordinates in radians (will be wrapped to `[0, 2pi)` internally).

This is plain traceable Julia — call it inside `Reactant.@jit` /
`Reactant.@compile` to get a compiled setpts step, or call it directly with
`ConcreteRArray`s for one-off use.
"""
function set_nufft_points(plan::NUFFTPlan{T,D}, points::Vararg{AbstractVector,D}) where {T,D}
    return set_nufft_points(plan, points)
end

function set_nufft_points(
    plan::NUFFTPlan{T,D}, points::NTuple{D,AbstractVector}
) where {T,D}
    M = length(points[1])
    @assert all(p -> length(p) == M, points) "All coordinate vectors must share length"

    nchunks = max(1, cld(M, plan.chunk_size))
    chunk_size = cld(M, nchunks)
    M_pad = nchunks * chunk_size


    perm, invp, base_s, frac_s, mask = _setpts_traced(
        points, plan.ngrid, plan.nspread, plan.bin_dims, plan.nbins, M_pad,
    )

    return NUFFTSetPts{
        T,D,typeof(plan),typeof(perm),typeof(base_s),typeof(frac_s),typeof(mask),
    }(
        plan, M, M_pad, nchunks, chunk_size, perm, invp, base_s, frac_s, mask,
    )
end
