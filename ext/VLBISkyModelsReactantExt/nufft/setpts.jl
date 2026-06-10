#==============================================================================
set_nufft_points: per-dim base index / fractional offset + bin-sort permutation.

Mirrors cuFINUFFT setpts: wrap to [0, 2pi), scale to the oversampled grid,
take floor/frac, and compute a spatial-bin `sortperm`. Regular Julia traced
through Reactant.

The per-point stencil data (`base`, `frac`) is stored in *original* point
order. The bin-sort permutation `perm` is only applied chunk-wise by the
type-1 spread (where scatter locality matters); the type-2 interpolation
reads points in original order, so its output needs no final un-sorting
gather. (That also sidesteps a Reactant 0.2.264 miscompilation of gathers
whose complex operand is a concatenate/dynamic_update_slice — see
spread_interp.jl.)
==============================================================================#

"""
    NUFFTSetPts{T,D}

Output of [`set_nufft_points`](@ref). Holds the per-point stencil metadata
consumed by the execute step.
"""
struct NUFFTSetPts{T <: Real, D, P, A, B, F}
    plan::P
    M::Int                 # point count
    perm::A                # length M: bin-sorted position -> original index
    base::B                # NTuple{D, length-M Int}: 0-based leftmost stencil cell
    frac::F                # NTuple{D, length-M T}: Horner offset in [0, 1)
end

ndims_(::NUFFTSetPts{<:Any, D}) where {D} = D
Base.eltype(::NUFFTSetPts{T}) where {T} = complex(T)

# --- Traced kernel ---------------------------------------------------------

function _setpts_traced(
        points::NTuple{D, AbstractVector},
        ngrid::NTuple{D, Int},
        nspread::Int,
        bin_dims::NTuple{D, Int},
        nbins::NTuple{D, Int},
    ) where {D}
    T = real(Reactant.unwrapped_eltype(eltype(points[1])))
    period = T(2 * pi)
    # Stencil placement: base = floor(s - w/2 + 1), frac = s - base - (w/2 - 1).
    # Equivalent to FINUFFT's `i1 = ceil(s - w/2)` for non-integer s, and works
    # uniformly for even and odd `nspread`. This placement keeps every stencil
    # cell's kernel argument z inside [-1, 1], where the Horner polynomial fit
    # (see `horner_coefficients`) is valid.
    halfw_minus_one = T(nspread - 2) / T(2)

    s = ntuple(d -> mod.(points[d], period) .* (T(ngrid[d]) / period), Val(D))
    s_shift = ntuple(d -> s[d] .- halfw_minus_one, Val(D))
    base = ntuple(d -> floor.(Int, s_shift[d]), Val(D))
    frac = ntuple(d -> s_shift[d] .- base[d], Val(D))

    # Spatial bin id per point; `perm` groups points with nearby stencils so
    # the type-1 scatter targets are clustered.
    stride = 1
    bin_id = mod.(base[1], ngrid[1]) .÷ bin_dims[1]
    for d in 2:D
        stride *= nbins[d - 1]
        bin_id = bin_id .+ (mod.(base[d], ngrid[d]) .÷ bin_dims[d]) .* stride
    end
    perm = sortperm(bin_id)

    return perm, base, frac
end

# --- User-facing entry point -----------------------------------------------

"""
    set_nufft_points(plan::NUFFTPlan{T,D}, x_1, ..., x_D) -> NUFFTSetPts
    set_nufft_points(plan, (x_1, ..., x_D)) -> NUFFTSetPts

Bind point coordinates to a plan. Each `x_d` is a length-`M` array of real
coordinates in radians (wrapped to `[0, 2pi)` internally).

This is plain traceable Julia — call it inside `Reactant.@jit` /
`Reactant.@compile` to get a compiled setpts step, or call it directly with
`ConcreteRArray`s for one-off use.
"""
function set_nufft_points(plan::NUFFTPlan{T, D}, points::Vararg{AbstractVector, D}) where {T, D}
    return set_nufft_points(plan, points)
end

function set_nufft_points(
        plan::NUFFTPlan{T, D}, points::NTuple{D, AbstractVector}
    ) where {T, D}
    M = length(points[1])
    @assert all(p -> length(p) == M, points) "All coordinate vectors must share length"

    perm, base, frac = _setpts_traced(
        points, plan.ngrid, plan.nspread, plan.bin_dims, plan.nbins,
    )

    return NUFFTSetPts{
        T, D, typeof(plan), typeof(perm), typeof(base), typeof(frac),
    }(
        plan, M, perm, base, frac,
    )
end
