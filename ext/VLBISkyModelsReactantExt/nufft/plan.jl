#==============================================================================
NUFFTPlan: host-side plan object holding everything that depends only on
(eltype, transform type, nmodes, options). Built once, then re-used across
many setpts/execute calls.

The plan does NOT hold any traced arrays. Reactant's `@compile` cache keys
its compiled functions on argument types, so we don't need our own dict.
==============================================================================#

"""
    NUFFTPlan{T,D,K}

Static plan parameters. `T` is the real eltype, `D` is the dimensionality,
`K` is the transform type (1 or 2).

Construct via [`plan_nufft`](@ref).
"""
struct NUFFTPlan{T <: Real, D, K}
    nmodes::NTuple{D, Int}
    ngrid::NTuple{D, Int}
    iflag::Int                        # +1 or -1
    eps::T
    sigma::T
    nspread::Int                      # = w
    beta::T
    bin_dims::NTuple{D, Int}
    nbins::NTuple{D, Int}
    chunk_size::Int
    horner_coefs::Matrix{T}           # (w, deg+1)
    phi_hat::NTuple{D, Vector{T}}      # length nmodes[d] each
end

nufft_type(::NUFFTPlan{<:Any, <:Any, K}) where {K} = K
Base.eltype(::NUFFTPlan{T}) where {T} = T
ndims_(::NUFFTPlan{<:Any, D}) where {D} = D
nspread(p::NUFFTPlan) = p.nspread

# Smallest n >= n0 that is a product of {2, 3, 5, 7}. Standard FFT-friendly size.
function _next_smooth(n0::Integer)
    n = max(Int(n0), 1)
    primes = (2, 3, 5, 7)
    while true
        m = n
        for p in primes
            while m > 1 && m % p == 0
                m = m ÷ p
            end
        end
        m == 1 && return n
        n += 1
    end
    return
end

# Default `α` multiplier on `w` for the bin-sort permutation granularity.
# `bin_dim ≥ w` is required for binning to provide locality benefit (each
# NU point's stencil spans `w` cells per dim).
#
# A1 sweep finding (see PROFILE.md): GPU performance vs `bin_dim` is *very*
# non-monotonic with sharp cache cliffs (5–25× swings between adjacent bin
# sizes). No single `α` is robust — every default lands in good pockets
# for some workloads and bad pockets for others. We therefore keep the
# original cuFINUFFT-style heuristic (`max(floor, 2w)`) as the default
# (it was the configuration fix #4 was tuned against and avoids the worst
# regressions), and expose the multiplier for per-workload tuning.
#
# **CPU prefers larger bins** (α≈3 wins by 25–35% on 2D M=10⁶ type-1).
# Override via `plan_nufft(...; bin_dims=(b,b,...))` or by setting
# `AUTO_BIN_ALPHA[]` per run.
#
# Layout: AUTO_BIN_ALPHA[][K][D] for K in 1:2 (type), D in 1:3 (dim).
const AUTO_BIN_ALPHA = Ref(
    (
        (2.0, 2.0, 1.0),   # type-1
        (2.0, 2.0, 1.0),   # type-2
    )
)

# Heuristic bin width per dim. `bin_dim = max(floor_d, w, ceil(α·w))`;
# floor and α together reproduce the original cuFINUFFT-style heuristic
# (`max(32, 2w)` for 1D, `max(16, 2w)` for 2D, `max(8, w+1)` for 3D).
function _auto_bin_dims(
        D::Integer, K::Integer, w::Integer; alpha::Real = AUTO_BIN_ALPHA[][K][D],
    )
    floor_d = D == 1 ? 32 : (D == 2 ? 16 : 8)
    b = max(floor_d, w, ceil(Int, alpha * w))
    return ntuple(_ -> b, D)
end

"""
    plan_nufft(T, nufft_type, nmodes; iflag=-1, ntrans=1, opts=VLBISkyModels.ReactantNUFFTAlg(T), kwargs...)

Construct a NUFFT plan for real eltype `T`, transform type 1 or 2, and a
`D`-tuple of mode counts. `kwargs` override fields of `opts`.

`iflag` follows the NUFFT convention: `<0` ⇒ `exp(-i k·x)` (forward FFT),
`>0` ⇒ `exp(+i k·x)` (backward FFT).
"""
function plan_nufft(
        ::Type{T}, K::Integer, nmodes::NTuple{D, Integer};
        iflag::Integer = -1,
        opts::VLBISkyModels.ReactantNUFFTAlg{T} = VLBISkyModels.ReactantNUFFTAlg(T),
        kwargs...,
    ) where {T <: Real, D}
    @assert K == 1 || K == 2 "Only NUFFT type 1 and 2 are supported"
    @assert D == length(nmodes) "Dimension mismatch"
    @assert all(>(0), nmodes) "Mode counts must be positive"

    eps_ = T(get(kwargs, :eps, opts.eps))
    sigma = T(get(kwargs, :sigma, opts.sigma))
    nsp = Int(get(kwargs, :nspread, opts.nspread))
    cz = Int(get(kwargs, :chunk_size, opts.chunk_size))
    bd_in = get(kwargs, :bin_dims, opts.bin_dims)

    if nsp < 0
        nsp = _nspread_for_eps(T, eps_, sigma)
    end

    # Oversampled grid: at least sigma*N, FFT-friendly, room for kernel halo.
    modes_t = ntuple(d -> Int(nmodes[d]), D)
    ngrid_raw = ntuple(d -> max(2 * nsp + 1, ceil(Int, sigma * modes_t[d])), D)
    ngrid_t = ntuple(d -> _next_smooth(ngrid_raw[d]), D)

    bd = if isempty(bd_in)
        _auto_bin_dims(D, Int(K), nsp)
    else
        @assert length(bd_in) == D "bin_dims must have length D=$D"
        ntuple(d -> Int(bd_in[d]), D)
    end
    nb = ntuple(d -> max(1, cld(ngrid_t[d], bd[d])), D)

    beta = expsemicircle_beta(T, nsp, sigma)
    coefs = horner_coefficients(T, nsp, sigma)
    phih = ntuple(d -> phi_hat_1d(T, nsp, sigma, ngrid_t[d], modes_t[d]), D)

    return NUFFTPlan{T, D, Int(K)}(
        modes_t,
        ngrid_t,
        iflag >= 0 ? 1 : -1,
        eps_, sigma, nsp, beta,
        bd, nb, cz,
        coefs, phih,
    )
end

# Single-mode shortcut (1D)
plan_nufft(::Type{T}, K::Integer, n::Integer; kwargs...) where {T <: Real} =
    plan_nufft(T, K, (Int(n),); kwargs...)
