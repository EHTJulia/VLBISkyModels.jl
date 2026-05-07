#==============================================================================
NUFFT options. Defaults follow FINUFFT (sigma=2, w from eps via the
log10 heuristic). `chunk_size` and `bin_dims` control the bin-sorted
chunked-scatter strategy.
==============================================================================#


# Per-FINUFFT setup_spreader: w = ceil(log10(1/eps)) + (sigma==2 ? 1 : 2),
# capped at 16, floored at 2.
function _nspread_for_eps(::Type{T}, eps::Real, sigma::Real) where {T <: Real}
    e = max(T(eps), eps_tolerance_floor(T))
    base = ceil(Int, log10(T(1) / e))
    extra = sigma >= 1.99 ? 1 : 2
    return clamp(base + extra, 2, 16)
end

# Smallest tolerance we target without losing all accuracy.
eps_tolerance_floor(::Type{Float32}) = 1.0f-7
eps_tolerance_floor(::Type{Float64}) = 1.0e-13
eps_tolerance_floor(::Type{T}) where {T <: Real} = eps(T)
