"""
    ReactantNUFFTAlg{T}

Configuration knobs that don't depend on the runtime point distribution.

# Fields
- `eps::T`                — target tolerance (default `1e-6`).
- `sigma::T`              — oversampling factor (default 2.0; 1.25 supported).
- `nspread::Int`          — kernel half-width `w`. If `< 0`, picked from `eps`.
- `chunk_size::Int`       — points per spread chunk. Default 65536.
- `bin_dims::NTuple{D,Int}` or `NTuple{0,Int}` — bin width per dim used
  to define the sort order; `()` means "auto" (heuristic per dim).
"""
Base.@kwdef struct ReactantNUFFTAlg{T <: Real}
    eps::T = T(1.0e-6)
    sigma::T = T(2)
    nspread::Int = -1                  # < 0 ⇒ derive from eps
    chunk_size::Int = 65536
    bin_dims::Tuple = ()               # () ⇒ auto
end

# Convenience constructor with auto-promotion.
function ReactantNUFFTAlg(
        ::Type{T};
        eps::Real = 1.0e-6,
        sigma::Real = 2,
        nspread::Integer = -1,
        chunk_size::Integer = 65536,
        bin_dims::Tuple = (),
    ) where {T <: Real}
    return ReactantNUFFTAlg{T}(
        T(eps), T(sigma), Int(nspread), Int(chunk_size), bin_dims
    )
end
