struct ReactantNUFFTAlg{T <: Real, B <: Tuple} <: NUFT
    eps::T
    sigma::T
    nspread::Int                  # < 0 ⇒ derive from eps
    chunk_size::Int
    bin_dims::B               # () ⇒ auto
end

"""
    ReactantNUFFTAlg{T}

Uses the Reactant implementation of the NUFFT in the extension. 
Most settings are reasonable defaults, but there are some options that
can be tuned for performance. 

Most people can just call `ReactantNUFFTAlg(T)` to get a good default for their type, 
but the fields are documented below for those who want to tune.

!!! warn
    To use you must first import Reactant. 

# Fields
- `eps::T`                — target tolerance (default `1e-9` if `T` is `Float64`).
- `sigma::T`              — oversampling factor (default 2.0; 1.25 supported).
- `nspread::Int`          — kernel half-width `w`. If `< 0`, picked from `eps`.
- `chunk_size::Int`       — points per spread chunk. Default 65536.
- `bin_dims::NTuple{D,Int}` or `NTuple{0,Int}` — bin width per dim used
  to define the sort order; `()` means "auto" (heuristic per dim).
"""
function ReactantNUFFTAlg(
        ::Type{T};
        eps::Real = T isa Float64 ? 1.0e-9 : 1.0e-6,  # default tighter for Float64
        sigma::Real = 2,
        nspread::Integer = -1,
        chunk_size::Integer = 65536,
        bin_dims::B = (),
    ) where {T <: Real, B <: Tuple}
    return ReactantNUFFTAlg{T, B}(
        T(eps), T(sigma), Int(nspread), Int(chunk_size), bin_dims
    )
end
ReactantNUFFTAlg(; kwargs...) = ReactantNUFFTAlg(Float64; kwargs...)
