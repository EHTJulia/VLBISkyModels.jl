export ReactantNUFFTAlg

struct ReactantNUFFTAlg{T <: Real, B <: Tuple} <: NUFT
    reltol::T
    sigma::T
    nspread::Int                  # < 0 ⇒ derive from reltol
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

The plan's arrays live where the visibility domain's coordinates do: on the device for a
domain moved there with `Reactant.to_rarray`, on the host otherwise.

# Fields
- `reltol::T`             — target relative tolerance (default `1e-9` if `T` is `Float64`).
- `sigma::T`              — oversampling factor (default 2.0; 1.25 supported).
- `nspread::Int`          — kernel half-width `w`. If `< 0`, picked from `reltol`.
- `chunk_size::Int`       — points per spread/interp chunk. Default 131072.
  The type-2 interp processes the M nonuniform points in `ceil(M/chunk_size)`
  chunks via a single traced loop; fewer chunks is faster (less loop overhead)
  at the cost of a larger per-chunk gather buffer, so this trades speed against
  GPU memory. The default keeps M up to ~1.3e5 in a single chunk.
- `bin_dims::NTuple{D,Int}` or `NTuple{0,Int}` — bin width per dim used
  to define the sort order; `()` means "auto" (heuristic per dim).
"""
function ReactantNUFFTAlg(
        ::Type{T};
        reltol::Real = T === Float64 ? 1.0e-9 : 1.0e-6,  # default tighter for Float64
        sigma::Real = 2,
        nspread::Integer = -1,
        chunk_size::Integer = 131072,
        bin_dims::B = (),
    ) where {T <: Real, B <: Tuple}
    return ReactantNUFFTAlg{T, B}(
        T(reltol), T(sigma), Int(nspread), Int(chunk_size), bin_dims
    )
end
ReactantNUFFTAlg(; kwargs...) = ReactantNUFFTAlg(Float64; kwargs...)
