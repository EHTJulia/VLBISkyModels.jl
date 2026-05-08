#==============================================================================
Public-facing dispatch and convenience wrappers.
==============================================================================#

# --------------------- direct DFT ground truth (for tests) ------------------

# Centered mode axis: for nmodes = N, indices m ∈ {-N/2, ..., N/2 - 1}.
@inline _centered_mode_axis(n::Int) = (-fld(n, 2)):(cld(n, 2) - 1)

"""
    direct_type1(points, c, nmodes; iflag=-1) -> fk

Brute-force `O(M * prod(nmodes))` reference for tests.
"""
function direct_type1(
        points::NTuple{D, AbstractVector},
        c::AbstractVector,
        nmodes::NTuple{D, Int};
        iflag::Integer = -1,
    ) where {D}
    M = length(c)
    realT = real(eltype(points[1]))
    period = realT(2 * pi)
    phase = zeros(realT, M, nmodes...)
    ones_tail = ntuple(_ -> 1, Val(D))
    for d in 1:D
        x = reshape(mod.(points[d], period), M, ones_tail...)
        md = reshape(
            realT.(_centered_mode_axis(nmodes[d])),
            ntuple(i -> i == d + 1 ? nmodes[d] : 1, Val(D + 1)),
        )
        phase = phase .+ x .* md
    end
    kernel = cis.(realT(iflag) .* phase)
    cv = reshape(c, M, ones_tail...)
    return dropdims(sum(cv .* kernel; dims = 1); dims = 1)
end

"""
    direct_type2(points, fk; iflag=-1) -> c

Brute-force `O(M * prod(size(fk)))` reference for tests.
"""
function direct_type2(
        points::NTuple{D, AbstractVector},
        fk::AbstractArray;
        iflag::Integer = -1,
    ) where {D}
    nmodes = ntuple(d -> size(fk, d), D)
    M = length(points[1])
    realT = real(eltype(points[1]))
    period = realT(2 * pi)
    phase = zeros(realT, M, nmodes...)
    ones_tail = ntuple(_ -> 1, Val(D))
    for d in 1:D
        x = reshape(mod.(points[d], period), M, ones_tail...)
        md = reshape(
            realT.(_centered_mode_axis(nmodes[d])),
            ntuple(i -> i == d + 1 ? nmodes[d] : 1, Val(D + 1)),
        )
        phase = phase .+ x .* md
    end
    kernel = cis.(realT(iflag) .* phase)
    fkv = reshape(fk, 1, size(fk)...)
    rdims = ntuple(i -> i + 1, Val(D))
    return dropdims(sum(fkv .* kernel; dims = rdims); dims = rdims)
end

# --------------------- execute dispatch -------------------------------------

"""
    execute_nufft(prep::NUFFTSetPts, data) -> result

Dispatch on the plan's transform type. Body uses `@opcall` so it must be
called inside a Reactant trace — wrap your call in `Reactant.@jit` /
`Reactant.@compile`.
"""
function execute_nufft(prep::NUFFTSetPts, data::AbstractArray)
    K = nufft_type(prep.plan)
    if K == 1
        return execute_type1(prep, data)
    elseif K == 2
        return execute_type2(prep, data)
    else
        error("Unsupported NUFFT type $K")
    end
end

"""
    execute_nufft!(out, prep::NUFFTSetPts, data) -> out

In-place version of `execute_nufft`. Use this if you want to reuse the same output array for multiple calls.
"""
function execute_nufft!(out::AbstractArray, prep::NUFFTSetPts, data::AbstractArray)
    tmp = execute_nufft(prep, data)
    copyto!(out, tmp)
    return out
end

# --------------------- convenience wrappers ---------------------------------

# 1D varargs convenience
function _to_points_tuple(::Val{D}, x::AbstractVector) where {D}
    @assert D == 1 "Single coordinate vector but plan dimensionality is $D"
    return (x,)
end
_to_points_tuple(::Val{D}, xs::NTuple{D, AbstractVector}) where {D} = xs

# Real eltype helper that handles both Concrete/TracedRArrays and plain arrays.
@inline _real_eltype(x::AbstractArray) = real(Reactant.unwrapped_eltype(eltype(x)))

"""
    nufft_type1(points, c, nmodes; iflag=-1, kwargs...) -> fk

One-shot Type-1 NUFFT convenience wrapper: builds a plan host-side, calls
`set_nufft_points` (traceable), then `execute_nufft` (traceable). Pure Julia
— wrap the call in `Reactant.@jit` to compile + execute, e.g.

    fk = Reactant.@jit nufft_type1(points, c, nmodes; eps=1e-6)

For repeated calls with the same point set, build the plan and
`set_nufft_points` outside the trace and reuse the prep:

    plan = plan_nufft(T, 1, nmodes; eps=1e-6)
    prep = Reactant.@jit set_nufft_points(plan, points)
    fk1 = Reactant.@jit execute_nufft(prep, c1)
    fk2 = Reactant.@jit execute_nufft(prep, c2)
"""
function nufft_type1(
        points::NTuple{D, AbstractVector},
        c::AbstractArray,
        nmodes::NTuple{D, Integer};
        iflag::Integer = -1,
        kwargs...,
    ) where {D}
    @assert size(c, 1) == length(points[1]) "Strength count must match number of points"
    T = _real_eltype(points[1])
    plan = plan_nufft(T, 1, nmodes; iflag, kwargs...)
    prep = set_nufft_points(plan, points)
    return execute_nufft(prep, c)
end

nufft_type1(x::AbstractVector, c::AbstractArray, n::Integer; kwargs...) =
    nufft_type1((x,), c, (Int(n),); kwargs...)

"""
    nufft_type2(points, fk; iflag=-1, kwargs...) -> c

One-shot Type-2 NUFFT. Same conventions as [`nufft_type1`](@ref): pure
traceable Julia, wrap in `Reactant.@jit` to compile + execute.
"""
function nufft_type2(
        points::NTuple{D, AbstractVector},
        fk::AbstractArray;
        iflag::Integer = -1,
        kwargs...,
    ) where {D}
    nmodes = ntuple(d -> size(fk, d), D)
    T = _real_eltype(points[1])
    plan = plan_nufft(T, 2, nmodes; iflag, kwargs...)
    prep = set_nufft_points(plan, points)
    return execute_nufft(prep, fk)
end

nufft_type2(x::AbstractVector, fk::AbstractVector; kwargs...) = nufft_type2((x,), fk)
