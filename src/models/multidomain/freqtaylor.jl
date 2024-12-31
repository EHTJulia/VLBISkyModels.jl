export TaylorSpectral

struct TaylorSpectral{N,P,T<:NTuple{N},F<:Real} <: FrequencyParams
    param::P
    index::T
    freq0::F
end

"""
    TaylorSpectral(param, index::NTuple{N}, freq0::Real) -> TaylorSpectral{N}

Creates a frequency model that expands the parameter in a Taylor series defined by 
    `param * exp(∑ₙ index[n] * log(Fr / freq0)^n)`.
i.e. an expansion in log(Fr / freq0) where Fr is the frequency of the observation, 
`freq0` is the reference frequency, `param` is the parameter value at `freq0`.

The `N` in index defines the order of the Taylor expansion. If `index` is a `<:Real`
then the expansion is of order 1.

!!! warn
    This feature is experimental and is not considered part of the public stable API.

"""
TaylorSpectral(param, index::Real, freq0) = TaylorSpectral(param, (index,), freq0)

@fastmath @inline function build_param(model::TaylorSpectral{N}, p) where {N}
    lf = log(p.Fr / model.freq0)
    arg = reduce(+, ntuple(n -> @inbounds(model.index[n]) * lf^n, Val(N)))
    param = model.param * exp(arg)
    return param
end
