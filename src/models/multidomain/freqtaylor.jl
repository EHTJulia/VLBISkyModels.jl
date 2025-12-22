export TaylorSpectral

struct TaylorSpectral{N, P, T <: NTuple{N}, F <: Real, P0} <: FrequencyParams{P}
    param::P
    index::T
    freq0::F
    p0::P0

    @doc """
    TaylorSpectral(param, index::NTuple{N}, freq0::Real, p0=zero(param))

    Creates a frequency model that expands the parameter in a Taylor series defined by 
        `param * exp(∑ₙ index[n] * log(Fr / freq0)^n)` + p0.
    i.e. an expansion in log(Fr / freq0) where Fr is the frequency of the observation, 
    `freq0` is the reference frequency, `param` is the parameter value at `freq0`.
    You can optionally add a constant term `p0` to the expansion that defines the zeroth order term
    or offset.

    The `N` in index defines the order of the Taylor expansion. If `index` is a `<:Real`
    then the expansion is of order 1.
    """
    function TaylorSpectral(param, index::NTuple{N}, freq0::Real, p0 = zero(param)) where {N}
        return new{N, typeof(param), typeof(index), typeof(freq0), typeof(p0)}(
            param, index,
            freq0, p0
        )
    end
end

function TaylorSpectral(param, index::Real, freq0, p0 = zero(param))
    return TaylorSpectral(param, (index,), freq0, p0)
end

@fastmath @inline function ComradeBase.build_param(model::TaylorSpectral{N}, p) where {N}
    lf = log(p.Fr / model.freq0)
    arg = reduce(+, ntuple(n -> @inbounds(model.index[n]) * lf^n, Val(N)))
    param = model.param * exp(arg) + model.p0
    return param
end
