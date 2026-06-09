export PolySpectral

@doc """
    PolySpectral(param, index::NTuple{N}, freq0::Number, p0=zero(param))
    Creates a frequency model that expands the parameter in a polynomial series defined by 
        `param * exp(∑ₙ index[n] * log(Fr / freq0)^n)` + p0.
    i.e. an expansion in log(Fr / freq0) where Fr is the frequency of the observation, 
    `freq0` is the reference frequency, `param` is the parameter value at `freq0`,
    e.g. the reference image.
    You can optionally add a constant term `p0` to the expansion that defines the zeroth order term
    or offset.
    The `N` in index defines the order of the polynomial expansion. 
    `N`=1 corresponds to spectral index, `N`=2 corresponds to spectral curvature, etc.
    If `index` is a `<:Number` then the expansion is of order 1.
    """
struct PolySpectral{N, P, T <: NTuple{N}, F <: Number, P0} <: ComradeBase.FrequencyParams{P}
    param::P
    index::T
    freq0::F
    p0::P0

    function PolySpectral(param, index::NTuple{N}, freq0::Number, p0 = zero(param)) where {N}
        return new{N, typeof(param), typeof(index), typeof(freq0), typeof(p0)}(
            param, index,
            freq0, p0
        )
    end
end

# version of PolySpectral where index is a single number
# turns index into a tuple so the rest of the code works
function PolySpectral(param, index::Number, freq0, p0 = zero(param)) 
    return PolySpectral(param, (index,), freq0, p0)
end

# poly spectral reference frequency parameterization: one for each frequency
function build_reference_frequency(model::PolySpectral{N}, freqlist::AbstractVector)
    return ntuple(i -> log(freqlist[i]/model.freq0), Val(N))
end

# spectral model implementation
@fastmath function build_spectral(param, index::NTuple{N}, ref_freq, p0, modeltype::M) where {N, M<:PolySpectral}
    arg = reduce(+, ntuple(n -> @inbounds(index[n]) * ref_freq^n, Val(N)))
    return param * exp(arg) + p0
end

# This allows Julia to do LICM on the inner loop
# @fastmath mylog(x) = x > 0 ? log(x) : NaN