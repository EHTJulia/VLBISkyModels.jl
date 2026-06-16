export PolySpectral, TaylorSpectral, build_param, build_param!
import ComradeBase: build_param

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
struct PolySpectral{N, P, T <: NTuple{N}, F <: Number, P0} <: ComradeBase.FrequencyParams{Any}
    params::P
    index::T
    freq0::F
    p0::P0
end

# feed in a single params: geometric modeling
function PolySpectral(params::Number, index::NTuple{N}, freq0::Number, p0 = zero(params)) where {N}
    return PolySpectral{N, typeof(params), typeof(index), typeof(freq0), typeof(p0)}(params, index, freq0, p0)
end

# feed in an image for params: imaging
function PolySpectral(params::AbstractArray, index::NTuple{N}, freq0::Number, p0 = zero(params)) where {N}
    return MultiDomainModel(params, PolySpectral(index, freq0, p0))
end

# don't feed in params: imaging
function PolySpectral(index::NTuple{N}, freq0::Number, p0 = 0.) where {N}
    return PolySpectral{N, Nothing, typeof(index), typeof(freq0), typeof(p0)}(nothing, index, freq0, p0)
end

# wrap index in tuples
PolySpectral(params::AbstractArray, index::Number, freq0::Number, p0 = zero(params)) = PolySpectral(params, (index,), freq0, p0)
PolySpectral(params::Number, index::Number, freq0::Number, p0 = zero(params)) = PolySpectral(params, (index,), freq0, p0)
PolySpectral(index::Number, freq0::Number, p0 = 0.0) = PolySpectral((index,), freq0, p0)

# spectral model expansion
function arg_expand(domain::PolySpectral{N}, p) where {N}
    x = log(p.Fr / domain.freq0)
    return reduce(+, ntuple(n -> @inbounds(domain.index[n]) * x^n, Val(N)))
end

# if param doesn't exist and build_param is called on PolySpectral
function ComradeBase.build_param(domain::PolySpectral{N, Nothing}, p) where {N}
    arg = arg_expand(domain, p)
    return exp(arg) + domain.p0 
end

# if param does exist and build_param is called on PolySpectral
function ComradeBase.build_param(domain::PolySpectral{N}, p) where {N}
    return build_param!(domain.params, domain, p)
end

# if unrolled via MultiDomainModel for multidomain imaging
function build_param!(param, domain::PolySpectral{N}, p) where {N}
    arg = arg_expand(domain, p)
    return param .* exp.(arg) .+ domain.p0
end

const TaylorSpectral = PolySpectral