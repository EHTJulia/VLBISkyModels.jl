export PolySpectral, TaylorSpectral
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
struct PolySpectral{N, P, T <: NTuple{N}, F <: Number, P0} <: ComradeBase.FrequencyParams{P}
    param::P
    index::T
    freq0::F
    p0::P0

    function PolySpectral(param, index::NTuple{N}, freq0::Number, p0 = zero(param)) where {N}
        return new{N, typeof(param), typeof(index), typeof(freq0), typeof(p0)}(
            param, index, freq0, p0
        )
    end

    # PolySpectral inherits parameters from MultiDomainImage: ContinuousImage implementation
    function PolySpectral(index::NTuple{N}, freq0::Number, p0 = 0.0) where {N}
        return new{N, typeof(nothing), typeof(index), typeof(freq0), typeof(p0)}(
            nothing, index, freq0, p0
        )
    end
end

# functionality for PolySpectral to inherit the image parameters from MultiDomainImage
# create array of PolySpectral objects of size and parameters equal to the image
setdomainparam(d::PolySpectral, p) = PolySpectral(p, d.index, d.freq0, d.p0)
function setdomainparam(d::PolySpectral, params::AbstractArray) # spatially varying spectral params
    return map(CartesianIndices(params)) do ind
        PolySpectral(params[ind], _getindices(d.index, ind), d.freq0, d.p0)
    end
end

# version of PolySpectral where index is a single number
# turns index into a tuple so the rest of the code works
function PolySpectral(param, index::Number, freq0::Number, p0 = zero(param)) 
    return PolySpectral(param, (index,), freq0, p0)
end

# version of PolySpectral which is a function that evaluates the expression at a given frequency
(spec::PolySpectral)(p) = build_param(spec, p)

# spectral model expansion
function build_param(spec::PolySpectral{N}, p) where {N}
    x = log(p.Fr/ spec.freq0)
    arg = reduce(+, ntuple(n -> @inbounds(spec.index[n]) * x^n, Val(N)))
    return spec.param .* exp.(arg) .+ spec.p0
end

const TaylorSpectral = PolySpectral