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

    # user includes model parameters in the definition of PolySpectral: intended for geometric model implementation
    function PolySpectral(param, index::NTuple{N}, freq0::Number, p0 = zero(param)) where {N}
        return new{N, typeof(param), typeof(index), typeof(freq0), typeof(p0)}(
            param, index, freq0, p0
        )
    end

    # PolySpectral inherits parameters from MultiDomainImage: intended for ContinuousImage implementation
    function PolySpectral(index::NTuple{N}, freq0::Number, p0 = nothing) where {N}
        return new{N, typeof(nothing), typeof(index), typeof(freq0), typeof(p0)}(
            nothing, index, freq0, p0
        )
    end
end

# function for PolySpectral to inherit the image parameters from MultiDomainImage
setdomainparam(d::PolySpectral, param) = PolySpectral(param, d.index, d.freq0, d.p0)

# version of PolySpectral where index is a single number
# turns index into a tuple so the rest of the code works
function PolySpectral(param, index::Number, freq0, p0 = zero(param)) 
    return PolySpectral(param, (index,), freq0, p0)
end

# version of PolySpectral which evaluates the expression at a given frequency
function (spec::PolySpectral{N})(frtuple::@NamedTuple{Fr::Float64}) where {N}
    ref_freq = build_reference_frequency(spec, frtuple.Fr)
    val = build_spectral(spec.param, spec.index, ref_freq, spec.p0)
    return val
end


# poly spectral reference frequency parameterization: one for each frequency
function build_reference_frequency(model::PolySpectral, freqlist::AbstractVector)
    return map(freq -> log(freq/model.freq0), freqlist)
end

function build_reference_frequency(model::PolySpectral, freq::Number)
    return log(freq/model.freq0)
end

# spectral model expansion
@fastmath function build_spectral(param, index::NTuple{N}, ref_freq, p0) where {N, M<:PolySpectral}
    arg = reduce(+, ntuple(n -> @inbounds(index[n]) * ref_freq^n, Val(N)))
    return param * exp(arg) + p0
end



# use build_param for geometric model implementation
function build_param(spec::PolySpectral, p)
    ref_freq = build_reference_frequency(spec, p.Fr)
    return map(val ->  @inline build_spectral(val, index, ref_freq, spec.p0), spec.param) # dispatch to apply the spectral model
end

const TaylorSpectral = PolySpectral