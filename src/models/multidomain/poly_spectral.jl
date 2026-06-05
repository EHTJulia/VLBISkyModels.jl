export PolySpectral

struct PolySpectral{N, P, T <: NTuple{N}, F <: Number, P0} <: ComradeBase.FrequencyParams{P}
    param::P
    index::T
    freq0::F
    p0::P0

    @doc """
    PolySpectral(param, index::NTuple{N}, freq0::Number, p0=zero(param))

    Creates a frequency model that expands the parameter in a polynomial series defined by 
        `param * exp(∑ₙ index[n] * log(Fr / freq0)^n)` + p0.
    i.e. an expansion in log(Fr / freq0) where Fr is the frequency of the observation, 
    `freq0` is the reference frequency, `param` is the parameter value at `freq0`.
    You can optionally add a constant term `p0` to the expansion that defines the zeroth order term
    or offset.

    The `N` in index defines the order of the polynomial expansion. 
    `N`=1 corresponds to spectral index, `N`=2 corresponds to spectral curvature, etc.
    If `index` is a `<:Number` then the expansion is of order 1.
    """
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

### poly spectral specific ###

# spectral model implementation
@fastmath function build_spectral(param, index::NTuple{N}, ref_freq, p0, ::PolySpectral)
    arg = reduce(+, ntuple(n -> @inbounds(index[n]) * ref_freq^n, Val(N)))
    return param * exp(arg) + p0
end




### general to any spectral model ###

# spatially varying spectral parameters
@fastmath @inline function ComradeBase.build_param(model::M, p) where {M<:ComradeBase.FrequencyParams{<:AbstractArray}}
    out = similar(model.param)
    return build_param!(out, model, p)
end

# single value version (constant spectral parameters across the image)
# can skip build_param! and go directly to the spectral model
@fastmath @inline function ComradeBase.build_param(model::M, p) where {M<:ComradeBase.FrequencyParams{<:Int}}
    lf = build_reference_frequency(model, p)
    return build_spectral(model.param, model.index, lf, model.p0, M)
end


# applying the spectral expansion
@fastmath @inline function build_param!(out, model::M, p) where {M<:ComradeBase.FrequencyParams}
    mp = model.param # image parameters
    mp0 = model.p0 # initial spectral parameters
    ref_freq = build_reference_frequency(model, p) # model-specific reference frequency parameterization
    # @trace track_numbers=false: reactant-ification
    @trace track_numbers=false for i in eachindex(out, mp) # calculate one pixel at a time
        index = _getindices(model.index, i) # for each pixel, grab the corresponding spectral parameters
        out[i] = @inline build_spectral(mp[i], index, ref_freq, mp0, M) # dispatch to apply the spectral model
    end
    return out
end

# This allows Julia to do LICM on the inner loop
# @fastmath mylog(x) = x > 0 ? log(x) : NaN

@inline _getindices(index::NTuple{N, <:AbstractArray}, i) where {N} = ntuple(n -> index[n][i], Val(N))
@inline _getindices(index::NTuple, i) = index
