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

include("poly_spectral.jl")