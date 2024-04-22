export modelimage

abstract type AbstractModelImage{M} <: ComradeBase.AbstractModel end

using Enzyme: EnzymeRules

# Now we define a bunch of getters and all set them to be non-differentiable
# since they should all be static
getplan(m::ModelImage{M, <:NUFTCache}) where {M} = m.cache.plan
EnzymeRules.inactive(::typeof(getplan), args...) = nothing
ChainRulesCore.@non_differentiable getplan(m)

getgrid(m::ModelImage) = m.cache.grid
ChainRulesCore.@non_differentiable getgrid(m::ModelImage)
EnzymeRules.inactive(::typeof(getgrid), args...) = nothing


getphases(m::ModelImage{M, <:NUFTCache}) where {M} = m.cache.phases
EnzymeRules.inactive(::typeof(getphases), args...) = nothing


@inline function visibility_point(mimg::ModelImage{M,<:FFTCache}, u, v, time, freq) where {M}
    return mimg.model.sitp(u, v)
end



ChainRulesCore.@non_differentiable checkuv(alg, u::AbstractArray, v::AbstractArray)
EnzymeRules.inactive(::typeof(checkuv), args...) = nothing

function applyft(p::NUFTPlan, img::AbstractArray)
    vis =  nuft(getplan(p), img)
    return conj.(vis).*getphases(m)
end
