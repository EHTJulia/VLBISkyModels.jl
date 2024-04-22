export FourierDualGrid, DFTAlg

abstract type FourierTransform end

"""
    $(TYPEDEF)

This defines an abstract cache that can be used to
hold or precompute some computations.
"""
abstract type AbstractFourierDualDomain end

forward_plan(g::AbstractFourierDualDomain) = getfield(g, :plan_forward)
reverse_plan(g::AbstractFourierDualDomain) = getfield(g, :plan_reverse)
imagedomain(g::AbstractFourierDualDomain) = getfield(g, :imagedomain)
visdomain(g::AbstractFourierDualDomain) = getfield(g, :visdomain)
algorithm(g::AbstractFourierDualDomain) = getfield(g, :algorithm)


struct FourierDualDomain{ID<:AbstractDomain, VD<:AbstractDomain, A<:FourierTransform, PI<:AbstractPlan, PD<:AbstractPlan, P} <: AbstractFourierDualDomain
    imagedomain::ID
    visdomain::VD
    algorithm::A
    plan_forward::PI
    plan_reverse::PD
    pulse::P
end
pulse(g::FourierDualDomain)  = getfield(g, :pulse)

function Serialization.serialize(s::Serialization.AbstractSerializer, cache::FourierDualDomain)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.alg)
    Serialization.serialize(s, cache.imagedomain)
    Serialization.serialize(s, cache.visdomain)
    Serialization.serialize(s, cache.pulse)

end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:FourierDualDomain})
    alg = Serialization.deserialize(s)
    imd = Serialization.deserialize(s)
    vid = Serialization.deserialize(s)
    pul = Serialization.serialize(s, cache.pulse)
    return FourierDualDomain(imd, vid, alg, pul)
end

"""
    uvgrid(grid::AbstractRectiGrid)

Converts from a image domain recti-grid (X,Y) to a Fourier or
visibility domain grid with spatial dimensions (U,V). Note the
other dimensions are not changed.
"""
function uvgrid(grid::AbstractRectiGrid)
    uu,vv = uviterator(length(X), step(X), length(Y), step(Y))
    puv = merge((U=uu, V=vv), delete(named_dims(grid), (:X, :Y)))
    g = RectiGrid(puv, execute(grid), header(grid))
    return g
end

function ifftfreq(n::Int, fs::Real)
    if iseven(n)
        x = LinRange(-n÷2, n÷2-1, n)*fs/n
        return x .+ step(x)/2
    else
        n = n-1
        x =  LinRange(-n÷2, n÷2, n+1)*fs/(n+1)
        return x
    end
end

function xygrid(grid::AbstractRectiGrid)

end


function FourierDualDomain(imagedomain::AbstractGrid, visdomain::AbstractGrid, algorithm, pulse=DeltaPulse())
    plan_forward, plan_reverse = create_plans(algorithm, imagedomain, visdomain, pulse)
    return FourierDualDomain(imagedomain, visdomain, plan_forward, plan_reverse, algorithm, pulse)
end

function FourierDualDomain(imagedomain::AbstractRectiGrid, pulse=DeltaPulse())
    visd = uvgrid(imagedomain)
    alg = FFTAlg()
    plan_forward, plan_reverse = create_plans(alg, imagedomain, visd, pulse)
    return FourierDualDomain(imagedomain, visdomain, plan_forward, plan_reverse, algorithm, pulse)
end


function visibilitymap_analytic(m::AbstractModel, grid::AbstractFourierDualDomain)
    return visibilitymap_analytic(m, visdomain(grid))
end

function intensitymapmap_analytic(m::AbstractModel, grid::AbstractFourierDualDomain)
    return intensitymap_analytic(m, imagedomain(grid))
end

function visibilitymap_numeric(m::AbstractModel, grid::AbstractFourierDualDomain)
    img = intensitymap_analytic(m, imagedomain(grid))
    vis = applyft(forward_plan(grid), img)
    return vis
end

function intensitymap_numeric(m::AbstractModel, grid::AbstractFourierDualDomain)
    vis = visibilitymap_analytic(m, visdomain(grid))
    img = applyift(reverse_plan(grid), vis)
    return img
end

abstract type AbstractPlan end
getplan(p::AbstractPlan) = getfield(p, :plan)
getphases(p::AbstractPlan) = getfield(p, :phases)

function create_plans(imagedomain, visdomain, algorithm, pulse)
    plan_forward = create_forward_plan(algorithm, imagedomain, visdomain, pulse)
    plan_reverse = inverse_plan(plan_forward)
    return plan_forward, plan_reverse
end


include("fft_alg.jl")
include("nuft/nuft.jl")
