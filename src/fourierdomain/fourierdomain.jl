export FourierDualDomain

abstract type FourierTransform end

"""
    $(TYPEDEF)

This defines an abstract cache that can be used to
hold or precompute some computations.
"""
abstract type AbstractFourierDualDomain <: AbstractDomain end

forward_plan(g::AbstractFourierDualDomain) = getfield(g, :plan_forward)
reverse_plan(g::AbstractFourierDualDomain) = getfield(g, :plan_reverse)
imgdomain(g::AbstractFourierDualDomain)  = getfield(g, :imgdomain)
visdomain(g::AbstractFourierDualDomain)    = getfield(g, :visdomain)
algorithm(g::AbstractFourierDualDomain)    = getfield(g, :algorithm)


abstract type AbstractPlan end
getplan(p::AbstractPlan) = getfield(p, :plan)
getphases(p::AbstractPlan) = getfield(p, :phases)

function create_plans(algorithm, imgdomain, visdomain, pulse)
    plan_forward = create_forward_plan(algorithm, imgdomain, visdomain, pulse)
    plan_reverse = inverse_plan(plan_forward)
    return plan_forward, plan_reverse
end



struct FourierDualDomain{ID<:AbstractSingleDomain, VD<:AbstractSingleDomain, A<:FourierTransform, PI<:AbstractPlan, PD<:AbstractPlan, P} <: AbstractFourierDualDomain
    imgdomain::ID
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
    Serialization.serialize(s, cache.algorithm)
    Serialization.serialize(s, cache.imgdomain)
    Serialization.serialize(s, cache.visdomain)
    Serialization.serialize(s, cache.pulse)

end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:FourierDualDomain})
    alg = Serialization.deserialize(s)
    imd = Serialization.deserialize(s)
    vid = Serialization.deserialize(s)
    pul = Serialization.deserialize(s)
    return FourierDualDomain(imd, vid, alg, pul)
end

"""
    uviterator(nx, dx, ny dy)

Construct the u,v iterators for the Fourier transform of the image
with pixel sizes `dx, dy` and number of pixels `nx, ny`

If you are extending Fourier transform stuff please use these functions
to ensure that the centroid is being properly computed.
"""
function uviterator(nx, dx, ny, dy)
    U = fftshift(fftfreq(nx, inv(dx)))
    V = fftshift(fftfreq(ny, inv(dy)))
    return (;U, V)
end


"""
    uvgrid(grid::AbstractRectiGrid)

Converts from a image domain recti-grid with spatial dimension (X,Y)
to a Fourier or visibility domain grid with spatial dimensions (U,V).
Note the other dimensions are not changed.

For the inverse see [`xygrid`](@ref)
"""
function uvgrid(grid::AbstractRectiGrid)
    (;X, Y) = grid
    uu,vv = uviterator(length(X), step(X), length(Y), step(Y))
    puv = merge((U=uu, V=vv), delete(named_dims(grid), (:X, :Y)))
    g = RectiGrid(puv; executor=executor(grid), header=header(grid))
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

function xyiterator(nu, du, nv, dv)
    X = ifftfreq(nu, inv(du))
    Y = ifftfreq(nv, inv(dv))
    return (;X, Y)
end

"""
    xygrid(grid::AbstractRectiGrid)

Converts from a visi1bility domain recti-grid with spatial dimensions
(U,V) to a image domain grid with spatial dimensions (X,Y). Note the
other dimensions are not changed.

For the inverse see [`uvgrid`](@ref)

"""
function xygrid(grid::AbstractRectiGrid)
    (;U, V) = grid
    x, y = xyiterator(length(U), step(U), length(V), step(V))
    pxy = merge((X=x, Y=y), delete(named_dims(grid), (:U, :V)))
    g = RectiGrid(pxy, executor(grid), header(grid))
    return g
end


function FourierDualDomain(imgdomain::AbstractSingleDomain, visdomain::AbstractSingleDomain, algorithm, pulse=DeltaPulse())
    plan_forward, plan_reverse = create_plans(algorithm, imgdomain, visdomain, pulse)
    return FourierDualDomain(imgdomain, visdomain, algorithm, plan_forward, plan_reverse, pulse)
end

function create_vismap(arr::AbstractArray, g::AbstractFourierDualDomain)
    return ComradeBase.create_map(arr, visdomain(g))
end

function create_imgmap(arr::AbstractArray, g::AbstractFourierDualDomain)
    return ComradeBase.create_map(arr, imgdomain(g))
end


function visibilitymap_analytic(m::AbstractModel, grid::AbstractFourierDualDomain)
    return visibilitymap_analytic(m, visdomain(grid))
end

function visibilitymap_numeric(m::AbstractModel, grid::AbstractFourierDualDomain)
    img = intensitymap_analytic(m, imgdomain(grid))
    vis = applyft(forward_plan(grid), img)
    return vis
end

function intensitymap_analytic(m::AbstractModel, grid::AbstractFourierDualDomain)
    return intensitymap_analytic(m, imgdomain(grid))
end

function intensitymap_numeric(m::AbstractModel, grid::AbstractFourierDualDomain)
    # This is because I want to make a grid that is the same size as the image
    # so we revert to the standard method and not what ever was cached
    img = intensitymap_numeric(m, imgdomain(grid))
    return img
end



include("fft_alg.jl")
include("nuft/nuft.jl")
