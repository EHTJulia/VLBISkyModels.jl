abstract type AbstractNUFTPlan <: AbstractPlan end
abstract type NUFT <: FourierTransform end

"""
    $(TYPEDEF)

Internal type used to store the cache for a non-uniform Fourier transform (NUFT).

The user should instead create this using the [`FourierDualDomain`](@ref) function.
"""
struct NUFTPlan{A,P,M,I,T} <: AbstractNUFTPlan
    alg::A # which algorithm to use
    plan::P #NUFT matrix or plan
    phases::M #FT phases needed to phase center things
    indices::I # imgdomain Ti/Fr indices mapped to visdomain indices
    totalvis::T # Total number of visibility points
end

# We do this for speed an readability since all seems to be very slow 
_compare(nv::NamedTuple{N}, val) where {N} = mapreduce(n -> (nv[n] == val[n]), *, N)

# creates the indexing plan for the multidomain nuft. Returns a tuple with 
# iminds whose elements defines the index in the image domain  
# visinds whose elements are the indices of the visibilities that correspond to that imind  
# The order of data is currently set by imgdomain. 

# The order of data is currently set by imgdomain. 
function plan_indices(imgdomain::AbstractRectiGrid, visdomain::UnstructuredDomain)
    # TODO: Change the ordering so that visdomain is accessed in a constant stride so  
    # we can utilize in-place nuft and save a bunch of allocations
    spatialdims = ComradeBase.dims(imgdomain)[3:end]
    nms = map(name, spatialdims)

    # DimPoints stack overflows for an empty tuple  
    isempty(spatialdims) && return (0, 0)

    itr = pairs(DimPoints(spatialdims))
    T = typeof(first(first(itr)))
    iminds = T[]
    visinds = Vector{Int}[]
    visp = domainpoints(visdomain)
    for (i, vals) in itr
        nv = NamedTuple{nms}(vals)
        push!(iminds, i)
        push!(visinds, findall(p -> _compare(nv, p), visp))
    end
    return iminds, visinds
end

function plan_nuft(alg::NUFT, imagegrid::AbstractRectiGrid,
                   visdomain::UnstructuredDomain, indices)
    # check_image_uv(imagegrid, visdomain) 
    # Check if Ti or Fr in visdomain are subset of imgdomain Ti or Fr if present
    points = domainpoints(visdomain)
    iminds, visinds = indices

    uv = UnstructuredDomain(points[visinds[1]], executor(visdomain), header(visdomain))
    tplan = plan_nuft_spatial(alg, imagegrid, uv)
    plans = Dict{typeof(iminds[1]),typeof(tplan)}()

    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        visind = visinds[i]
        uv = UnstructuredDomain(points[visind], executor(visdomain), header(visdomain))
        plans[imind] = plan_nuft_spatial(alg, imagegrid, uv)
    end
    return plans
end

function create_forward_plan(algorithm::NUFT, imgdomain::AbstractRectiGrid,
                             visdomain::UnstructuredDomain)
    phases = make_phases(algorithm, imgdomain, visdomain)
    indices = plan_indices(imgdomain, visdomain)
    if hasproperty(imgdomain, :Ti) || hasproperty(imgdomain, :Fr)
        plan = plan_nuft(algorithm, imgdomain, visdomain, indices)
    else
        plan = plan_nuft_spatial(algorithm, imgdomain, visdomain)
    end
    return NUFTPlan(algorithm, plan, phases, indices, size(visdomain)[1])
end

function inverse_plan(plan::NUFTPlan)
    return NUFTPlan(plan.alg, plan.plan', inv.(plan.phases), plan.indices, plan.totalvis)
end

function inverse_plan(plan::NUFTPlan{<:FourierTransform,<:AbstractDict})
    iminds, visinds = plan.indices

    inverse_plans_t = plan.plan[iminds[1]]'
    inverse_plans = Dict{typeof(iminds[1]),typeof(inverse_plans_t)}()

    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        inverse_plans[imind] = plan.plan[imind]'
    end

    return NUFTPlan(plan.alg, inverse_plans, inv.(plan.phases), plan.indices, plan.totalvis)
end

@inline function nuft(A, b::IntensityMap)
    return _nuft(A, baseimage(b))
end

@inline function nuft(A, b::IntensityMap{<:StokesParams})
    I = _nuft(A, baseimage(stokes(b, :I)))
    Q = _nuft(A, baseimage(stokes(b, :Q)))
    U = _nuft(A, baseimage(stokes(b, :U)))
    V = _nuft(A, baseimage(stokes(b, :V)))
    return StructArray{StokesParams{eltype(I)}}((; I, Q, U, V))
end

# @inline function nuft(A, b::StokesIntensityMap)
#     I = _nuft(A, parent(stokes(b, :I)))
#     Q = _nuft(A, parent(stokes(b, :Q)))
#     U = _nuft(A, parent(stokes(b, :U)))
#     V = _nuft(A, parent(stokes(b, :V)))
#     return StructArray{StokesParams{eltype(I)}}((; I, Q, U, V))
# end

include(joinpath(@__DIR__, "nfft_alg.jl"))

include(joinpath(@__DIR__, "dft_alg.jl"))
