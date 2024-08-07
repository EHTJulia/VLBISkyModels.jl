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
    totalvis:: T # Total number of visibility points
end

# This functions creates a tuple of vectors 1) what indices in imgdomain
# correponds to 2) what all indices in the visdomain 
# The indices tuple will be cached in NUFTPlan
# Functions are overloaded based on the structure of RectiGrid
function plan_indices(imgdomain::AbstractRectiGrid{<:Tuple{X,Y,Ti,Fr}}, visdomain::UnstructuredDomain)
    iminds = Tuple{Int,Int}[] 
    visinds = Vector{Int}[]
    vis_points = domainpoints(visdomain)
    Fr = imgdomain.Fr
    Ti = imgdomain.Ti
    
    for (j, fr) in pairs(Fr) 
        for (i, ti) in pairs(Ti)
            push!(iminds, (i,j))
            push!(visinds, findall(p -> p.Ti==ti && p.Fr==fr, vis_points))
        end
    end

    return (iminds, visinds)
end

function plan_indices(imgdomain::AbstractRectiGrid{<:Tuple{X,Y,Fr,Ti}}, visdomain::UnstructuredDomain)
    iminds = Tuple{Int,Int}[] 
    visinds = Vector{Int}[]
    vis_points = domainpoints(visdomain)
    Ti = imgdomain.Ti
    Fr = imgdomain.Fr

    for (i, ti) in pairs(Ti)
        for (j, fr) in pairs(Fr) 
            push!(iminds, (j,i))
            push!(visinds, findall(p -> p.Ti==ti && p.Fr==fr, vis_points))
        end
    end 
    return (iminds, visinds)
end

function plan_indices(imgdomain::AbstractRectiGrid{<:Tuple{X,Y,Ti}}, visdomain::UnstructuredDomain)
    iminds = Int[]
    visinds = Vector{Int}[]
    vis_points = domainpoints(visdomain)
    Ti = imgdomain.Ti

    for (i, ti) in pairs(Ti)
        push!(iminds, i)
        push!(visinds, findall(p -> p.Ti==ti, vis_points))
    end
    return (iminds, visinds)
end

function plan_indices(imgdomain::AbstractRectiGrid{<:Tuple{X,Y,Fr}}, visdomain::UnstructuredDomain)
    iminds = Int[]
    visinds = Vector{Int}[]
    vis_points = domainpoints(visdomain)
    Fr = imgdomain.Fr

    for (j, fr) in pairs(Fr) 
        push!(iminds, j)
        push!(visinds, findall(p -> p.Fr==fr, vis_points))
    end
    return (iminds, visinds)
end

# Function has been modified to process imgdomain/visdomain with Ti or Fr or both.
# It calls the new plan_nuft  when Ti or Fr is present or else calls
# the old spatial function
function create_forward_plan(algorithm::NUFT, imgdomain::AbstractRectiGrid, visdomain::UnstructuredDomain)
    phases = make_phases(algorithm, imgdomain, visdomain)
    if hasproperty(imgdomain, :Ti) || hasproperty(imgdomain, :Fr) && hasproperty(visdomain, :Ti) || hasproperty(visdomain, :Fr)
        indices = plan_indices(imgdomain, visdomain)
        plans = plan_nuft(algorithm, imgdomain, visdomain, indices)
        return NUFTPlan(algorithm, plans, phases, indices, size(visdomain)[1])
    else
        indices =  Vector{Tuple{Int, Int}}()
        plan = plan_nuft_spatial(algorithm, imgdomain.X, imgdomain.Y, visdomain.U, visdomain.V)
        return NUFTPlan(algorithm, plan, phases, indices, size(visdomain)[1])
    end
end

# Added plan indices and totalvis points to the NUFTPlan in this function
function inverse_plan(plan::NUFTPlan)
    return NUFTPlan(plan.alg, plan.plan', inv.(plan.phases), plan.indices, plan.totalvis)
end

# This a new function is overloaded to handle when NUFTPlan has plans
# as dictionaries in the case of Ti or Fr case
function inverse_plan(plan::NUFTPlan{<:FourierTransform, <:AbstractDict})
    iminds, visinds = plan.indices

    inverse_plans_t = plan.plan[iminds[1]]'
    inverse_plans = Dict{typeof(iminds[1]), typeof(inverse_plans_t)}()

    for i in eachindex(iminds, visinds)
        imind=iminds[i]
        inverse_plans[imind...] = plan.plan[imind...]'
    end

    return NUFTPlan(plan.alg, inverse_plans, inv.(plan.phases), plan.indices, plan.totalvis)
end


@inline function nuft(A, b::AbstractArray)
    return _nuft(A, b)
end

@inline function nuft(A, b::IntensityMap)
    return nuft(A, baseimage(b))
end

@inline function nuft(A, b::AbstractArray{<:StokesParams})
    I = _nuft(A, stokes(b, :I))
    Q = _nuft(A, stokes(b, :Q))
    U = _nuft(A, stokes(b, :U))
    V = _nuft(A, stokes(b, :V))
    return StructArray{StokesParams{eltype(I)}}((;I, Q, U, V))
end

@inline function nuft(A, b::StokesIntensityMap)
    I = _nuft(A, stokes(b, :I))
    Q = _nuft(A, stokes(b, :Q))
    U = _nuft(A, stokes(b, :U))
    V = _nuft(A, stokes(b, :V))
    return StructArray{StokesParams{eltype(I)}}((;I, Q, U, V))
end










include(joinpath(@__DIR__, "nfft_alg.jl"))

include(joinpath(@__DIR__, "dft_alg.jl"))
