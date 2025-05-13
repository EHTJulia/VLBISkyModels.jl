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

getindices(p::NUFTPlan) = getfield(p, :indices)
EnzymeRules.inactive(::typeof(getindices), args...) = nothing

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

    itr = DimPoints(spatialdims)
    visp = domainpoints(visdomain)
    inds = map(eachindex(itr), itr) do i, vals
        nv = NamedTuple{nms}(vals)
        # Check if visinds are strided if so switch to a iterator
        visind = findall(p -> _compare(nv, p), visp)
        dfs = diff(visind)
        if all(==(dfs[1]), dfs)
            if dfs[1] == 1
                # Extract information to let it know we have a contiguous array
                return (i, visind[1]:visind[end])
            else
                return (i, visind[1]:dfs[1]:visind[end])
            end

        else
            return (i, visind)
        end
    end

    iminds = vec(parent(first.(inds)))
    visinds = vec(parent(last.(inds)))

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

function applyft(p::AbstractNUFTPlan, img::AbstractArray)
    vis = nuft(p, img)
    ph = getphases(p)
    for i in eachindex(vis, p)
        vis[i] *= ph[i]
    end
    # vis .*= getphases(p)
    return vis
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

function _nuft(A::NUFTPlan, b::AbstractArray{<:Real})
    return _nuft(getplan(A), b)
end

function _nuft(A, b::AbstractArray{<:Real})
    out = similar(b, eltype(A), size(A)[1])
    _nuft!(out, A, b)
    return out
end

# Special overload for multidomain nuft
@inline function _nuft(p::NUFTPlan{<:FourierTransform,<:AbstractDict},
                       img::AbstractArray{<:Real})
    vis_list = similar(baseimage(img), Complex{eltype(img)}, p.totalvis)
    plans = getplan(p)
    iminds, visinds = getindices(p)
    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        visind = visinds[i]
        vis_view = @view(vis_list[visind])
        _nuft!(vis_view, plans[imind], @view(img[:, :, imind]))
    end
    return vis_list
end

function _nuft(A::NUFTPlan, b::AbstractArray{<:ForwardDiff.Dual})
    return _frule_nuft(A, b)
end

function _frule_nuft(A::NUFTPlan, b::AbstractArray{<:ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # Compute the fft
    p = getplan(A)
    buffer = ForwardDiff.value.(b)
    xtil = p * complex.(buffer)
    out = similar(buffer, Complex{ForwardDiff.Dual{T,V,P}})
    # Now take the deriv of nuft
    ndxs = ForwardDiff.npartials(first(b))
    dxtils = ntuple(ndxs) do n
        buffer .= ForwardDiff.partials.(b, n)
        return p * complex.(buffer)
    end
    out = similar(xtil, Complex{ForwardDiff.Dual{T,V,P}})
    for i in eachindex(out)
        dual = getindex.(dxtils, i)
        prim = xtil[i]
        red = ForwardDiff.Dual{T,V,P}(real(prim), ForwardDiff.Partials(real.(dual)))
        imd = ForwardDiff.Dual{T,V,P}(imag(prim), ForwardDiff.Partials(imag.(dual)))
        out[i] = Complex(red, imd)
    end
    return out
end

include(joinpath(@__DIR__, "nfft_alg.jl"))

include(joinpath(@__DIR__, "dft_alg.jl"))

include(joinpath(@__DIR__, "finufft.jl"))
