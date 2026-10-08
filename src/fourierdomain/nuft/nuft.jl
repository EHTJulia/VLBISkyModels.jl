abstract type AbstractNUFTPlan <: AbstractPlan end
abstract type NUFT <: FourierTransform end

"""
    $(TYPEDEF)

Internal type used to store the cache for a non-uniform Fourier transform (NUFT).

The user should instead create this using the [`FourierDualDomain`](@ref) function.
"""
struct NUFTPlan{A, P, M, I, S} <: AbstractNUFTPlan
    alg::A # which algorithm to use
    plan::P #NUFT matrix or plan
    phases::M #FT phases needed to phase center things
    indices::I # image planes and the flat point indices that use each
    visshape::S # size of the visibility domain
end

getindices(p::NUFTPlan) = getfield(p, :indices)
EnzymeRules.inactive(::typeof(getindices), args...) = nothing

# Plan construction runs once, on the host: the domain with its coordinates copied to `Array`s.
# Reactant arrays cannot be indexed or broadcast outside `@jit`.
_hostdomain(visdomain::StructuredDomain) = DD.rebuild(visdomain; coords = map(Array, ComradeBase.coords(visdomain)))

# A coordinate from `shapedcoords` broadcast up to the full domain shape.
function _fullcoord(c, sz)
    m = Broadcast.materialize(c)
    out = similar(m, sz)
    out .= m
    return out
end

# The image plane of every point of `visdomain`, as `CartesianIndex`es over the non-spatial
# dims of `imgdomain`, with the domain's shape.
function _pointplanes(imgdomain::AbstractRectiGrid, visdomain::StructuredDomain)
    sc = ComradeBase.shapedcoords(_hostdomain(visdomain))
    frames = map(DD.otherdims(imgdomain, (X, Y))) do d
        n = name(d)
        haskey(sc, n) || throw(
            ArgumentError(
                "the image grid has a `$n` dim, but the visibility domain has no `$n` coordinate or dim to match its planes against; domain coordinates are $(keys(sc))"
            )
        )
        return _fullcoord(frameindex(d, Broadcast.materialize(sc[n])), size(visdomain))
    end
    return CartesianIndex.(frames...)
end

# The image planes the points use, and for each the flat indices of its points. A run of
# indices with a constant stride is stored as a range.
function plan_indices(imgdomain::AbstractRectiGrid, visdomain::StructuredDomain)
    isempty(DD.otherdims(imgdomain, (X, Y))) && return (0, 0)
    planes = vec(_pointplanes(imgdomain, visdomain))
    iminds = sort!(unique(planes))
    visinds = map(iminds) do p
        return _asrange(findall(==(p), planes))
    end
    return iminds, visinds
end

function _asrange(inds::Vector{Int})
    length(inds) == 1 && return inds[1]:inds[1]
    step = inds[2] - inds[1]
    all(==(step), diff(inds)) || return inds
    return step == 1 ? (inds[1]:inds[end]) : (inds[1]:step:inds[end])
end

# The points of `visdomain` as one flat list, in the domain's memory order, stored in the
# array type of its baseline coordinates.
_pointlist(visdomain::StructuredDomain{<:Tuple{<:Pt}}) = visdomain
function _pointlist(visdomain::StructuredDomain)
    sc = ComradeBase.shapedcoords(_hostdomain(visdomain))
    sz = size(visdomain)
    cs = ComradeBase.coords(visdomain)
    proto = haskey(cs, :U) ? cs.U : cs.u
    U = _like(proto, vec(_fullcoord(sc.U, sz)))
    V = _like(proto, vec(_fullcoord(sc.V, sz)))
    return UnstructuredDomain((; U, V); executor = executor(visdomain), header = header(visdomain))
end

# The phases built from the points must keep the domain's array type: under Reactant, a host
# `Vector` indexed by a traced index overflows the stack (EnzymeAD/Reactant.jl#3301).
_like(proto, a) = copyto!(similar(proto, eltype(a), size(a)), a)

function _subdomain(visdomain::StructuredDomain, visind)
    U = visdomain.U[visind]
    V = visdomain.V[visind]
    return UnstructuredDomain((; U, V); executor = executor(visdomain), header = header(visdomain))
end

function plan_nuft(
        alg::NUFT, imagegrid::AbstractRectiGrid,
        visdomain::StructuredDomain, indices
    )
    iminds, visinds = indices

    tplan = plan_nuft_spatial(alg, imagegrid, _subdomain(visdomain, visinds[1]))
    plans = Dict{typeof(iminds[1]), typeof(tplan)}()

    for i in eachindex(iminds, visinds)
        plans[iminds[i]] = plan_nuft_spatial(alg, imagegrid, _subdomain(visdomain, visinds[i]))
    end
    return plans
end

function create_forward_plan(
        algorithm::NUFT, imgdomain::AbstractRectiGrid,
        visdomain::StructuredDomain
    )
    pts = _pointlist(visdomain)
    phases = make_phases(algorithm, imgdomain, pts)
    indices = plan_indices(imgdomain, visdomain)
    if isempty(DD.otherdims(imgdomain, (X, Y)))
        plan = plan_nuft_spatial(algorithm, imgdomain, pts)
    else
        plan = plan_nuft(algorithm, imgdomain, pts, indices)
    end
    return NUFTPlan(algorithm, plan, phases, indices, size(visdomain))
end

function inverse_plan(plan::NUFTPlan)
    return NUFTPlan(plan.alg, plan.plan', inv.(plan.phases), plan.indices, plan.visshape)
end

function inverse_plan(plan::NUFTPlan{<:FourierTransform, <:AbstractDict})
    iminds, visinds = plan.indices

    inverse_plans_t = plan.plan[iminds[1]]'
    inverse_plans = Dict{typeof(iminds[1]), typeof(inverse_plans_t)}()

    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        inverse_plans[imind] = plan.plan[imind]'
    end

    return NUFTPlan(plan.alg, inverse_plans, inv.(plan.phases), plan.indices, plan.visshape)
end

function applyft(p::AbstractNUFTPlan, img::AbstractArray)
    vis = nuft(p, img)
    applyphases!(vis, p.phases)
    return _withshape(vis, p.visshape)
end

# `reshape` to a vector's own size still allocates a new array header.
_withshape(vis::AbstractVector, ::Tuple{Int}) = vis
_withshape(vis, sz) = reshape(vis, sz)

function applyft(plan::AbstractNUFTPlan, img::StokesMap)
    vI = applyft(plan, stokes(img, :I))
    vQ = applyft(plan, stokes(img, :Q))
    vU = applyft(plan, stokes(img, :U))
    vV = applyft(plan, stokes(img, :V))
    return _stokesparams(vI, vQ, vU, vV)
end

function applyphases!(vis::AbstractArray, phases::AbstractArray)
    @inbounds begin
        @trace for i in eachindex(vis, phases)
            tmp = rgetindex(vis, i) * rgetindex(phases, i)
            rsetindex!(vis, tmp, i)
        end
    end
    return vis
end

function applyphases!(vis::AbstractArray, phases::Number)
    @inbounds begin
        @trace for i in eachindex(vis)
            tmp = rgetindex(vis, i) * phases
            rsetindex!(vis, tmp, i)
        end
    end
    return vis
end

@inline function nuft(A, b::IntensityMap)
    return _nuft(A, baseimage(b))
end

function _nuft(A::NUFTPlan, b)
    return _nuft(getplan(A), b)
end

vissize(A) = first(size(A))

function _nuft(A, b)
    out = similar(b, eltype(A), vissize(A))
    _nuft!(out, A, b)
    return out
end

# Special overload for multidomain nuft
@inline function _nuft(
        p::NUFTPlan{<:FourierTransform, <:AbstractDict},
        img::AbstractArray{<:Number}
    )
    vis_list = similar(baseimage(img), complex(eltype(img)), prod(p.visshape))
    plans = getplan(p)
    iminds, visinds = getindices(p)
    for i in eachindex(iminds, visinds)
        imind = iminds[i]
        visind = visinds[i]
        length(visind) == 0 && continue
        vis_view = @view(vis_list[visind])

        _nuft!(vis_view, plans[imind], @view(img[:, :, imind]))
    end
    return vis_list
end

function _nuft(A::NUFTPlan, b::AbstractArray{<:ForwardDiff.Dual})
    return _frule_nuft(A, b)
end

function _frule_nuft(A::NUFTPlan, b::AbstractArray{<:ForwardDiff.Dual{T, V, P}}) where {T, V, P}
    # Compute the fft
    p = getplan(A)
    buffer = ForwardDiff.value.(b)
    xtil = p * complex.(buffer)
    out = similar(buffer, complex(ForwardDiff.Dual{T, V, P}))
    # Now take the deriv of nuft
    ndxs = ForwardDiff.npartials(first(b))
    dxtils = ntuple(ndxs) do n
        buffer .= ForwardDiff.partials.(b, n)
        return p * complex.(buffer)
    end
    out = similar(xtil, complex(ForwardDiff.Dual{T, V, P}))
    for i in eachindex(out)
        dual = getindex.(dxtils, i)
        prim = xtil[i]
        red = ForwardDiff.Dual{T, V, P}(real(prim), ForwardDiff.Partials(real.(dual)))
        imd = ForwardDiff.Dual{T, V, P}(imag(prim), ForwardDiff.Partials(imag.(dual)))
        out[i] = Complex(red, imd)
    end
    return out
end

include(joinpath(@__DIR__, "nfft_alg.jl"))

include(joinpath(@__DIR__, "dft_alg.jl"))

include(joinpath(@__DIR__, "finufft.jl"))

include(joinpath(@__DIR__, "nonuniformffts.jl"))

include(joinpath(@__DIR__, "nfft_reactant.jl"))
