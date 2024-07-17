function ChainRulesCore.rrule(::Type{SA}, t::Tuple) where {SA<:StructArray}
    sa = SA(t)
    pt = ProjectTo(t)
    function _structarray_tuple_pullback(Δ)
        # @info "Δ: " Δ
        ps = map(x -> getproperty(Δ, x), propertynames(Δ))
        return NoTangent(), pt(ps)
    end
    return sa, _structarray_tuple_pullback
end

function ChainRulesCore.rrule(::Type{SA}, t::NamedTuple{Na}) where {Na,SA<:StructArray}
    sa = SA(t)
    pt = ProjectTo(t)
    function _structarray_tuple_pullback(Δd)
        Δ = unthunk(Δd)
        # @info "Δ: " Δ
        ps = getproperty.(Ref(Δ), Na)
        nps = NamedTuple{Na}(ps)
        return NoTangent(), pt(nps)
    end
    return sa, _structarray_tuple_pullback
end

# using StructArrays
function ChainRulesCore.ProjectTo(x::StructArray)
    return ProjectTo{StructArray}(; eltype=eltype(x), names=propertynames(x), dims=size(x))
end

# function extract_components(len, comp, names)
#     map(names) do n
#         c = getproperty(comp, n)
#         typeof(c) <: AbstractZero && return Fill(0, len)
#         return c
#     end
# end

function (project::ProjectTo{StructArray})(dx::StructArray)
    return dx
end

make_tangent(::AbstractZero, dims) = Zeros(dims)
make_tangent(x::AbstractArray, dims) = reshape(x, dims)

function (project::ProjectTo{StructArray})(dx::Tangent)
    # @info typeof(dx.components)
    backing = map(p -> getproperty(dx.components, p), project.names)
    comps = map(x -> make_tangent(x, project.dims), backing)
    return StructArray{project.eltype}(comps)
end

function (project::ProjectTo{StructArray})(dx::AbstractArray)
    @assert project.eltype === eltype(dx) "The eltype of the array is not the same there is an error in a ChainRule"
    r = StructArray(dx)
    return r
end

# function (project::ProjectTo{StructArray})(dx::AbstractArray) where {T}
#     # Extract the properties
#     println(typeof(dx))
#     comps = map(p->getproperty.(dx, Ref(p)), project.names)
#     StructArray{T}(comps)
# end

# function (project::ProjectTo{StructArray})(dx::AbstractArray{<:Tangent})
#     return StructArray{project.eltype}(map(p->getproperty.(dx, p), propertynames(names)))
# end

# function (project::ProjectTo{StructArray{T}})(dx::AbstractZero) where {T}
#     return dx
# end

function _ctsimg_pb(Δ::Tangent, pr)
    pb = pr(Δ.img)
    return NoTangent(), pb, NoTangent()
end

function _ctsimg_pb(Δ::AbstractThunk, pr)
    pb = _ctsimg_pb(unthunk(Δ), pr)
    return pb
end

function ChainRulesCore.rrule(::Type{ContinuousImage}, data::IntensityMapTypes,
                              pulse::Pulse)
    img = ContinuousImage(data, pulse)
    pd = ProjectTo(data)
    function pb(Δ)
        # @info "Input: " Δ.img
        # @info "PT: " pd
        return _ctsimg_pb(Δ, pd)
    end
    return img, pb
end

getm(m::AbstractModel) = m
getm(m::Tuple) = m[2]

function ChainRulesCore.rrule(::typeof(visibilitymap_analytic),
                              m::Union{GeometricModel,PolarizedModel,CompositeModel,
                                       ModifiedModel}, g::AbstractSingleDomain)
    vis = visibilitymap_analytic(m, g)
    function _composite_visibilitymap_analytic_pullback(Δ)
        dg = UnstructuredDomain(map(zero, named_dims(g)); executor=executor(g),
                                header=header(g))

        dvis = UnstructuredMap(similar(parent(vis)), dg)
        dvis .= unthunk(Δ)
        rvis = UnstructuredMap(zero(vis), g)
        d = autodiff(Reverse, visibilitymap_analytic!, Const, Duplicated(rvis, dvis),
                     Active(m))
        dm = getm(d[1])
        tm = __extract_tangent(dm)
        return NoTangent(), tm, Tangent{typeof(g)}(; dims=dims(dvis))
    end

    return vis, _composite_visibilitymap_analytic_pullback
end

function ChainRulesCore.rrule(::typeof(intensitymap_analytic),
                              m::Union{GeometricModel,AbstractImageTemplate,PolarizedModel,
                                       CompositeModel,ModifiedModel},
                              p::ComradeBase.RectiGrid)
    img = intensitymap_analytic(m, p)
    function _composite_intensitymap_analytic_pullback(Δ)
        dimg = zero(img)
        dimg .= unthunk(Δ)
        rimg = zero(img)
        d = autodiff(Reverse, intensitymap_analytic!, Const, Duplicated(rimg, dimg),
                     Active(m))
        dm = getm(d[1])
        tm = __extract_tangent(dm)
        tandp = Tangent{typeof(p)}(; dims=dims(dimg), header=header(dimg))
        return NoTangent(), tm, tandp
    end
    return img, _composite_intensitymap_analytic_pullback
end
