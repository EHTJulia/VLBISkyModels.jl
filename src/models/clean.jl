export load_clean_components, MultiComponentModel

using DelimitedFiles

"""
    MultiComponentModel(beam::AbstractModel, fluxes::AbstractVector, x::AbstractVector, y::AbstractVector)

Build a model with a base model type `beam` where fluxes, x, y corresond to the flux, and positions
of the components. This can be used to construct clean like models.
"""
struct MultiComponentModel{M<:AbstractModel, F, V<:AbstractVector} <: AbstractModel
    base::M
    flux::F
    x::V
    y::V
end

flux(m::MultiComponentModel)  = flux(m.base)*sum(m.flux)
radialextent(m::MultiComponentModel) = 2*maximum(x->hypot(x...), zip(m.x, m.y)) + radialextent(m.base)

@inline Base.getindex(m::MultiComponentModel, i::Int) = modify(m.base, Shift(m.x[i], m.y[i]), Renormalize(m.flux[i]))

imanalytic(::Type{<:MultiComponentModel{M}}) where {M} =  imanalytic(M)
visanalytic(::Type{<:MultiComponentModel{M}}) where {M} = visanalytic(M)
ispolarized(::Type{<:MultiComponentModel{M}}) where {M} = ispolarized(M)

convolved(m1::MultiComponentModel, m2::AbstractModel) = MultiComponentModel(convolved(m1.base, m2), m1.flux, m1.x, m1.y)
convolved(m1::AbstractModel, m2::MultiComponentModel) = convolved(m2, m1)

function intensity_point(m::MultiComponentModel, p)
    s = zero(p.X)
    for i in eachindex(m.x, m.y, m.flux)
        s += ComradeBase.intensity_point(m[i], p)
    end
    return s
end

function visibility_point(m::MultiComponentModel, p)
    s = zero(Complex{eltype(p.U)})
    for i in eachindex(m.x, m.y, m.flux)
        s += ComradeBase.visibility_point(m[i], p)
    end
    return s
end

function load_clean_components(fname, beam=DeltaPulse())
    !endswith(fname, ".mod") && @warn "File doesn't end with .mod are you sure this is a clean MOD file?"
    f, x, y = open(fname, "r") do io
        out = readdlm(io, comments=true, comment_char='!')
        f = out[:, 1]
        # components are stored in mas
        r = μas2rad(out[:, 2])*1000
        θ = out[:, 3]
        x = r.*sind.(θ)
        y = r.*cosd.(θ)
        return f, x, y
    end
    return MultiComponentModel(beam, f, x, y)
end

function _visibilitymap_multi!(vis, base, flux, x, y)
    visibilitymap!(vis, MultiComponentModel(base, flux, x, y))
    return nothing
end

function ChainRulesCore.rrule(::typeof(visibilitymap_analytic), m::MultiComponentModel, g::UnstructuredDomain)
    vis = visibilitymap_analytic(m, g)
    function _composite_visibilitymap_analytic_pullback(Δ)
        dg = UnstructuredDomain(map(zero, named_dims(g)); executor=executor(g), header=header(g))

        dvis = UnstructuredMap(similar(parent(vis)), dg)
        dvis .= unthunk(Δ)
        rvis = UnstructuredMap(zero(vis), g)
        df = zero(m.flux)
        dx = zero(m.x)
        dy = zero(m.y)
        if fieldnames(typeof(m)) === ()
            tm = Const(m)
        else
            tm = Active(m)
        end
        d = autodiff(Reverse, _visibilitymap_multi!, Const, Duplicated(rvis, dvis), Const(m.base), Duplicated(m.flux, df), Duplicated(m.x, dx), Duplicated(m.y, dy))
        dm = getm(d[1])
        tangentm = __extract_tangent(dm)
        return NoTangent(), Tangent{typeof(m)}(base=tangentm, flux=df, x=dx, y=dy), Tangent{typeof(g)}(dims=dims(dvis))
    end

    return vis, _composite_visibilitymap_analytic_pullback
end
__extract_tangent(::Nothing) = ZeroTangent()
