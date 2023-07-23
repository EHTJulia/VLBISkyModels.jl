export load_clean_components, MultiComponentModel

using DelimitedFiles

function reflect_vis(u, v, vis)
    un = copy(u)
    vn = copy(v)
    visn = copy(vis)
    un .= -un
    vn .= -vn
    visn .= conj.(vis)
    u2 = vcat(u, un)
    v2 = vcat(v, vn)
    vis2 = vcat(vis, visn)
    return u2, v2, vis2
end


"""
    MultiComponentModel(beam::AbstractModel, fluxes::AbstractVector, x::AbstractVector, y::AbstractVector)

Build a model with a base model type `beam` where fluxes, x, y corresond to the flux, and positions
of the components. This can be used to easily construct clean like models.
"""
struct MultiComponentModel{M, F, V<:AbstractVector} <: AbstractModel
    base::M
    flux::F
    x::V
    y::V
end

Base.getindex(m::MultiComponentModel, i::Int) = modify(m.base, Shift(m.x[i], m.y[i]), Renormalize(m.flux[i]))

imanalytic(::Type{<:MultiComponentModel{M}}) where {M} =  imanalytic(M)
visanalytic(::Type{<:MultiComponentModel{M}}) where {M} = visanalytic(M)
ispolarized(::Type{<:MultiComponentModel{M}}) where {M} = ispolarized(M)
radialextent(m::MultiComponentModel) = maximum(hypot.(m.x, m.y)) + ComradeBase.radialextent(m.base)

convolved(m1::MultiComponentModel, m2::AbstractModel) = MultiComponentModel(convolved(m1.base, m2), m1.flux, m1.x, m1.y)
convolved(m1::AbstractModel, m2::MultiComponentModel) = convolved(m2, m1)

function intensity_point(m::MultiComponentModel, p)
    s = zero(p.X)
    for i in eachindex(m.x, m.y, m.flux)
        s += ComradeBase.intensity_point(m[i], p)
    end
    return s
end

function visibility_point(m::MultiComponentModel, x, y, t, f)
    s = zero(x)
    for i in eachindex(m.x, m.y, m.flux)
        s += ComradeBase.visibility_point(m[i], x, y, t, f)
    end
    return s
end

function load_clean_components(fname, beam=DeltaPulse())
    !endswith(fname, ".mod") && @warn "File doesn't end with .mod are you sure this is a clean MOD file?"
    f, x, y = open(fname, "r") do io
        out = readdlm(io, comments=true, comment_char='!')
        f = out[:, 1]
        r = μas2rad(out[:, 2])*1000
        θ = out[:, 3]
        x = r.*sind.(θ)
        y = r.*cosd.(θ)
        return f, x, y
    end
    return MultiComponentModel(beam, f, x, y)
end
