export dirty_image, dirty_beam, load_clean_components, MultiComponentModel

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
    dirty_image(fov::Real, npix::Int, obs::EHTObservation{T,<:EHTVisibilityDatum}) where T

Computes the dirty image of the complex visibilities assuming a field of view of `fov`
and number of pixels `npix` using the complex visibilities found in the observation `obs`.

The `dirty image` is the inverse Fourier transform of the measured visibilties assuming every
other visibility is zero.

"""
function dirty_image(fov::Real, npix::Int, u::AbstractArray{T}, v::AbstractArray{T}, vis::AbstractArray{Complex{T}}) where {T<:Real}
    # First we double the baselines, i.e. we reflect them and conjugate the measurements
    # This ensures a real NFFT
    u2, v2, vis2 = reflect_vis(obs)

    # Get the number of pixels
    alg = NFFTAlg(u2, v2)
    img = IntensityMap(zeros(npix, npix), imagepixels(fov, fov, npix, npix))
    cache = create_cache(alg, img)
    m = cache.plan'*conj.(vis2)
    return IntensityMap(real.(m)./npix^2, imagepixels(fov, fov, npix, npix))
end


"""
    dirty_beam(fov::Real, npix::Int, obs::EHTObservation{T,<:EHTVisibilityDatum}) where T

Computes the dirty beam of the complex visibilities assuming a field of view of `fov`
and number of pixels `npix` using baseline coverage found in `obs`.

The `dirty beam` is the inverse Fourier transform of the (u,v) coverage assuming every
visibility is unity and everywhere else is zero.

"""
function dirty_beam(fov, npix, u::AbstractArray{T}, v::AbstractArray{T}, vis::AbstractArray{Complex{T}}) where {T<:Real}
    u2, v2, vis2 = reflect_vis(u, v, vis)
    vis2 .= complex(one(T), zero(T))
    return dirty_image(fov, npix, u2, v2, vis2)
end


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
radialextent(m::MultiComponentModel) = maximum(hypot.(m.x, m.y)) + Comrade.radialextent(m.base)

convolved(m1::MultiComponentModel, m2::AbstractModel) = MultiComponentModel(convolved(m1.base, m2), m1.flux, m1.x, m1.y)
convolved(m1::AbstractModel, m2::MultiComponentModel) = convolved(m2, m1)

function intensity_point(m::MultiComponentModel, p)
    s = zero(p.X)
    for i in eachindex(m.x, m.y, m.flux)
        s += Comrade.intensity_point(m[i], p)
    end
    return s
end

function visibility_point(m::MultiComponentModel, x, y, t, f)
    s = zero(x)
    for i in eachindex(m.x, m.y, m.flux)
        s += Comrade.visibility_point(m[i], x, y, t, f)
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
