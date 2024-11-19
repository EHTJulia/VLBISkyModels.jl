export InterpolatedModel

struct InterpolatedModel{M<:AbstractModel,SI} <: AbstractModel
    model::M
    sitp::SI
end

@inline visanalytic(::Type{<:InterpolatedModel}) = IsAnalytic()
@inline imanalytic(::Type{<:InterpolatedModel}) = IsAnalytic()
@inline ispolarized(::Type{<:InterpolatedModel{M}}) where {M} = ispolarized(M)

intensity_point(m::InterpolatedModel, p) = intensity_point(m.model, p)
visibility_point(m::InterpolatedModel, p) = m.sitp(p.U, p.V)

function Base.show(io::IO, m::InterpolatedModel)
    return print(io, "InterpolatedModel(", m.model, ")")
end

"""
    $(SIGNATURES)

Computes an representation of the model in the Fourier domain using interpolations and FFTs.
This is useful to construct models that aren't directly representable in the Fourier domain.

# Note
This is mostly used for testing and debugging purposes. In general people should use the
[`FourierDualDomain`](@ref) functionality to compute the Fourier transform of a model.
"""
function InterpolatedModel(model::AbstractModel, grid::AbstractRectiGrid;
                           algorithm::FFTAlg=FFTAlg())
    dual = FourierDualDomain(grid, algorithm)
    return InterpolatedModel(model, dual)
end

radialextent(m::InterpolatedModel) = radialextent(m.model)
flux(m::InterpolatedModel) = flux(m.model)

function build_intermodel(img::IntensityMap, plan, alg::FFTAlg, pulse=DeltaPulse())
    vis = applyft(plan, img)
    grid = axisdims(img)
    griduv = build_padded_uvgrid(grid, alg)
    # phasecenter!(vis, grid, griduv)
    (; X, Y) = grid
    (; U, V) = griduv
    sitp = create_interpolator(U, V, vis, stretched(pulse, step(X), step(Y)))
    return sitp
end

function InterpolatedModel(model::AbstractModel,
                           d::FourierDualDomain{<:AbstractRectiGrid,<:AbstractSingleDomain,
                                                <:FFTAlg})
    img = intensitymap(model, imgdomain(d))
    sitp = build_intermodel(img, forward_plan(d), algorithm(d))
    return InterpolatedModel{typeof(model),typeof(sitp)}(model, sitp)
end

function intensitymap(m::InterpolatedModel, grid::AbstractRectiGrid)
    return intensitymap(m.model, grid)
end

function intensitymap!(img::IntensityMap, m::InterpolatedModel)
    return intensitymap!(img, m.model)
end

# internal function that creates the interpolator objector to evaluate the FT.
function create_interpolator(U, V, vis::AbstractArray{<:Complex}, pulse)
    # Construct the interpolator
    #itp = interpolate(vis, BSpline(Cubic(Line(OnGrid()))))
    #etp = extrapolate(itp, zero(eltype(vis)))
    #scale(etp, u, v)

    p1 = BicubicInterpolator(U, V, real(vis), NoBoundaries())
    p2 = BicubicInterpolator(U, V, imag(vis), NoBoundaries())
    function (u, v)
        pl = visibility_point(pulse, (U=u, V=v))
        return pl * (p1(u, v) + 1im * p2(u, v))
    end
end

function create_interpolator(U, V, vis::StructArray{<:StokesParams}, pulse)
    # Construct the interpolator
    pI_real = BicubicInterpolator(U, V, real(vis.I), NoBoundaries())
    pI_imag = BicubicInterpolator(U, V, imag(vis.I), NoBoundaries())

    pQ_real = BicubicInterpolator(U, V, real(vis.Q), NoBoundaries())
    pQ_imag = BicubicInterpolator(U, V, imag(vis.Q), NoBoundaries())

    pU_real = BicubicInterpolator(U, V, real(vis.U), NoBoundaries())
    pU_imag = BicubicInterpolator(U, V, imag(vis.U), NoBoundaries())

    pV_real = BicubicInterpolator(U, V, real(vis.V), NoBoundaries())
    pV_imag = BicubicInterpolator(U, V, imag(vis.V), NoBoundaries())

    function (u, v)
        pl = visibility_point(pulse, (U=u, V=v))
        return StokesParams(pI_real(u, v) * pl + 1im * pI_imag(u, v) * pl,
                            pQ_real(u, v) * pl + 1im * pQ_imag(u, v) * pl,
                            pU_real(u, v) * pl + 1im * pU_imag(u, v) * pl,
                            pV_real(u, v) * pl + 1im * pV_imag(u, v) * pl)
    end
end
