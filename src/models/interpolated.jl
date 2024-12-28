export InterpolatedModel

struct InterpolatedModel{M<:AbstractModel,SI} <: AbstractModel
    model::M
    sitp::SI
end

@inline visanalytic(::Type{<:InterpolatedModel}) = IsAnalytic()
@inline imanalytic(::Type{<:InterpolatedModel}) = IsAnalytic()
@inline ispolarized(::Type{<:InterpolatedModel{M}}) where {M} = ispolarized(M)

intensity_point(m::InterpolatedModel, p) = intensity_point(m.model, p)
visibility_point(m::InterpolatedModel, p) = m.sitp(p)

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
    phasecenter!(vis, grid, griduv)
    dx, dy = pixelsizes(grid)
    sitp = create_interpolator(griduv, vis, stretched(pulse, dx, dy))
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

using NamedTupleTools

myselect(p, kg) = map(k->p[k], kg)

# internal function that creates the interpolator objector to evaluate the FT.
function create_interpolator(g, vis::AbstractArray{<:Complex,N}, pulse) where {N}
    # Construct the interpolator
    #itp = interpolate(vis, BSpline(Cubic(Line(OnGrid()))))
    #etp = extrapolate(itp, zero(eltype(vis)))
    #scale(etp, u, v)
    itp = RectangleGrid(map(ComradeBase.basedim, dims(g))...)
    kg = keys(g)
    visre = real(vis)
    visim = imag(vis)
    f = let kg=kg, itp=itp, visre=visre, visim=visim, pulse=pulse
        p->begin
            pl = visibility_point(pulse, p)
            # xx = select(p, kg)
            x = SVector{N}(myselect(p, kg))
            vreal = interpolate(itp, visre, x)
            vimag = interpolate(itp, visim, x)
            return pl * (vreal + 1im * vimag)    
        end
    end
end

function create_interpolator(g, vis::StructArray{<:StokesParams}, pulse)
    # Construct the interpolator
    itp = RectangleGrid(map(ComradeBase.basedim, dims(g))...)

    vIreal = real(vis.I)
    vIimag = imag(vis.I)

    vQreal = real(vis.Q)
    vQimag = imag(vis.Q)

    vUreal = real(vis.U)
    vUimag = imag(vis.U)

    vVreal = real(vis.V)
    vVimag = imag(vis.V)

    function (p)
        pl = visibility_point(pulse, p)
        x = SVector(values(p))
        return StokesParams(interpolate(itp, vIreal, x) * pl + 1im * interpolate(itp, vIimag, x) * pl,
                            interpolate(itp, vQreal, x) * pl + 1im * interpolate(itp, vQimag, x) * pl,
                            interpolate(itp, vUreal, x) * pl + 1im * interpolate(itp, vUimag, x) * pl,
                            interpolate(itp, vVreal, x) * pl + 1im * interpolate(itp, vVimag, x) * pl)
    end
end
