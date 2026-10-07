export InterpolatedModel

struct InterpolatedModel{M <: AbstractModel, SI} <: AbstractModel
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
function InterpolatedModel(
        model::AbstractModel, grid::AbstractRectiGrid;
        algorithm::FFTAlg = FFTAlg()
    )
    dual = FourierDualDomain(grid, algorithm)
    return InterpolatedModel(model, dual)
end

radialextent(m::InterpolatedModel) = radialextent(m.model)
flux(m::InterpolatedModel) = flux(m.model)

function build_intermodel(img::IntensityMap, plan, alg::FFTAlg, pulse = DeltaPulse())
    vis = applyft(plan, img)
    grid = axisdims(img)
    griduv = build_padded_uvgrid(grid, alg)
    phasecenter!(vis, grid, griduv)
    dx, dy = pixelsizes(grid)
    return create_interpolator(griduv, vis, stretched(pulse, dx, dy))
end

function build_intermodel(img::StokesMap, plan, alg::FFTAlg, pulse = DeltaPulse())
    return StokesInterpolator(
        build_intermodel(stokes(img, :I), plan, alg, pulse),
        build_intermodel(stokes(img, :Q), plan, alg, pulse),
        build_intermodel(stokes(img, :U), plan, alg, pulse),
        build_intermodel(stokes(img, :V), plan, alg, pulse),
    )
end

struct StokesInterpolator{FI, FQ, FU, FV}
    I::FI
    Q::FQ
    U::FU
    V::FV
end
(s::StokesInterpolator)(p) = StokesParams(s.I(p), s.Q(p), s.U(p), s.V(p))

function InterpolatedModel(
        model::AbstractModel,
        d::FourierDualDomain{
            <:AbstractRectiGrid, <:AbstractSingleDomain,
            <:FFTAlg,
        }
    )
    img = intensitymap(model, imgdomain(d))
    sitp = build_intermodel(img, forward_plan(d), algorithm(d))
    return InterpolatedModel{typeof(model), typeof(sitp)}(model, sitp)
end

function intensitymap(m::InterpolatedModel, grid::AbstractRectiGrid)
    return intensitymap(m.model, grid)
end

function intensitymap!(img::IntensityMap, m::InterpolatedModel)
    return intensitymap!(img, m.model)
end

myselect(p, kg) = map(Base.Fix1(getindex, p), kg)

# internal function that creates the interpolator objector to evaluate the FT.
function create_interpolator(g, vis::AbstractArray{<:Complex, N}, pulse) where {N}
    # Construct the interpolator
    itp = RectangleGrid(map(ComradeBase.basedim, dims(g))...)
    kg = keys(g)
    visre = real(vis)
    visim = imag(vis)
    # - sign is because we need to move into the frame of the vertical-horizontal image
    rm = ComradeBase.rotmat(g)'
    return f = let kg = kg, itp = itp, visre = visre, visim = visim, pulse = pulse
        p -> begin
            pl = visibility_point(pulse, p)
            U2 = _rotatex(p.U, p.V, rm)
            V2 = _rotatey(p.U, p.V, rm)
            p2 = merge(p, (; U = U2, V = V2))
            x = SVector{N}(myselect(p2, kg))
            vreal = interpolate(itp, visre, x)
            vimag = interpolate(itp, visim, x)
            return pl * complex(vreal, vimag)
        end
    end
end
