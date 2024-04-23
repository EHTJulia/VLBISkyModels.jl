export InterpolatedModel

struct InterpolatedModel{M<:AbstractModel, SI} <: AbstractModel
    model::M
    sitp::SV
end

@inline visanalytic(::Type{<:InterpolatedModel}) = IsAnalytic()
@inline imanalytic(::Type{<:InterpolatedModel})  = IsAnalytic()
@inline ispolarized(::Type{<:InterpolatedModel{M}}) where {M} = ispolarized(M)

intensity_point(m::InterpolatedModel, p) = intensity_point(m.model, p)
visibility_point(m::InterpolatedModel, p) = m.sitp(p.U, p.V)


function Base.show(io::IO, m::InterpolatedModel)
    print(io, "InterpolatedModel(", m.model, ")")
end


function InterpolatedModel(model::AbstractModel, grid::AbstractRectiGrid)
    dual = FourierDualDomain(grid)
    InterpolatedModel(model, dual)
end

function InterpolatedModel(
        model::AbstractModel,
        d::FourierDualDomain{<:AbstractRectiGrid, <:AbstractRectiGrid, <:FFTAlg})
        img = intensitymap(model, imgdomain(d))
        pimg = padimage(img, algorithm(d))
        vis = applyft(forward_plan(d), pimg)
        (;X, Y) = imgdomain(d)
        (;U, V) = visdomain(d)
        sitp = create_interpolator(U, V, vis, stretched(pulse(d), step(X), step(Y)))
        return InterpolatedModel{typeof(model), typeof(sitp)}(model, sitp)
end

# internal function that creates the interpolator objector to evaluate the FT.
function create_interpolator(U, V, vis::AbstractArray{<:Complex}, pulse)
    # Construct the interpolator
    #itp = interpolate(vis, BSpline(Cubic(Line(OnGrid()))))
    #etp = extrapolate(itp, zero(eltype(vis)))
    #scale(etp, u, v)

    p1 = BicubicInterpolator(U, V, real(vis), NoBoundaries())
    p2 = BicubicInterpolator(U, V, imag(vis), NoBoundaries())
    function (u,v)
        pl = visibility_point(pulse, (U=u, V=v))
        #- sign because AIPSCC is 2pi i
        return pl*(p1(u,v) - 1im*p2(u,v))
    end
end

function create_interpolator(U, V, vis::StructArray{<:StokesParams}, pulse)
    # Construct the interpolator
    pI_real = BicubicInterpolator(U, V, real(vis.I), NoBoundaries())
    pI_imag = BicubicInterpolator(U, V, real(vis.I), NoBoundaries())

    pQ_real = BicubicInterpolator(U, V, real(vis.Q), NoBoundaries())
    pQ_imag = BicubicInterpolator(U, V, real(vis.Q), NoBoundaries())

    pU_real = BicubicInterpolator(U, V, real(vis.U), NoBoundaries())
    pU_imag = BicubicInterpolator(U, V, real(vis.U), NoBoundaries())

    pV_real = BicubicInterpolator(U, V, real(vis.V), NoBoundaries())
    pV_imag = BicubicInterpolator(U, V, real(vis.V), NoBoundaries())


    function (u,v)
        pl = visibility_point(pulse, (U=u, V=v))
        #- sign because AIPSCC is 2pi i
        return StokesParams(
            pI_real(u,v)*pl - 1im*pI_imag(u,v)*pl,
            pQ_real(u,v)*pl - 1im*pQ_imag(u,v)*pl,
            pU_real(u,v)*pl - 1im*pU_imag(u,v)*pl,
            pV_real(u,v)*pl - 1im*pV_imag(u,v)*pl,
        )
    end
end
