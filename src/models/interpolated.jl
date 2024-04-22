struct InterpolatedModel{M, SI} <: AbstractModel
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
        img = intensitymap(model, imagedomain(d))
        pimg = padimage(img, algorithm(d))
        vis = applyft(forward_plan(d), pimg)
        (;X, Y) = imagedomain(d)
        (;U, V) = visdomain(d)
        vispc = phasecenter(vis, X, Y, U, V)
        sitp = create_interpolator(U, V, vispc, stretched(pulse(d), step(X), step(Y)))
        return InterpolatedModel{typeof(model), typeof(sitp)}(model, sitp)
end
