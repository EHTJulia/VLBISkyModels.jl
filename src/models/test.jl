#    `NonAnalyticTest`
#An internal model used primarly for testing. Any model passed to it will be interpreted
#as not having an analytic Fourier transform.
struct NonAnalyticTest{M} <: AbstractModel
    model::M
end

ComradeBase.visanalytic(::Type{<:NonAnalyticTest}) = NotAnalytic()
ComradeBase.ispolarized(::Type{<:NonAnalyticTest{M}}) where {M} = ispolarized(M)

@inline radialextent(m::NonAnalyticTest) = radialextent(m.model)
@inline intensity_point(m::NonAnalyticTest, p) = intensity_point(m.model, p)

flux(m::NonAnalyticTest) = flux(m.model)
