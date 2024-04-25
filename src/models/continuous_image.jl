export ContinuousImage

"""
    ContinuousImage{A<:IntensityMap, P} <: AbstractModel
    ContinuousImage(img::Intensitymap, kernel)

The basic continuous image model for VLBISkyModels. This expects a IntensityMap style object as its imag
as well as a image kernel or pulse that allows you to evaluate the image at any image
and visibility location. The image model is

    I(x,y) = ∑ᵢ Iᵢⱼ κ(x-xᵢ, y-yᵢ)

where `Iᵢⱼ` are the flux densities of the image `img` and κ is the intensity function for the
`kernel`.


"""
struct ContinuousImage{A <: IntensityMapTypes, P} <: AbstractModel
    """
    Discrete representation of the image.
    """
    img::A
    """
    Image Kernel that transforms from the discrete image to a continuous one. This is
    sometimes called a pulse function in `eht-imaging`.
    """
    kernel::P
end

function Base.show(io::IO, img::ContinuousImage{A, P}) where {A, P}
    sA = split("$A", ",")[1]
    sA = sA*"}"
    print(io, "ContinuousImage{$sA, $P}($(size(img)))")
end

ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A<:StokesIntensityMap} = IsPolarized()
ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A<:IntensityMap{<:StokesParams}} = IsPolarized()
ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A<:IntensityMap{<:Real}} = NotPolarized()
ComradeBase.flux(m::ContinuousImage) = flux(parent(m))*flux(m.kernel)

ComradeBase.stokes(cimg::ContinuousImage, v) = ContinuousImage(stokes(parent(cimg), v), cimg.kernel)
ComradeBase.centroid(m::ContinuousImage) = centroid(parent(m))
Base.parent(m::ContinuousImage)         = m.img
Base.length(m::ContinuousImage)         = length(parent(m))
Base.size(m::ContinuousImage)           = size(parent(m))
Base.size(m::ContinuousImage, i::Int)   = size(parent(m), i::Int)
Base.firstindex(m::ContinuousImage)     = firstindex(parent(m))
Base.lastindex(m::ContinuousImage)      = lastindex(parent(m))
Base.iterate(m::ContinuousImage)        = iterate(parent(m))
Base.iterate(m::ContinuousImage, state) = iterate(parent(m), state)

Base.IteratorSize(::ContinuousImage{A, P}) where {A,P} = Base.IteratorSize(M)
Base.IteratorEltype(::ContinuousImage{A, P}) where {A,P} = Base.IteratorEltype(M)
Base.eltype(::ContinuousImage{A, P}) where {A,P} = eltype(A)

Base.getindex(img::ContinuousImage, args...) = getindex(parent(img), args...)
Base.axes(m::ContinuousImage) = axes(parent(m))
ComradeBase.domainpoints(m::ContinuousImage) = domainpoints(parent(m))
ComradeBase.named_dims(m::ContinuousImage) = named_dims(parent(m))
ComradeBase.axisdims(m::ContinuousImage) = axisdims(parent(m))

Base.similar(m::ContinuousImage, ::Type{S}, dims) where {S} = ContinuousImage(similar(parent(m), S, dims), m.kernel)

function ContinuousImage(img::IntensityMapTypes, pulse::Pulse)
    return ContinuousImage{typeof(img), typeof(pulse)}(img, pulse)
end


function ContinuousImage(im::AbstractMatrix, g::AbstractRectiGrid, pulse)
    img = IntensityMap(im, g)
    return ContinuousImage(img, pulse)
end

function InterpolatedModel(model::ContinuousImage, d::FourierDualDomain{<:AbstractRectiGrid, <:AbstractSingleDomain, <:FFTAlg})
    img = parent(model)
    sitp = build_intermodel(img, forward_plan(d), algorithm(d), model.kernel)
    return InterpolatedModel{typeof(model), typeof(sitp)}(model, sitp)
end



# IntensityMap will obey the Comrade interface. This is so I can make easy models
visanalytic(::Type{<:ContinuousImage}) = NotAnalytic() # not analytic b/c we want to hook into FFT stuff
imanalytic(::Type{<:ContinuousImage}) = IsAnalytic()

radialextent(c::ContinuousImage) = maximum(values(fieldofview(c.img)))/2

function intensity_point(m::ContinuousImage, p)
    dx, dy = pixelsizes(m.img)
    sum = zero(eltype(m.img))
    ms = stretched(m.kernel, dx, dy)
    @inbounds for (I, p0) in pairs(domainpoints(m.img))
        dp = (X=(p.X - p0.X), Y=(p.Y - p0.Y))
        k = intensity_point(ms, dp)
        sum += m.img[I]*k
    end
    return sum
end

convolved(cimg::ContinuousImage, m::AbstractModel) = ContinuousImage(cimg.img, convolved(cimg.kernel, m))
convolved(cimg::AbstractModel, m::ContinuousImage) = convolved(m, cimg)

# @inline function ModifiedModel(m::ContinuousImage, t::Tuple)
#     doesnot_uv_modify(t) === Static.False() && throw(
#                           ArgumentError(
#                             "ContinuousImage does not support modifying the uv plane."*
#                             "This would require a dynamic grid which is not currently implemented"*
#                             "Transformations like rotations just introduce additional degeneracies,
#                              making imaging more difficult"
#                             ))
#     return ModifiedModel{typeof(m), typeof(t)}(m, t)
# end

function visibilitymap_numeric(m::ContinuousImage, grid::AbstractFourierDualDomain)
    # We need to make sure that the grid is the same size as the image
    checkgrid(axisdims(m), imgdomain(grid))
    img = parent(m)
    vis = applyft(forward_plan(grid), img)
    return applypulse(vis, m.kernel, grid)
end



function applypulse(vis, pulse, gfour::AbstractFourierDualDomain)
    grid = imgdomain(gfour)
    griduv = visdomain(gfour)
    dx, dy = pixelsizes(grid)
    mp = stretched(pulse, dx, dy)
    return vis.*visibility_point.(Ref(mp), domainpoints(griduv))
end

function ChainRulesCore.rrule(::typeof(applypulse), vis, pulse, grid)
    out = applypulse(vis, pulse, grid)
    pv = ProjectTo(vis)
    function __applypulse_pb(Δ)
        griduv = visdomain(grid)
        dx, dy = pixelsizes(imgdomain(grid))
        mp = stretched(pulse, dx, dy)
        Δvis = unthunk(Δ).*conj.(visibility_point.(Ref(mp), domainpoints(griduv)))
        return NoTangent(), pv(Δvis), NoTangent(), NoTangent()
    end
    return out, __applypulse_pb
end


# Make a special pass through for this as well
function visibilitymap_numeric(m::ContinuousImage, grid::FourierDualDomain{GI, GV, <:FFTAlg}) where {GI, GV}
    minterp = InterpolatedModel(m, grid)
    return visibilitymap(minterp, visdomain(grid))
end


function checkgrid(imgdims, grid)
    !(dims(imgdims) == dims(grid)) && throw(ArgumentError(
        "The image dimensions in `ContinuousImage`\n"*
        "and the visibility grid passed to `visibilitymap`\n"*
        "do not match. This is not currently supported."))
end
ChainRulesCore.@non_differentiable checkgrid(::Any, ::Any)
EnzymeRules.inactive(::typeof(checkgrid), args...) = nothing
