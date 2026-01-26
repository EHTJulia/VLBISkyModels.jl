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

Note that if the image grid is the same as the grid passed to `intensitymap` then the discrete image 
is used directly for efficiency.
"""
struct ContinuousImage{A <: IntensityMap, P} <: AbstractModel
    """
    Discrete representation of the image.
    """
    img::A
    """
    Image kernel or pulse that transforms from the discrete image to a continuous one.
    """
    kernel::P
end

make_map(cimg::ContinuousImage) = cimg.img

function Base.show(io::IO, img::ContinuousImage{A, P}) where {A, P}
    sA = split("$A", ",")[1]
    sA = sA * "}"
    return print(io, "ContinuousImage{$sA, $P}($(size(img)))")
end

function ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A <: IntensityMap{<:StokesParams}}
    return IsPolarized()
end
function ComradeBase.ispolarized(::Type{<:ContinuousImage{A}}) where {A <: IntensityMap{<:Real}}
    return NotPolarized()
end
ComradeBase.flux(m::ContinuousImage) = flux(make_map(m)) * flux(m.kernel)

function ComradeBase.stokes(cimg::ContinuousImage, v)
    return ContinuousImage(stokes(make_map(cimg), v), cimg.kernel)
end
ComradeBase.centroid(m::ContinuousImage) = centroid(make_map(m))
Base.parent(cimg::ContinuousImage) = make_map(cimg)
Base.length(m::ContinuousImage) = length(make_map(m))
Base.size(m::ContinuousImage) = size(make_map(m))
Base.size(m::ContinuousImage, i::Int) = size(make_map(m), i::Int)
Base.firstindex(m::ContinuousImage) = firstindex(make_map(m))
Base.lastindex(m::ContinuousImage) = lastindex(make_map(m))
Base.eltype(::ContinuousImage{A, P}) where {A, P} = eltype(A)

Base.getindex(img::ContinuousImage, args...) = getindex(make_map(img), args...)
Base.axes(m::ContinuousImage) = axes(make_map(m))
ComradeBase.domainpoints(m::ContinuousImage) = domainpoints(make_map(m))
ComradeBase.axisdims(m::ContinuousImage) = axisdims(make_map(m))

function ContinuousImage(img::IntensityMap, pulse::Pulse)
    return ContinuousImage{typeof(img), typeof(pulse)}(img, pulse)
end

function ContinuousImage(im::AbstractMatrix, g::AbstractRectiGrid, pulse)
    img = IntensityMap(im, g)
    return ContinuousImage(img, pulse)
end

function InterpolatedModel(
        model::ContinuousImage,
        d::FourierDualDomain{
            <:AbstractRectiGrid, <:AbstractSingleDomain,
            <:FFTAlg,
        }
    )
    img = make_map(model) # intensity map
    sitp = build_intermodel(img, forward_plan(d), algorithm(d), model.kernel)
    return InterpolatedModel{typeof(model), typeof(sitp)}(model, sitp)
end

# IntensityMap will obey the Comrade interface. This is so I can make easy models
visanalytic(::Type{<:ContinuousImage}) = NotAnalytic() # not analytic b/c we want to hook into FFT stuff
imanalytic(::Type{<:ContinuousImage}) = IsAnalytic()

radialextent(c::ContinuousImage) = maximum(values(fieldofview(c.img))) / 2

function intensity_point(m::ContinuousImage, p)
    @unpack_params img = m(p)
    dx, dy = pixelsizes(m.img)
    sum = zero(eltype(m.img))
    ms = stretched(m.kernel, dx, dy)
    @inbounds for (I, p0) in pairs(domainpoints(m.img))
        dp = (X = (p.X - p0.X), Y = (p.Y - p0.Y))
        k = intensity_point(ms, dp)
        sum += m.img[I] * k
    end
    return sum
end

function convolved(cimg::ContinuousImage, m::AbstractModel)
    return ContinuousImage(cimg.img, convolved(cimg.kernel, m))
end
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


@inline function visibilitymap_numeric(m::ContinuousImage, grid::FourierDualDomain)
    # We need to make sure that the grid is the same size as the image
    checkgrid(axisdims(m), imgdomain(grid))
    img = make_map(m)
    vis = applyft(forward_plan(grid), img)
    return applypulse!(vis, m.kernel, grid)
end

@inline function visibilitymap_numeric(
        m::ContinuousImage,
        grid::FourierDualDomain{GI, GV, <:FFTAlg}
    ) where {
        GI <: AbstractSingleDomain,
        GV <: AbstractSingleDomain,
    }
    minterp = InterpolatedModel(m, grid)
    return visibilitymap(minterp, visdomain(grid))
end

function applypulse!(vis, pulse, gfour::AbstractFourierDualDomain)
    grid = imgdomain(gfour)
    guv = visdomain(gfour)
    dx, dy = pixelsizes(grid)
    mp = stretched(pulse, dx, dy)
    # we grab the parent array since for some reason Enzyme struggles to see
    # through the broadcast
    pvis = parent(vis)
    dp = domainpoints(guv)
    pvis .*= visibility_point.(Ref(mp), dp)
    # for i in eachindex(pvis, dp)
    #     pvis[i] *= visibility_point(mp, dp[i])
    # end
    # pvis .*= visibility_point.(Ref(mp), dp)
    return vis
end

# function intensitymap_analytic!(img::IntensityMap, m::Union{ContinuousImage, ModifiedModel{<:ContinuousImage}})
#     intensitymap_numeric!(img, m)
#     # guv = uvgrid(axisdims(img))
#     # U = guv.U.*ones(length(guv.V))' |> vec
#     # V = ones(length(guv.U)).*guv.V' |> vec
#     # gfour = FourierDualDomain(g, UnstructuredDomain((;U, V)), FFTAlg())
#     # vis = reshape(parent(visibilitymap_numeric(m, gfour)), size(img))
#     # img .= real.(ifftshift(ifft!(fftshift(conj.(vis)))))
#     return nothing
# end

# function visibilitymap_numeric!(img::IntensityMap, m::ContinuousImage)
#     gfour = FourierDualDomain(axisdims(parent(m)), axisdims(img), FFTAlg())
#     img .= visibilitymap_numeric(m, gfour)
#     return nothing
# end

# Make a special pass through for this as well
function visibilitymap_numeric(
        m::ContinuousImage,
        grid::FourierDualDomain{GI, GV, <:FFTAlg}
    ) where {GI, GV}
    minterp = InterpolatedModel(m, grid)
    return visibilitymap(minterp, visdomain(grid))
end

function checkgrid(imgdims, grid)
    truth = (dims(imgdims) == dims(grid))
    return truth
end
ChainRulesCore.@non_differentiable checkgrid(::Any, ::Any)
EnzymeRules.inactive(::typeof(checkgrid), args...) = nothing

# A special pass through for Modified ContinuousImage
const ScalingTransform = Union{Shift, Renormalize}
function visibilitymap_numeric(
        m::ModifiedModel{M, T},
        p::FourierDualDomain
    ) where {
        M <: ContinuousImage, N,
        T <: NTuple{
            N,
            ScalingTransform,
        },
    }
    ispol = ispolarized(M)
    vbase = visibilitymap_numeric(m.model, p)
    puv = visdomain(p)
    _apply_scaling!(ispol, m.transform, vbase, puv)
    return vbase
end


@inline function _apply_scaling!(mbase, t::Tuple, vbase, p)
    # out = similar(vbase)
    pvbase = baseimage(vbase)
    uc = unitscale(complex(eltype(p.U)), mbase)
    dp = domainpoints(p)
    @inbounds for I in eachindex(pvbase, dp)
        pvbase[I] = last(@inline modify_uv(mbase, t, dp[I], uc)) * pvbase[I]
    end
    # pvbase .= last.(modify_uv.(Ref(mbase), Ref(t), domainpoints(p), Ref(uc))) .* pvbase
    return nothing
end
