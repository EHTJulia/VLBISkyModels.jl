export ContinuousImage, spatialdims

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
struct ContinuousImage{A, G, P} <: AbstractModel
    """
    Discrete representation of the image.
    """
    array::A
    """
    grid of the image
    """
    grid::G
    """
    Image Kernel that transforms from the discrete image to a continuous one. This is
    sometimes called a pulse function in `eht-imaging`.
    """
    kernel::P
end

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
ComradeBase.flux(m::ContinuousImage) = flux(IntensityMap(m.array, m.grid)) * flux(m.kernel)

function ComradeBase.stokes(cimg::ContinuousImage, v)
    return ContinuousImage(stokes(parent(cimg), v), cimg.kernel)
end
ComradeBase.centroid(m::ContinuousImage) = centroid(parent(m))
Base.parent(m::ContinuousImage) = m.array
Base.length(m::ContinuousImage) = length(m.grid)
Base.size(m::ContinuousImage) = size(m.grid)
Base.size(m::ContinuousImage, i::Int) = size(m.grid, i::Int)
Base.firstindex(m::ContinuousImage) = firstindex(domainpoints(m.grid))
Base.lastindex(m::ContinuousImage) = lastindex(domainpoints(m.grid))

Base.eltype(::ContinuousImage{A, P}) where {A<:AbstractMatrix, P} = eltype(A)
Base.eltype(::ContinuousImage{A,P}) where {T,A<:DomainParams{T},P} = T

Base.getindex(img::ContinuousImage, args...) = getindex(parent(img), args...)
Base.axes(m::ContinuousImage) = axes(parent(m))
ComradeBase.domainpoints(m::ContinuousImage) = domainpoints(m.grid)
ComradeBase.axisdims(m::ContinuousImage) = m.grid

function ContinuousImage(img::IntensityMap, pulse::Pulse)
    return ContinuousImage{typeof(parent(img)),typeof(axisdims(img)), typeof(pulse)}(parent(img), axisdims(img), pulse)
end

function InterpolatedModel(
        model::ContinuousImage,
        d::FourierDualDomain{
            <:AbstractRectiGrid, <:AbstractSingleDomain,
            <:FFTAlg,
        }
    )
    img = parent(model)
    sitp = build_intermodel(img, forward_plan(d), algorithm(d), model.kernel)
    return InterpolatedModel{typeof(model), typeof(sitp)}(model, sitp)
end

# IntensityMap will obey the Comrade interface. This is so I can make easy models
visanalytic(::Type{<:ContinuousImage}) = NotAnalytic() # not analytic b/c we want to hook into FFT stuff
imanalytic(::Type{<:ContinuousImage}) = IsAnalytic()

radialextent(c::ContinuousImage) = maximum(values(fieldofview(c.img))) / 2

function intensity_point(m::ContinuousImage, p)
    img = parent(m)
    dx, dy = pixelsizes(m.grid)
    T = typeof(build_param(first(img), p))
    sum = zero(T)
    ms = stretched(m.kernel, dx, dy)
    @inbounds for (I, p0) in pairs(domainpoints(m))
        dp = (X=(p.X - p0.X), Y=(p.Y - p0.Y))
        k = intensity_point(ms, dp)
        sum += build_param(m.array[I], p) * k
    end
    return sum
end

function intensitymap_analytic!(img::IntensityMap, m::ContinuousImage{<:DomainParams})
    dft = dims(img)[3:end]
    darr = DimArray(parent(domainpoints(RectiGrid(dft))), dft)
    for TF  in DimIndices(darr)
        p0 = build_param(m.array, darr[TF])
        mtf = ContinuousImage(p0,spatialdims(img), m.kernel)
        intensitymap_analytic!(@view(img[TF]), mtf)
    end
    return nothing
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
    img = IntensityMap(m.array, m.grid)
    vis = applyft(forward_plan(grid), img)
    return applypulse!(vis, m.kernel, grid)
end

@inline function visibilitymap_numeric(m::ContinuousImage{I}, grid::FourierDualDomain) where {D<:DomainParams, I<:AbstractArray{D}}
    checkgrid(axisdims(m), spatialdims(imgdomain(grid)))
    img = IntensityMap(m.array, m.grid)
    gimg = imgdomain(grid)
    mfimg = allocate_imgmap(m, gimg)
    dimp = DimArray(parent(domainpoints(gimg)), dims(gimg))
    @inbounds for i in DimIndices(dimp)
        pfr = dimp[i]
        mfimg[i] = @inline build_param(img[i[1:2]], pfr)
    end
    vis = applyft(forward_plan(grid), mfimg)
    return applypulse!(vis, m.kernel, grid)
end

@inline function visibilitymap_numeric(m::ContinuousImage{D}, grid::FourierDualDomain) where {D<:DomainParams}
    checkgrid(axisdims(m), spatialdims(imgdomain(grid)))
    gimg = imgdomain(grid)
    mfimg = allocate_imgmap(m, gimg)
    dft = dims(gimg)[3:end]
    darr = DimArray(parent(domainpoints(RectiGrid(dft))), dft)
    Threads.@threads for TF in DimIndices(darr)
        pfr = darr[TF]
        build_param!(@view(mfimg[TF]), m.array, pfr)
    end
    vis = applyft(forward_plan(grid), mfimg)
    return applypulse!(vis, m.kernel, grid)
end


@inline function visibilitymap_numeric(m::ContinuousImage,
                                       grid::FourierDualDomain{GI,GV,<:FFTAlg}) where {GI<:AbstractSingleDomain,
                                                                                       GV<:AbstractSingleDomain}
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
    for i in eachindex(pvis, dp)
        pvis[i] *= visibility_point(mp, dp[i])
    end
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

spatialdims(g::AbstractRectiGrid) = rebuild(g; dims=dims(g)[1:2])
spatialdims(img::IntensityMap) = spatialdims(axisdims(img))

function checkgrid(imgdims, grid)
    return !(dims(imgdims) == dims(grid)) &&
        throw(
        ArgumentError(
            "The image dimensions in `ContinuousImage`\n" *
                "and the visibility grid passed to `visibilitymap`\n" *
                "do not match. This is not currently supported."
        )
    )
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
    uc = unitscale(Complex{eltype(p.U)}, mbase)
    dp = domainpoints(p)
    @inbounds for I in eachindex(pvbase, dp)
        pvbase[I] = last(@inline modify_uv(mbase, t, dp[I], uc)) * pvbase[I]
    end
    return nothing
end