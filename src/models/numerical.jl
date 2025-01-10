function _fft(img::AbstractArray{<:StokesParams{<:Real}})
    vI = complex(stokes(img, :I))
    vQ = complex(stokes(img, :Q))
    vU = complex(stokes(img, :U))
    vV = complex(stokes(img, :V))
    p = plan_fft!(vI, 1:2)
    p * vI
    p * vQ
    p * vU
    p * vV
    return StructArray{StokesParams{eltype(I)}}((vI, vQ, vU, vV))
end

function _fft(img::AbstractArray{<:Real})
    vI = complex(img)
    fft!(vI, 1:2)
    return vI
end

function AbstractFFTs.ifft!(vis::AbstractArray{<:StokesParams}, region=1:ndims(vis))
    vI = stokes(vis, :I)
    vQ = stokes(vis, :Q)
    vU = stokes(vis, :U)
    vV = stokes(vis, :V)
    p = plan_ifft!(vI, region)
    p * vI
    p * vQ
    p * vU
    p * vV
    return StructArray{StokesParams{eltype(I)}}((vI, vQ, vU, vV))
end

function AbstractFFTs.fftshift(vis::AbstractArray{<:StokesParams}, region=1:ndims(vis))
    vI = stokes(vis, :I)
    vQ = stokes(vis, :Q)
    vU = stokes(vis, :U)
    vV = stokes(vis, :V)
    return StructArray{StokesParams{eltype(I)}}((fftshift(vI, region),
                                                 fftshift(vQ, region),
                                                 fftshift(vU, region),
                                                 fftshift(vV, region)))
end

# Special because I just want to do the straight FFT thing no matter what
function intensitymap_numeric!(img::IntensityMap, m::AbstractModel)
    grid = axisdims(img)
    griduv = uvgrid(grid)
    # We do this so the array isn't a StructArray
    vis = allocate_vismap(m, griduv)
    visibilitymap!(vis, m)
    visk = ifftshift(parent(phasedecenter!(vis, grid, griduv)), 1:2)
    ifft!(visk, 1:2)
    img .= real.(visk)
    return nothing
end

# function intensitymap_numeric!(vis::IntensityMap{<:StokesParams}, m::AbstractModel)
#     intensitymap_numeric!(stokes(vis, :I), m)
#     intensitymap_numeric!(stokes(vis, :Q), m)
#     intensitymap_numeric!(stokes(vis, :U), m)
#     intensitymap_numeric!(stokes(vis, :V), m)
#     return nothing
# end

function intensitymap_numeric(m::AbstractModel, grid::AbstractSingleDomain)
    img = allocate_imgmap(m, grid)
    intensitymap_numeric!(img, m)
    return img
end

# Special because I just want to do the straight FFT thing no matter what
function visibilitymap_numeric!(vis::IntensityMap, m::AbstractModel)
    grid = axisdims(vis)
    gridxy = xygrid(grid)
    img = allocate_imgmap(m, gridxy)
    intensitymap!(img, m)
    tildeI = _fft(parent(img))
    baseimage(vis) .= fftshift(tildeI, 1:2)
    phasecenter!(vis, gridxy, grid)
    return nothing
end

function visibilitymap_numeric(m::AbstractModel, grid::AbstractRectiGrid)
    vis = allocate_vismap(m, grid)
    visibilitymap_numeric!(vis, m)
    return vis
end

function intensitymap_numeric(::AbstractModel, ::UnstructuredDomain)
    throw(ArgumentError("UnstructuredDomain not supported for numeric intensity maps." *
                        "To make this well defined you must first specify a [`FourierDualDomain`](@ref)" *
                        "for the grid."))
end

function visibilitymap_numeric(::AbstractModel, ::UnstructuredDomain)
    throw(ArgumentError("UnstructuredDomain not supported for numeric intensity maps." *
                        "To make this well defined you must first specify a [`FourierDualDomain`](@ref)" *
                        "for the grid."))
end
