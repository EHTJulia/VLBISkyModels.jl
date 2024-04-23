function AbstractFFTs.ifft!(vis::AbstractArray{<:StokesParams}, region)
    vI = stokes(vis, :I)
    vQ = stokes(vis, :Q)
    vU = stokes(vis, :U)
    vV = stokes(vis, :V)
    p = plan_ifft!(vI, region)
    p*vI
    p*vQ
    p*vU
    p*vV
    return StructArray{StokesParams{eltype(I)}}((vI, vQ, vU, vV))
end


function intensitymap_numeric!(img::IntensityMap{T}, m::AbstractModel) where {T<:Real}
    grid = axisdims(img)
    griduv = uvgrid(grid)
    # We do this so the array isn't a StructArray
    vis = allocate_vismap(m, griduv)
    visibilitymap_analytic!(vis, m)
    visk = ifftshift(parent(phasedecenter!(vis, grid, griduv)))
    ifft!(visk)
    img .= real.(visk)
    return nothing
end

function intensitymap_numeric!(vis::IntensityMap{<:StokesParams}, m::AbstractModel)
    intensitymap_numeric!(stokes(vis, :I), m)
    intensitymap_numeric!(stokes(vis, :Q), m)
    intensitymap_numeric!(stokes(vis, :U), m)
    intensitymap_numeric!(stokes(vis, :V), m)
    return nothing
end


function intensitymap_numeric(m::AbstractModel, grid::AbstractRectiGrid)
    img = allocate_vismap(m, grid)
    intensitymap_numeric!(img, m)
    return img
end

function visibilitymap_numeric!(vis::IntensityMap{T}, m::AbstractModel) where {T<:Complex}
    grid = axisdims(vis)
    gridxy = xygrid(grid)
    img = allocate_imgmap(m, gridxy)
    intensitymap!(img, m)
    fft!(img)
    vis .= phasecenter!(fftshift(parent(img)), grid, griduv)
    return nothing
end

function visibilitymap_numeric!(vis::IntensityMap{<:StokesParams}, m::AbstractModel)
    visibilitymap_numeric!(stokes(vis, :I), m)
    visibilitymap_numeric!(stokes(vis, :Q), m)
    visibilitymap_numeric!(stokes(vis, :U), m)
    visibilitymap_numeric!(stokes(vis, :V), m)
    return nothing
end

function visibilitymap_numeric(m::AbstractModel, grid::AbstractRectiGrid)
    vis = allocate_vismap(m, grid)
    visibilitymap_numeric!(vis, m)
    return vis
end

function intensitymap_numeric(::AbstractModel, ::UnstructuredDomain)
    throw(ArgumentError(
        "UnstructuredDomain not supported for numeric intensity maps."*
        "To make this well defined you must first specify a [`FourierDualDomain`](@ref)"*
        "for the grid."
        ))
end

function visibilitymap_numeric(::AbstractModel, ::UnstructuredDomain)
    throw(ArgumentError(
        "UnstructuredDomain not supported for numeric intensity maps."*
        "To make this well defined you must first specify a [`FourierDualDomain`](@ref)"*
        "for the grid."
        ))
end
