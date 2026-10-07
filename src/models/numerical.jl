# Special because I just want to do the straight FFT thing no matter what
function intensitymap_numeric!(img::IntensityMap, m::AbstractModel)
    grid = axisdims(img)
    griduv = uvgrid(grid)
    vis = allocate_vismap(m, griduv)
    visibilitymap!(vis, m)
    visk = ifftshift(phasedecenter!(baseimage(vis), grid, griduv), 1:2)
    ifft!(visk, 1:2)
    baseimage(img) .= real.(visk)
    return nothing
end

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
    fftshift!(baseimage(vis), fft(baseimage(img), 1:2), 1:2)
    phasecenter!(baseimage(vis), gridxy, grid)
    return nothing
end

function visibilitymap_numeric(m::AbstractModel, grid::AbstractRectiGrid)
    vis = allocate_vismap(m, grid)
    visibilitymap_numeric!(vis, m)
    return vis
end

function intensitymap_numeric(::AbstractModel, ::StructuredDomain)
    throw(
        ArgumentError(
            "StructuredDomain not supported for numeric intensity maps. " *
                "To make this well defined you must first specify a `FourierDualDomain` " *
                "for the grid."
        )
    )
end

function visibilitymap_numeric(::AbstractModel, ::StructuredDomain)
    throw(
        ArgumentError(
            "StructuredDomain not supported for numeric visibility maps. " *
                "To make this well defined you must first specify a `FourierDualDomain` " *
                "for the grid."
        )
    )
end
