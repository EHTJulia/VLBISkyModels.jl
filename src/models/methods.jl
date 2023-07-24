@inline function _visibility_primitive(::NotAnalytic, mimg, u, v, time, freq)
    return mimg.cache.sitp(u, v)
end

# internal method for computing an image of a non-analytic image model. The
# `executor` if for parallelization but is not used for this method.
function intensitymap_numeric!(img::IntensityMap, m)
    # nx, ny = size(img)
    (;X, Y) = axisdims(img)
    vis = fouriermap(m, axisdims(img))
    U, V = uviterator(length(X), step(X), length(Y), step(Y))
    visk = ifftshift(keyless_unname(phasedecenter!(vis, X, Y, U, V)))
    ifft!(visk)
    img .= real.(visk)
    return img
end

function intensitymap_numeric(m, grid::ComradeBase.AbstractDims)
    img = IntensityMap(zeros(map(length, dims(grid))), grid)
    intensitymap_numeric!(img, m)
    return img
end


# function intensitymap(::NotAnalytic, m, dims)
#     vis = ifftshift(ComradeBase.AxisKeys.keyless_unname(phasedecenter!(fouriermap(m, dims), dims.X, dims.Y)))
#     ifft!(vis)
#     return IntensityMap(real.(vis)./length(vis), dims)
# end
