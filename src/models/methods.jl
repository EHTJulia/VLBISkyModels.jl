# internal method for computing an image of a non-analytic image model. The
# `executor` if for parallelization but is not used for this method.
function intensitymap_numeric!(img::IntensityMap, m)
    # nx, ny = size(img)
    (;X, Y) = axisdims(img)
    vis = fouriermap(m, axisdims(img))
    U, V = uviterator(length(X), step(X), length(Y), step(Y))
    visk = ifftshift(parent(phasedecenter!(vis, X, Y, U, V)))
    ifft!(visk)
    img .= real.(visk)
    return img
end

function intensitymap_numeric(m, grid::ComradeBase.AbstractDomain)
    img = IntensityMap(zeros(map(length, dims(grid))), grid)
    intensitymap_numeric!(img, m)
    return img
end
