"""
    fouriermap(m, x)

Create a Fourier or visibility map of a model `m`
where the image is specified in the image domain by the
pixel locations `x` and `y`
"""
function fouriermap(m::M, dims::ComradeBase.AbstractGrid) where {M}
    fouriermap(visanalytic(M), m, dims)
end

function vmap(m::M, grid) where {M}
    return vmap(ispolarized(M), m, grid)
end

function vmap(::IsPolarized, m, grid)
    return map(grid) do p
        T = typeof(p.U)
        v = visibility_point(m, p.U, p.V, zero(T), zero(T))
        return v
    end
end

function vmap(::NotPolarized, m, grid)
    return visibility_point.(Ref(m), grid.U, grid.V, 0, 0)
end


function fouriermap(::IsAnalytic, m, dims::ComradeBase.AbstractGrid)
    X = dims.X
    Y = dims.Y
    uu,vv = uviterator(length(X), step(X), length(Y), step(Y))
    g = RectiGrid((U=uu, V=vv))
    griduv = imagegrid(g)
    vis = vmap(m, griduv)
    return IntensityMap(vis, g)
end


function fouriermap(::NotAnalytic, m, g::ComradeBase.AbstractGrid)
    X = g.X
    Y = g.Y
    img = IntensityMap(zeros(map(length, dims(g))), g)
    mimg = modelimage(m, img, FFTAlg(), DeltaPulse())
    uu,vv = uviterator(length(X), step(X), length(Y), step(Y))
    g = RectiGrid((U=uu, V=vv))
    griduv = imagegrid(g)
    vis = vmap(mimg, griduv)
    return IntensityMap(vis, g)
end
