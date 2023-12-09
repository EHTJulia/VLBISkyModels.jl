"""
    fouriermap(m, x)

Create a Fourier or visibility map of a model `m`
where the image is specified in the image domain by the
pixel locations `x` and `y`
"""
function fouriermap(m::M, dims::ComradeBase.AbstractDims) where {M}
    fouriermap(visanalytic(M), m, dims)
end

function fouriermap(::IsAnalytic, m, dims::ComradeBase.AbstractDims)
    X = dims.X
    Y = dims.Y
    uu,vv = uviterator(length(X), step(X), length(Y), step(Y))
    vis = visibility_point.(Ref(m), uu, vv', 0, 0)
    return KeyedArray(vis; U=uu, V=vv)
end


function fouriermap(::NotAnalytic, m, g::ComradeBase.AbstractDims)
    X = g.X
    Y = g.Y
    img = IntensityMap(zeros(map(length, dims(g))), g)
    mimg = modelimage(m, img, FFTAlg(), DeltaPulse())
    uu,vv = uviterator(length(X), step(X), length(Y), step(Y))
    # uvgrid = ComradeBase.grid(U=uu, V=vv)
    return KeyedArray(visibility_point.(Ref(mimg), uu, vv', 0, 0); U=uu, V=vv)
end
