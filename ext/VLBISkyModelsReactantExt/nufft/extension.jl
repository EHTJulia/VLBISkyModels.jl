using VLBISkyModels
using Reactant: unwrapped_eltype
using ReactantCore

function VLBISkyModels._jlnuft!(out, A::NUFFTSetPts, b::Reactant.AnyTracedRArray{<:Real})
    VLBISkyModels._jlnuft!(out, A, complex.(b))
    return nothing
end

function VLBISkyModels._jlnuft!(out, A::NUFFTSetPts, b::Reactant.AnyTracedRArray{<:Complex})
    execute_nufft!(out, A, b)
    return nothing
end

function VLBISkyModels.plan_nuft_spatial(
        alg::VLBISkyModels.ReactantNUFFTAlg, imgdomain::ComradeBase.AbstractRectiGrid, visdomain::ComradeBase.StructuredDomain
    )
    (; U, V) = visdomain
    T = eltype(U)
    dx, dy = pixelsizes(imgdomain)
    rm = ComradeBase.rotmat(imgdomain)'
    # No sign flip because we will use the FINUFFT +1 sign convention
    pl = plan_nufft(unwrapped_eltype(U), 2, size(imgdomain)[1:2]; iflag = +1, opts = alg)
    if ReactantCore.within_compile()
        u = convert(T, 2π) .* VLBISkyModels._rotatex.(U, V, Ref(rm)) .* dx
        v = convert(T, 2π) .* VLBISkyModels._rotatey.(U, V, Ref(rm)) .* dy
        pls = set_nufft_points(pl, (u, v))
    else
        u = @jit convert(T, 2π) .* VLBISkyModels._rotatex.(U, V, Ref(rm)) .* dx
        v = @jit convert(T, 2π) .* VLBISkyModels._rotatey.(U, V, Ref(rm)) .* dy
        pls = @jit set_nufft_points(pl, (u, v))
    end
    return pls
end

Base.adjoint(plan::NUFFTSetPts) = plan # Not needed Reactant is too smart for this
VLBISkyModels.vissize(plan::NUFFTSetPts) = plan.M

# A multi-plane image grid: one point set over every visibility, and the plane each point
# reads (a linear index over the grid's non-spatial dims).
struct PlaneStack{P, I}
    points::P
    plane::I
end

Base.adjoint(p::PlaneStack) = p

function VLBISkyModels.plan_nuft(
        alg::VLBISkyModels.ReactantNUFFTAlg, imgdomain::ComradeBase.AbstractRectiGrid,
        visdomain::ComradeBase.StructuredDomain, indices
    )
    iminds, visinds = indices
    planes = LinearIndices(size(imgdomain)[3:end])
    plane = zeros(Int, length(visdomain))
    for i in eachindex(iminds, visinds)
        plane[visinds[i]] .= planes[iminds[i]]
    end
    pts = VLBISkyModels.plan_nuft_spatial(alg, imgdomain, visdomain)
    if ReactantCore.within_compile()
        return PlaneStack(pts, Reactant.promote_to(Reactant.TracedRArray{Int, 1}, plane))
    end
    return PlaneStack(pts, Reactant.to_rarray(plane))
end

function VLBISkyModels._nuft(
        p::VLBISkyModels.NUFTPlan{<:VLBISkyModels.ReactantNUFFTAlg, <:PlaneStack},
        img::Reactant.AnyTracedRArray
    )
    s = VLBISkyModels.getplan(p)
    fk = complex.(reshape(img, size(img, 1), size(img, 2), :))
    # The result may be a lazy `ReshapedArray`, which is slow to read back from the device.
    out = similar(fk, s.points.M)
    return copyto!(out, execute_type2(s.points, fk, s.plane))
end
