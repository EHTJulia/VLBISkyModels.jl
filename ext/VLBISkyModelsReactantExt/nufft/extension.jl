using VLBISkyModels
using Reactant: unwrapped_eltype
using ReactantCore
using ComradeBase: DD, X, Y, Pt

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
# reads. A dim of the visibility domain that matches an image dim one-to-one (column `j`
# reads plane `j`) is a gather batching dim, listed in `batched` by its position among the
# image's plane dims; `plane` has the domain's shape plus one trailing dim holding the
# point's index along each remaining plane dim. Without such dims the domain is one flat
# dim of points.
struct PlaneStack{P, I, B}
    points::P
    plane::I
    batched::B
end

Base.adjoint(p::PlaneStack) = p

function VLBISkyModels.plan_nuft(
        alg::VLBISkyModels.ReactantNUFFTAlg, imgdomain::ComradeBase.AbstractRectiGrid,
        visdomain::ComradeBase.StructuredDomain, indices
    )
    iminds, visinds = indices
    K = ndims(imgdomain) - 2
    plane = zeros(Int, length(visdomain), K)
    for i in eachindex(iminds, visinds)
        plane[visinds[i], :] .= reshape(collect(Tuple(iminds[i])), 1, :)
    end
    plane, batched = _batchplanes(plane, imgdomain, visdomain)
    pts = VLBISkyModels.plan_nuft_spatial(alg, imgdomain, VLBISkyModels._pointlist(visdomain))
    if ReactantCore.within_compile()
        return PlaneStack(pts, Reactant.promote_to(Reactant.TracedRArray{Int, ndims(plane)}, plane), batched)
    end
    return PlaneStack(pts, Reactant.to_rarray(plane), batched)
end

function _batchplanes(plane, imgdomain, visdomain)
    pnames = map(DD.name, DD.otherdims(imgdomain, (X, Y)))
    vdims = DD.otherdims(visdomain, Pt)
    batched = map(d -> something(findfirst(==(DD.name(d)), pnames), 0), vdims)
    sz = size(visdomain)
    full = reshape(plane, sz..., length(pnames))
    onetoone = all(eachindex(batched)) do i
        k = batched[i]
        k > 0 || return false
        cols = selectdim(full, ndims(full), k)
        return all(I -> cols[I] == I[i + 1], CartesianIndices(cols))
    end
    (isempty(batched) || !onetoone) && return plane, ()
    keep = filter(!in(batched), 1:length(pnames))
    return full[ntuple(_ -> Colon(), length(sz))..., keep], batched
end

function VLBISkyModels._nuft(
        p::VLBISkyModels.NUFTPlan{<:VLBISkyModels.ReactantNUFFTAlg, <:PlaneStack},
        img::Reactant.AnyTracedRArray
    )
    s = VLBISkyModels.getplan(p)
    fk = complex.(img)
    # The result may be a lazy `ReshapedArray`, which is slow to read back from the device.
    out = similar(fk, s.points.M)
    return copyto!(out, execute_type2(s.points, fk, s.plane, s.batched))
end

# Device arrays in a plan must sit on the same mesh as the sharded inputs they meet under `@jit`.
function ComradeBase.shard(d::VLBISkyModels.FourierDualDomain, layout::ComradeBase.ShardLayout)
    s = invoke(ComradeBase.shard, Tuple{ComradeBase.AbstractDualDomain, ComradeBase.ShardLayout}, d, layout)
    rep = Reactant.Sharding.Replicated(layout.mesh)
    s = VLBISkyModels.@set s.plan_forward = _replicate(VLBISkyModels.forward_plan(s), rep)
    return VLBISkyModels.@set s.plan_reverse = _replicate(VLBISkyModels.reverse_plan(s), rep)
end

_replicate(x, _) = x
_replicate(x::Tuple, rep) = map(y -> _replicate(y, rep), x)
_replicate(x::Reactant.ConcreteRArray, rep) = Reactant.to_rarray(Array(x); sharding = rep)
function _replicate(p::VLBISkyModels.NUFTPlan, rep)
    p = VLBISkyModels.@set p.plan = _replicate(p.plan, rep)
    return VLBISkyModels.@set p.phases = _replicate(p.phases, rep)
end
_replicate(p::PlaneStack, rep) = PlaneStack(_replicate(p.points, rep), _replicate(p.plane, rep), p.batched)
function _replicate(p::NUFFTSetPts{T, D}, rep) where {T, D}
    perm, base, frac = _replicate((p.perm, p.base, p.frac), rep)
    return NUFFTSetPts{T, D, typeof(p.plan), typeof(perm), typeof(base), typeof(frac)}(p.plan, p.M, perm, base, frac)
end
