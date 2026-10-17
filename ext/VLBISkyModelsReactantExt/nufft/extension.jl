using VLBISkyModels
using Reactant: unwrapped_eltype
using ReactantCore
using ComradeBase: DD, X, Y, Node

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
# point's index along each plane dim, `1` along a batched one. Without such dims the domain
# is one flat dim of points.
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
    plane = _planeindex(indices, length(visdomain), ndims(imgdomain) - 2)
    plane, batched = _batchplanes(plane, imgdomain, visdomain)
    pts = VLBISkyModels.plan_nuft_spatial(alg, imgdomain, VLBISkyModels._pointlist(visdomain))
    if ReactantCore.within_compile()
        return PlaneStack(pts, Reactant.promote_to(Reactant.TracedRArray{Int, ndims(plane)}, plane), batched)
    end
    return PlaneStack(pts, Reactant.to_rarray(plane), batched)
end

# Each point's index along every plane dim, `(M, K)`.
function _planeindex((iminds, visinds), M, K)
    plane = zeros(Int, M, K)
    for i in eachindex(iminds, visinds)
        plane[visinds[i], :] .= reshape(collect(Tuple(iminds[i])), 1, :)
    end
    return plane
end

function _batchplanes(plane, imgdomain, visdomain)
    pnames = map(DD.name, DD.otherdims(imgdomain, (X, Y)))
    vdims = DD.otherdims(visdomain, Node)
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
    for k in batched
        selectdim(full, ndims(full), k) .= 1
    end
    return full, batched
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

# A plane stack split across devices along its plane dims. The points are grouped by the
# device that holds their plane into blocks padded to one length `Mb` (padding points read
# plane 1 and are discarded), so each device interpolates only its own points. `start` and
# `frac` are laid out as `(Mb, nblocks...)` (see `_interp_blocks`), block dim `i` splitting
# plane dim `batched[i]`, and `order[j]` is the position of point `j` in the flattened blocks.
struct BlockedPlaneStack{P, S, F, O, B}
    plan::P
    start::S
    frac::F
    order::O
    batched::B
end

function VLBISkyModels._nuft(
        p::VLBISkyModels.NUFTPlan{<:VLBISkyModels.ReactantNUFFTAlg, <:BlockedPlaneStack},
        img::Reactant.AnyTracedRArray
    )
    s = VLBISkyModels.getplan(p)
    ri = _interp_blocks(s.plan, _type2_grid(s.plan, complex.(img)), s.start, s.frac, s.batched)
    r = reshape(ri, :, 2)[s.order, :]
    return complex.(r[:, 1], r[:, 2])
end

# Device arrays in a plan must sit on the same mesh as the sharded inputs they meet under `@jit`.
function ComradeBase.shard(d::VLBISkyModels.FourierDualDomain, layout::ComradeBase.ShardLayout)
    s = invoke(ComradeBase.shard, Tuple{ComradeBase.AbstractDualDomain, ComradeBase.ShardLayout}, d, layout)
    rep = Reactant.Sharding.Replicated(layout.mesh)
    s = VLBISkyModels.@set s.plan_forward = _shardplan(VLBISkyModels.forward_plan(s), ComradeBase.imgdomain(s), layout)
    return VLBISkyModels.@set s.plan_reverse = _replicate(VLBISkyModels.reverse_plan(s), rep)
end

_shardplan(p, _, layout) = _replicate(p, Reactant.Sharding.Replicated(layout.mesh))

function _shardplan(p::VLBISkyModels.NUFTPlan{<:VLBISkyModels.ReactantNUFFTAlg, <:PlaneStack}, imgdomain, layout)
    pdims = DD.otherdims(imgdomain, (X, Y))
    nblocks = map(d -> _nshards(layout, DD.name(d)), pdims)
    batched = Tuple(findall(>(1), collect(nblocks)))
    rep = Reactant.Sharding.Replicated(layout.mesh)
    isempty(batched) && return _replicate(p, rep)
    for k in batched
        length(pdims[k]) % nblocks[k] == 0 || throw(
            ArgumentError("the image's `$(DD.name(pdims[k]))` dim has $(length(pdims[k])) planes, which do not split evenly over its $(nblocks[k]) shards")
        )
    end
    pts = VLBISkyModels.getplan(p).points
    plane = _planeindex(VLBISkyModels.getindices(p), pts.M, length(pdims))
    perblock = map(k -> length(pdims[k]) ÷ nblocks[k], 1:length(pdims))
    blockof(j) = CartesianIndex(map(k -> cld(plane[j, k], perblock[k]), batched))
    bshape = map(k -> nblocks[k], batched)
    counts = zeros(Int, bshape)
    for j in axes(plane, 1)
        counts[blockof(j)] += 1
    end
    Mb = maximum(counts)

    base = map(Array, pts.base)
    fracs = map(Array, pts.frac)
    N = pts.plan.ngrid
    start = ones(Int, Mb, bshape..., 3 + length(pdims))
    frac = map(f -> zeros(eltype(f), Mb, bshape...), fracs)
    order = zeros(Int, pts.M)
    L = LinearIndices((Mb, bshape...))
    fill!(counts, 0)
    for j in axes(plane, 1)
        b = blockof(j)
        i = counts[b] += 1
        start[i, b, 1] = mod(base[1][j], N[1]) + 1
        start[i, b, 2] = mod(base[2][j], N[2]) + 1
        for k in axes(plane, 2)
            start[i, b, 3 + k] = mod1(plane[j, k], perblock[k])
        end
        foreach((f, fs) -> f[i, b] = fs[j], frac, fracs)
        order[j] = L[i, b]
    end

    names = map(k -> DD.name(pdims[k]), batched)
    split = Reactant.Sharding.DimsSharding(layout.mesh, Tuple(2:(length(batched) + 1)), map(n -> layout.axes[n], names))
    blocks = BlockedPlaneStack(
        pts.plan, Reactant.to_rarray(start; sharding = split),
        map(f -> Reactant.to_rarray(f; sharding = split), frac),
        Reactant.to_rarray(order; sharding = rep), batched
    )
    p = VLBISkyModels.@set p.plan = blocks
    return VLBISkyModels.@set p.phases = _replicate(p.phases, rep)
end

# The number of devices a dim named `name` is split over.
function _nshards(layout, name)
    haskey(layout.axes, name) || return 1
    axs = layout.axes[name]
    mesh = layout.mesh
    return prod(a -> mesh.axis_sizes[findfirst(==(a), mesh.axis_names)], axs isa Symbol ? (axs,) : axs)
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
