using VLBISkyModels
using Reactant: unwrapped_eltype
using ReactantCore
using ComradeBase: DD, X, Y, Node

function VLBISkyModels._jlnuft!(out, A::NUFFTSetPts, b::Reactant.AnyTracedRArray{<:Real})
    VLBISkyModels._jlnuft!(out, A, complex.(b))
    return nothing
end

function VLBISkyModels._jlnuft!(out, A::NUFFTSetPts, b::Reactant.AnyTracedRArray{<:Complex})
    _checkondevice(A)
    execute_nufft!(out, A, b)
    return nothing
end

# Host plan arrays read while tracing are embedded in the compiled program as constants.
function _checkondevice(pts::NUFFTSetPts)
    first(pts.frac) isa Reactant.AnyTracedRArray || throw(
        ArgumentError(
            "the NUFFT plan is on the host; build the dual domain from a visibility domain moved " *
                "to the device with `Reactant.to_rarray`, or place it with `todevice(domain, layout)`"
        )
    )
    return nothing
end

function VLBISkyModels.plan_nuft_spatial(
        alg::VLBISkyModels.ReactantNUFFTAlg, imgdomain::ComradeBase.AbstractRectiGrid, visdomain::ComradeBase.StructuredDomain
    )
    (; U, V) = visdomain
    dx, dy = pixelsizes(imgdomain)
    rm = ComradeBase.rotmat(imgdomain)'
    # No sign flip because we will use the FINUFFT +1 sign convention
    pl = plan_nufft(unwrapped_eltype(U), 2, size(imgdomain)[1:2]; iflag = +1, opts = alg)
    # The point data live where the coordinates do: on the host for host arrays, so that
    # `todevice` can place them on the devices directly.
    U isa Reactant.AbstractConcreteArray && return @jit _setpoints(pl, U, V, rm, dx, dy)
    return _setpoints(pl, U, V, rm, dx, dy)
end

function _setpoints(pl, U, V, rm, dx, dy)
    twopi = convert(unwrapped_eltype(U), 2π)
    u = twopi .* VLBISkyModels._rotatex.(U, V, Ref(rm)) .* dx
    v = twopi .* VLBISkyModels._rotatey.(U, V, Ref(rm)) .* dy
    return set_nufft_points(pl, (u, v))
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
    elseif pts.frac[1] isa Reactant.AbstractConcreteArray
        return PlaneStack(pts, Reactant.to_rarray(plane), batched)
    end
    return PlaneStack(pts, plane, batched)
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
    _checkondevice(s.points)
    fk = complex.(img)
    # The result may be a lazy `ReshapedArray`, which is slow to read back from the device.
    out = similar(fk, s.points.M)
    return copyto!(out, execute_type2(s.points, fk, s.plane, s.batched))
end

# A plane stack split across devices. Points are laid out as `(local points, blocks...)`,
# block dim `i` splitting plane dim `batched[i]`; `blockshape` and `unperm` map this layout
# back to `visshape`.
struct BlockedPlaneStack{P, S, F, B, Z, Q, V}
    plan::P
    start::S
    frac::F
    batched::B
    blockshape::Z
    unperm::Q
    visshape::V
end

function VLBISkyModels._nuft(
        p::VLBISkyModels.NUFTPlan{<:VLBISkyModels.ReactantNUFFTAlg, <:BlockedPlaneStack},
        img::Reactant.AnyTracedRArray
    )
    s = VLBISkyModels.getplan(p)
    ri = _interp_blocks(s.plan, _type2_grid(s.plan, complex.(img)), s.start, s.frac, s.batched)
    r = permutedims(reshape(ri, s.blockshape..., 2), (s.unperm..., length(s.unperm) + 1))
    r = reshape(r, s.visshape..., 2)
    c = map(_ -> Colon(), s.visshape)
    return complex.(r[c..., 1], r[c..., 2])
end

# The plans are built on the host for the blocked visibility domain and placed by the layout.
function ComradeBase._placederived(d::VLBISkyModels.FourierDualDomain, hostvis, layout::ComradeBase.ShardLayout)
    img = ComradeBase.imgdomain(d)
    vis = ComradeBase.visdomain(d)
    fwd, rev = _shardplans(VLBISkyModels.create_plans(VLBISkyModels.algorithm(d), img, hostvis)..., img, vis, layout)
    d = VLBISkyModels.@set d.plan_forward = fwd
    return VLBISkyModels.@set d.plan_reverse = rev
end

_shardplans(fwd, rev, _, _, layout) = map(p -> _replicate(p, Reactant.Sharding.Replicated(layout.mesh)), (fwd, rev))

# The reverse plan of a plane stack reads the same points as the forward plan, which the two
# placed plans share; only the phases differ.
function _shardplans(
        fwd::VLBISkyModels.NUFTPlan{<:VLBISkyModels.ReactantNUFFTAlg, <:PlaneStack}, rev, imgdomain, vis, layout
    )
    pdims = DD.otherdims(imgdomain, (X, Y))
    names = map(DD.name, pdims)
    nblocks = map(n -> _nshards(layout, n), names)
    if all(==(1), nblocks)
        rep = Reactant.Sharding.Replicated(layout.mesh)
        p = _replicate(fwd, rep)
        return p, VLBISkyModels.@set p.phases = _replicate(rev.phases, rep)
    end
    for k in eachindex(pdims)
        length(pdims[k]) % nblocks[k] == 0 || throw(
            ArgumentError("the image's `$(names[k])` dim has $(length(pdims[k])) planes, which do not split evenly over its $(nblocks[k]) shards")
        )
    end
    vislayout = ComradeBase.executor(vis).layout
    vnames = map(DD.name, dims(vis))
    vsz = size(vis)
    nodeplanes = reverse(filter(k -> nblocks[k] > 1 && !(names[k] in vnames), collect(eachindex(names))))
    isempty(nodeplanes) && haskey(vislayout.axes, :Node) && throw(
        ArgumentError("the layout splits both `Node` and the image's planes; split only the planes, and the points follow them")
    )
    shape = [first(vsz) ÷ prod(k -> nblocks[k], nodeplanes; init = 1); map(k -> nblocks[k], nodeplanes)]
    locals = [1]
    blocks = collect(2:length(shape))
    batched = copy(nodeplanes)
    for (n, len) in zip(Base.tail(vnames), Base.tail(vsz))
        k = findfirst(==(n), names)
        if k === nothing
            haskey(vislayout.axes, n) && throw(
                ArgumentError("the layout splits the visibility domain's `$n` dim, which the image has no planes along; with split image planes, only dims the image shares can be split")
            )
            push!(shape, len)
            push!(locals, length(shape))
        elseif nblocks[k] == 1
            push!(shape, len)
            push!(locals, length(shape))
        else
            len % nblocks[k] == 0 || throw(
                ArgumentError("the visibility domain's `$n` dim has $len values, which do not split evenly over its $(nblocks[k]) shards")
            )
            push!(shape, len ÷ nblocks[k], nblocks[k])
            push!(locals, length(shape) - 1)
            push!(blocks, length(shape))
            push!(batched, k)
        end
    end
    byplane = sortperm(batched)
    perm = (locals..., blocks[byplane]...)
    batched = Tuple(batched[byplane])
    nb = length(batched)

    pts = VLBISkyModels.getplan(fwd).points
    plane = _planeindex(VLBISkyModels.getindices(fwd), pts.M, length(pdims))
    perblock = map((d, n) -> length(d) ÷ n, pdims, nblocks)
    permuted = permutedims(reshape(collect(1:pts.M), shape...), perm)
    point = reshape(permuted, :, size(permuted)[(end - nb + 1):end]...)
    base = pts.base
    fracs = pts.frac
    N = pts.plan.ngrid
    start = similar(point, (size(point)..., 3 + length(pdims)))
    frac = map(f -> similar(f, size(point)), fracs)
    for J in CartesianIndices(point)
        j = point[J]
        start[J, 1] = mod(base[1][j], N[1]) + 1
        start[J, 2] = mod(base[2][j], N[2]) + 1
        start[J, 3] = 1
        for k in eachindex(pdims)
            i = findfirst(==(k), batched)
            start[J, 3 + k] = i === nothing ? plane[j, k] : _withinblock(plane[j, k], perblock[k], J[i + 1], names[k])
        end
        foreach((f, fs) -> f[J] = fs[j], frac, fracs)
    end

    mesh = layout.mesh
    split = Reactant.Sharding.DimsSharding(mesh, Tuple(2:(nb + 1)), map(k -> layout.axes[names[k]], batched))
    stack = BlockedPlaneStack(
        pts.plan, Reactant.to_rarray(start; sharding = split), map(f -> Reactant.to_rarray(f; sharding = split), frac),
        batched, size(permuted), Tuple(invperm(collect(perm))), vsz
    )
    vsplit = Tuple(findall(in(keys(vislayout.axes)), collect(vnames)))
    vsharding = Reactant.Sharding.DimsSharding(mesh, vsplit, map(i -> vislayout.axes[vnames[i]], vsplit))
    p = VLBISkyModels.@set fwd.plan = stack
    placephases(q) = VLBISkyModels.@set p.phases = Reactant.to_rarray(reshape(q.phases, vsz); sharding = vsharding)
    return placephases(fwd), placephases(rev)
end

# The index of plane `q` within its block of `per` planes, which must be block `b`.
function _withinblock(q, per, b, name)
    cld(q, per) == b || throw(
        ArgumentError("the visibility domain's `$name` dim and the image's `$name` planes split differently over devices: points on the devices of image block $b use planes of block $(cld(q, per))")
    )
    return mod1(q, per)
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
_replicate(x::Array, rep) = Reactant.to_rarray(x; sharding = rep)
function _replicate(p::VLBISkyModels.NUFTPlan, rep)
    p = VLBISkyModels.@set p.plan = _replicate(p.plan, rep)
    return VLBISkyModels.@set p.phases = _replicate(p.phases, rep)
end
_replicate(p::PlaneStack, rep) = PlaneStack(_replicate(p.points, rep), _replicate(p.plane, rep), p.batched)
function _replicate(p::NUFFTSetPts{T, D}, rep) where {T, D}
    perm, base, frac = _replicate((p.perm, p.base, p.frac), rep)
    return NUFFTSetPts{T, D, typeof(p.plan), typeof(perm), typeof(base), typeof(frac)}(p.plan, p.M, perm, base, frac)
end
