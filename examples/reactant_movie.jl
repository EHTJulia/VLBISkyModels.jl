# # Multifrequency movies with Reactant

# This example evaluates a multifrequency movie with [Reactant](https://github.com/EnzymeAD/Reactant.jl),
# which compiles the model and its gradient with XLA, and then splits the movie across
# several devices. It uses the models and domains described in [Multidomain models](@ref).

# XLA must be told how many devices to simulate before Reactant starts. With a real
# multi-device machine leave this out.
ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

using VLBISkyModels
using Reactant
using Enzyme
redirect_stderr(devnull) do #hide
    Reactant.set_default_backend("cpu") #hide
    Reactant.devices() #hide
end #hide
nothing #hide

# ## The model

# The movie has four frames and two observing bands. Each frame has its own image, and a
# spectral index scales every pixel between the bands:

nx = 32
fov = μas2rad(120.0)
gs = spatialgrid(fov, fov, nx, nx)
ti = frames(Ti, [0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
frs = Fr([230.0e9, 345.0e9])
pulse = BSplinePulse{3}()
movie(frames) = MultiDomainImage(IntensityMap(frames, gs ⊗ ti), pulse, PolySpectral(-0.7, 230.0e9))

# The frames show a Gaussian that moves across the field of view, one image at the center
# of each frame:

blob(t) = modify(Gaussian(), Stretch(μas2rad(10.0)), Shift(μas2rad(10.0 * (t - 2.0)), 0.0))
truth = stack(t -> baseimage(intensitymap(blob(t), gs)), [0.5, 1.5, 2.5, 3.5])
size(truth)

# The data are one `(Node, Fr)` domain: each baseline is measured in both bands, at one time.
# Baselines are given in meters and converted to wavelengths with the `Fr` lookup.

npt = 400
dom = StructuredDomain(
    (Node(npt), frs);
    u = 3.0e6 .* randn(npt), v = 3.0e6 .* randn(npt), Ti = repeat(4 .* rand(npt), 1, 2),
)

# The image grid of the transform has every dim the visibilities depend on: the frames and
# the bands.

g = gs ⊗ ti ⊗ frs
ghost = FourierDualDomain(g, dom, NFFTAlg())
vtrue = baseimage(visibilitymap(movie(truth), ghost))
size(vtrue)

# ## Compiling the model

# `ReactantNUFFTAlg` is the nonuniform FFT that runs under Reactant. The visibility domain is
# moved to the device with `Reactant.to_rarray`; the image grid stays on the host, since it
# only sets up the plan.

alg = VLBISkyModels.ReactantNUFFTAlg(Float64; eps = 1.0e-10)
gdev = FourierDualDomain(g, Reactant.to_rarray(dom), alg)

vis(frames, gd) = baseimage(visibilitymap(movie(frames), gd))
framesdev = Reactant.to_rarray(truth)
vdev = @jit vis(framesdev, gdev)
maximum(abs, Array(vdev) .- vtrue)

# A loss and its gradient compile the same way. Enzyme writes the gradient into a shadow
# array created outside the compiled function.

chi2(frames, gd, data) = sum(abs2, vis(frames, gd) .- data)
function grad!(dframes, frames, gd, data)
    Enzyme.autodiff(Reverse, chi2, Active, Duplicated(frames, dframes), Const(gd), Const(data))
    return dframes
end

start = Reactant.to_rarray(zero(truth))
datadev = Reactant.to_rarray(vtrue)
dstart = Reactant.to_rarray(zero(truth))
size(@jit grad!(dstart, start, gdev, datadev))

# ## Splitting the movie across devices

# Sharding needs Reactant's IFRT runtime, selected with the Reactant preference
# `xla_runtime = "IFRT"`, for example in the `LocalPreferences.toml` next to the active
# project. A mesh arranges the devices in a grid with named axes. Here the frames are split over
# the `:t` axis and the bands over the `:f` axis, so each device holds two frames of one
# band.

mesh = Reactant.Sharding.Mesh(reshape(Reactant.devices()[1:4], 2, 2), (:t, :f))
layout = ShardLayout(mesh; Ti = :t, Fr = :f)

# `shard` places the dual domain on the mesh. The visibilities of each band stay on the
# devices that hold it, and each device interpolates only the points whose frame it holds.

gshard = shard(gdev, layout)

# The frames of the model have no `Fr` dim, so they are split along `Ti` alone. A gradient
# shadow is split the same way.

tlayout = ShardLayout(mesh; Ti = :t)
framesshard = baseimage(shard(IntensityMap(truth, gs ⊗ ti), tlayout))
vshard = @jit vis(framesshard, gshard)
maximum(abs, Array(vshard) .- vtrue)

#

startshard = baseimage(shard(IntensityMap(zero(truth), gs ⊗ ti), tlayout))
dshard = baseimage(shard(IntensityMap(zero(truth), gs ⊗ ti), tlayout))
datashard = Reactant.to_rarray(vtrue; sharding = Reactant.Sharding.DimsSharding(mesh, (2,), (:f,)))
size(@jit grad!(dshard, startshard, gshard, datashard))

# Each sharded dim of the image grid must split evenly over its devices. The sharding page of
# the [ComradeBase documentation](https://ptiede.github.io/ComradeBase.jl/stable/sharding/)
# describes layouts and the current limitations of sharded computations in more detail.
