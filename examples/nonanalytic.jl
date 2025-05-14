# # Modeling with non-analytic Fourier transforms
using VLBISkyModels

using Pkg #hide
Pkg.activate(joinpath(dirname(pathof(VLBISkyModels)), "..", "examples")) #hide

using CairoMakie

# While most of the models implemented in `VLBISkyModels` have an analytic
# Fourier transform this is not required. In this notebook we will
# describe how a user can do Bayesian model fitting with a
# non-analytic model.

# The `ExtendedRing` model is an example of a non-analytic model. The
# image structure is given by
# ```math
# I(r) = \frac{\beta^\alpha}{2\pi \Gamma(\alpha)} r^{-\alpha-2}e^{-\beta/r}
# ```

# This can be created as follows

m = ExtendedRing(8.0)
# The argument is `\alpha` in the above
# equation. `beta` is given by ``(1+\alpha)``.

# This model does not have a simple analytic Fourier transform, e.g.

VLBISkyModels.visanalytic(ExtendedRing)

# Therefore, to find the Fourier transform of the image we need to revert to numerical methods.
# For this notebook we will use the *fast Fourier transform* or FFT. Specifically we will
# use FFTW. To compute a numerical Fourier transform we first need to specify the image domain

gim = imagepixels(10.0, 10.0, 256, 256)

# Second we need to specify the list of points in the uv domain we are interested in.
# Since VLBI tends sparsely sample the UV plan we provide a specific type for this
# type called [`UnstructuredDomain`](@ref) that can be used to specify the UV points,

u = randn(1000) / 2
v = randn(1000) / 2
guv = UnstructuredDomain((U = u, V = v))

# We can now create a *dual domain* that contains both the image and the UV points and
# the specific Fourier transform algorithm we want to use. The options for algorithms are:

#  1. [`NFFTAlg`](@ref): Uses the Non-Uniform Fast Fourier Transform. Fast and accurate, this is the recommended algorithm for most cases.
#  2. [`DFTAlg`](@ref): Uses the Discrete Time Non-Uniform Fourier transform. Slow but exact, so only use for small image grids.
#  3. [`FFTAlg`](@ref): Moderately fast and moderately accurate. Typically only used internally for testing.
# For this example we will use `NFFTAlg` since it is the recommended algorithm for most cases.

gfour = FourierDualDomain(gim, guv, NFFTAlg())

# Given this `FourierDualDomain` everything now works as before with analytic models. For example
# we can compute the intensitymap of the model as

img = intensitymap(m, gfour)
imageviz(img)

# and the visibility map using
vis = visibilitymap(m, gfour)
fig, ax = scatter(hypot.(vis.U, vis.V), real.(vis); label = "Real")
scatter!(ax, hypot.(vis.U, vis.V), imag.(vis); label = "Imag")
axislegend(ax)
ax.xlabel = "uv-dist"
ax.ylabel = "Flux"
fig

# Additionally, you can also modify the models and add them in complete generality. For example

mmod = modify(m, Shift(2.0, 2.0)) + Gaussian()
mimg = intensitymap(mmod, gfour)
mvis = visibilitymap(mmod, gfour)

# Plotting everything gives
fig = Figure(; size = (800, 400))
ax1 = Axis(
    fig[1, 1]; xreversed = true, xlabel = "RA (radians)", ylabel = "Dec (radians)",
    aspect = 1
)
ax2 = Axis(fig[1, 2]; xlabel = "uv-dist", ylabel = "Amplitude")
image!(ax1, mimg; colormap = :afmhot)
scatter!(ax2, hypot.(mvis.U, mvis.V), abs.(mvis); label = "Real")
fig
