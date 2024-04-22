```@meta
EditURL = "../../../examples/nonanalytic.jl"
```

# Modeling with non-analytic Fourier transforms

````@example nonanalytic
using VLBISkyModels

using Pkg #hide
Pkg.activate(joinpath(dirname(pathof(VLBISkyModels)), "..", "examples")) #hide

using CairoMakie
````

While most of the models implemented in `VLBISkyModels` have an analytic
Fourier transform this is not required. In this notebook we will
describe how a user can do Bayesian model fitting with a
non-analytic model.

The `ExtendedRing` model is an example of a non-analytic model. The
image structure is given by
```math
I(r) = \frac{\beta^\alpha}{2\pi \Gamma(\alpha)} r^{-\alpha-2}e^{-\beta/r}
```

This can be created as follows

````@example nonanalytic
m = ExtendedRing(8.0)
````

The argument is `\alpha` in the above
equation. `beta` is given by ``(1+\alpha)``.

This is an example of a ring model that has a substantially different flux profile.
Let's plot the image first giving it some unity by applying `modify`

````@example nonanalytic
muas = modify(m, Stretch(μas2rad(20.0)))
img = intensitymap(muas, μas2rad(100.0),  μas2rad(100.0), 256, 256)
imageviz(img)
````

This function does not have a simple analytic Fourier transform, e.g.

````@example nonanalytic
VLBISkyModels.visanalytic(ExtendedRing)
````

Therefore, to find the Fourier transform of the image we need to revert to numerical methods.
For this notebook we will use the *fast Fourier transform* or FFT. Specifically we will
use FFTW. To compute a numerical Fourier transform we first need to specify the image.

````@example nonanalytic
img = IntensityMap(zeros(256, 256), 10.0, 10.0)
````

This will serve as our cache to store the image going forward. The next step is to create
a model wrapper that holds the model and the image. `VLBISkyModels` provides the `modelimage`
function to do exactly that

````@example nonanalytic
mimage = modelimage(m, img, FFTAlg())
````

the `alg` keyword argument then specifies that we want to use an FFT to compute the
Fourier transform. When `modelimage` is called, the FFT is performed and then we use
a bicubic interpolator on the resulting visibilitymap to construct a continuous representation
of the Fourier transform. Once we have this everything else is the same. Namely we can
calculatge the VLBI data products in the usual manner i.e.

````@example nonanalytic
u = randn(1000)/2
v = randn(1000)/2
````

Now we can plot our sampled visibilitymap

````@example nonanalytic
vis = visibilitymap(mimage, (U=u, V=v))
scatter(hypot.(u, v), real.(vis), label="Real")
scatter!(hypot.(u, v), imag.(vis), label="Imag")
axislegend()
current_figure()
````

We can also directly get the amplitudemap using:

````@example nonanalytic
amp = amplitudemap(mimage, (U=u, V=v))
scatter(hypot.(u, v), amp, label="Amplitude")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

