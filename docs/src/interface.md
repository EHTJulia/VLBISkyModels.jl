# Model Interface

## Defining the Model

`VLBISkyModels` aims to be more modular and extensible than previous VLBI modeling packages. Namely, simple models are composed to construct complicated source morphologies instead of making many different models. This is accomplished with a type and trait-based hierarchy.

Additionally, [`ComradeBase`](https://github.com/ptiede/ComradeBase.jl) is a low-dependency version of this package that defines this type and trait hierarchy that users can more easily incorporate into their packages.

To see how this works, we will go through a simplified implementation of the Gaussian model in `VLBISkyModels`. The Gaussian model is a simple, compact emission structure model that can constrain the typical characteristic size
of an image feature from VLBI data. To construct a Gaussian model, we will first define a struct:

```julia
struct MyGaussian{T} <: VLBISkyModels.AbstractModel
    size::T
end
```

where `size` is the standard deviation of the gaussian.

!!! note
    We typically do not include a `size` field in the model because a generic `size` can be specified
    with the [`modify`](@ref) function. However, for the sake of this example, we will include it,
    since it will highlight how to make models frequency dependent.

To specify the model we need to tell `VLBISkyModels` how we will represent the model in the image
and visibility domain. For a Gaussian, both the image and visibilty domain are analytic. These
are specified with the traits [`ComradeBase.visanalytic`](@ref) and [`ComradeBase.imanalytic`](@ref):

```julia
# Fourier and image domains are analytic
ComradeBase.visanalytic(::Type{<:MyGaussian}) = IsAnalytic()
ComradeBase.imanalytic(::Type{<:MyGaussian}) = IsAnalytic()
```

Finally, we can specify if the model is intrinsically polarized by using the `IsPolarized` and `NotPolarized()` trait

```julia
VLBISkyModels.ispolarized(::Type{<:MyGaussian}) = NotPolarized()
```

!!! note
    The actual implementation defines the Gaussian to be a subtype of `VLBISkyModels.GeometricModel`,
    which automatically defines these methods. However, for models that aren't a subtype of `GeometricModel`,
    we assume the image domain `IsAnalytic()` and the Fourier domain is `NotAnalytic()`.

Since both the image and visibility domain representation of the Gaussian are analytic, we need to
define an `intensity_point` and `visibility_point` method.

```julia
function ComradeBase.intensity_point(m::MyGaussian, p)
    (; X, Y) = p
    @unpack_params size = m(p)
    return exp(-(X^2 + Y^2) * inv(2 * size^2)) * inv(2π * size^2)
end

function ComradeBase.visibility_point(m::MyGaussian, p)
    (; U, V) = p
    @unpack_params size = m(p)
    return exp(-2π^2 * size^2 * (U^2 + V^2)) + 0im
end
```

Note that as of **0.6.6** we recommend retrieving the parameters of the model `m` using either
[`getparam`](@ref) or the convience macro [`@unpack_params`](@ref). This is because the parameters of the model
may not be simple numbers but functions of time and frequency. Finally, not that `@unpack_params`
is a macro that reduces boiler plate and is equivalent to,

```julia
size = getparam(m, :size, p) # identical to @unpack_params size = m(p)
```

Additionally, most models in `VLBISkyModels` has two additional functions one can implement if possible:

 1. `flux(m::MyGaussian)`: This defines the flux of a model. If this isn't defined, the model won't have a flux until an image is created. For a Gaussian, the definition is `flux(::MyGaussian) = 1.0`.
 2. `radialextent(::MyGaussian)`: This defines the model's default radial extent. For a Gaussian, we will consider the radial extent to be $5σ$, so `radialextent(m::MyGaussian) = 5.0*m.size`. Although note that for a frequency dependent Gaussian this will not work correctly.

This completely defines the model interface for `VLBISkyModels`. With this, you can call the usual user API to evaluate, fit, and plot the model. Additionally, we can start talking about
adding multiple Gaussians and modifying them. For instance, with a flux of 2 Jy. This can be created by `VLBISkyModels` as follows:

```julia
using Plots
gauss = 2.0 * MyGaussian(1.0)
ellgauss = modify(gauss, Stretch(1.0, 0.5))
fig = plot(gauss; layout=(1, 2), size=(800, 300))
plot!(fig[2], ellgauss; size=(800, 350))
```

![Image](gauss.png)

```julia
using Plots
u = rand(100) * 0.5;
v = rand(100) * 0.5;
vg = visibilitymap(gauss, u, v)
veg = visibilitymap(ellgauss, u, v)

Plots.scatter(hypot.(u, v), abs.(vg); label="Gaussian")
Plots.scatter!(hypot.(u, v), abs.(veg); label="Elliptical Gaussian")
```

![Image](vis.png)

## Making the model frequency dependent

A parameter of the model may be a [`MultiDomainParams`](@ref), a base value paired with
families that say how it changes with time and frequency. Since `intensity_point` and
`visibility_point` read `size` with `@unpack_params`, `MyGaussian` needs no changes. To make
the size frequency dependent, pair the base size with a [`PolySpectral`](@ref) expansion:

```julia
ν₀ = 230e9
σ₀ = 1.0
a = 1.0
size = MultiDomainParams(σ₀, PolySpectral(a, ν₀))
gauss = MyGaussian(size)
```

which makes a Gaussian whose size as a function of frequency is

```math
\sigma(\nu) = \sigma_0 (\nu / \nu_0)^a
```

[Multidomain models](@ref) covers time dependence, chains of several families, and writing
a new family.

!!! note
    This extension of the model to be time and frequency dependent is only necessary for models
    that aren't intrinsically dependent on time and frequency. For a model, that has some prescribed
    time and frequency dependence, the model should be defined explicitly in `intensity_point`/`visibility_point`
    if they are analytic in the image/visibility domain respectively.
