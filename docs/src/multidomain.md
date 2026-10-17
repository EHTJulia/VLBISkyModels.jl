# Multidomain models

A multidomain model varies with observing time `Ti`, frequency `Fr`, or both. The parameters
of a geometric model can change with time and frequency, an image can be a spectral cube or
a movie, and both are evaluated on grids and visibility domains that carry `Ti` and `Fr`.
This page covers the models; the
[ComradeBase documentation](https://ptiede.github.io/ComradeBase.jl/stable/domains/)
describes how to build the grids and domains.

```@example md
using VLBISkyModels
using Random #hide
Random.seed!(1) #hide
nothing #hide
```

## Parameters that vary with time and frequency

A parameter that varies is a [`MultiDomainParams`](@ref): a base value, the value at a
reference time and frequency, paired with one or more families that say how it departs from
the base. [`PolySpectral`](@ref) is a power law in frequency, so this Gaussian shrinks as the
frequency rises:

```@example md
σ = MultiDomainParams(1.0, PolySpectral(-1.0, 230.0e9))
m = modify(Gaussian(), Stretch(σ))
nothing #hide
```

The model is evaluated wherever the points carry a `Fr` coordinate, for example a list of
points observed at several frequencies:

```@example md
n = 200
dom = UnstructuredDomain((; U = randn(n) ./ 4, V = randn(n) ./ 4, Fr = rand([86.0e9, 230.0e9, 345.0e9], n)))
vis = visibilitymap(m, dom)
vis[1:3]
```

Both [`PolySpectral`](@ref) and [`PolyTemporal`](@ref) scale the base by default
(`link = log`). Pass `link = identity` for a parameter that changes additively, such as a
position that drifts at a constant rate:

```@example md
x0 = MultiDomainParams(0.0, PolyTemporal(0.5, 0.0; link = identity))
mt = modify(Gaussian(), Shift(x0, 0.0))
dt = UnstructuredDomain((; U = randn(n) ./ 4, V = randn(n) ./ 4, Ti = 4 .* rand(n)))
visibilitymap(mt, dt)[1:3]
```

Families compose in one chain and are applied in order, each transforming the result of the
previous one, so a size can vary in both time and frequency:

```@example md
MultiDomainParams(1.0, PolyTemporal(0.1, 0.0), PolySpectral(-1.0, 230.0e9))
```

A model reads its parameters with [`getparam`](@ref) or [`@unpack_params`](@ref), which
evaluate a chain at the point; see [Model Interface](interface.md).

## Writing a family

A family is a subtype of [`DomainParams`](@ref ComradeBase.DomainParams) whose type parameter is the
element type of the value it produces at a point. It defines
[`apply_param`](@ref ComradeBase.apply_param), which transforms the running value `base`, and
optionally [`paramfield`](@ref ComradeBase.paramfield), the part that depends on the point alone.
This family scales the base linearly with frequency:

```@example md
struct LinearSpectral{T, F} <: VLBISkyModels.DomainParams{T}
    slope::T
    freq0::F
end
VLBISkyModels.paramfield(f::LinearSpectral, p) = 1 .+ f.slope .* (p.Fr ./ f.freq0 .- 1)
VLBISkyModels.apply_param(base, ::LinearSpectral, field, p) = base .* field

ml = modify(Gaussian(), Stretch(MultiDomainParams(1.0, LinearSpectral(0.5, 230.0e9))))
visibilitymap(ml, dom)[1:3]
```

`field` is what `paramfield` returned for the same point. A family with no part that
depends on the point alone leaves `paramfield` undefined and ignores `field`.

When an image is evaluated on a grid, `p.Fr` may be the whole frequency axis at once while
`base` is the whole image. `paramfield` then runs once per frequency, while anything written
into the broadcast in `apply_param` runs once per pixel of the result, so put work that
depends only on the point in `paramfield`.

`apply_param` may also return a lazy `Base.broadcasted(...)` instead of an array. A chain of
such families composes without materializing, and a [`ContinuousImage`](@ref) evaluated at a
single point computes only the pixels under its kernel instead of the whole image.

## Image cubes and movies

A [`MultiDomainImage`](@ref) is a [`ContinuousImage`](@ref) whose pixels are a chain: the
base image and the families applied to every pixel. With a spectral index this is a cube:

```@example md
gs = spatialgrid(μas2rad(100.0), μas2rad(100.0), 32, 32)
base = intensitymap(modify(Gaussian(), Stretch(μas2rad(10.0))), gs)
cube = MultiDomainImage(base, BSplinePulse{3}(), PolySpectral(-0.7, 230.0e9))
gcube = gs ⊗ Fr([86.0e9, 230.0e9, 345.0e9])
vec(baseimage(flux(intensitymap(cube, gcube))))
```

A movie is an image whose grid has a `Ti` dim, one image per frame. [`frames`](@ref) builds
a `Ti` dim of intervals, and gaps between them are allowed:

```@example md
ti = frames(Ti, [0.0, 1.5, 4.0], [1.0, 3.0, 5.0])
gm = gs ⊗ ti
movie = ContinuousImage(IntensityMap(rand(32, 32, 3), gm), BSplinePulse{3}())
size(intensitymap(movie, gm))
```

The two combine: a `MultiDomainImage` whose base has a `Ti` dim has its own image per frame
and the families on top, as in the [Multifrequency movies with Reactant](@ref) example.

Evaluate a model only on the dims it varies along. A chain that does not depend on an extra
dim of the grid is evaluated once and copied along it, and every copy is transformed.

## Visibilities

The visibilities of an image come from a [`FourierDualDomain`](@ref) that pairs the image
grid with a visibility domain. Each point uses the image plane that [`frameindex`](@ref)
assigns to its `Ti` and `Fr`: the interval that contains it for a dim built with `frames`,
the equal value otherwise. A point that falls in no plane is an error:

```@example md
uvd = UnstructuredDomain((; U = 2.0e9 .* randn(n), V = 2.0e9 .* randn(n), Ti = 5 .* rand(n)))
err = try
    FourierDualDomain(gm, uvd, NFFTAlg())
catch e
    e
end
```

With the points inside the frames, the movie's visibilities have the shape of the domain:

```@example md
uvd = UnstructuredDomain((; U = 2.0e9 .* randn(n), V = 2.0e9 .* randn(n), Ti = rand([0.5, 2.0, 4.5], n) .+ 0.2 .* rand(n)))
size(visibilitymap(movie, FourierDualDomain(gm, uvd, NFFTAlg())))
```

Data measured at every frequency fit a `(Node, Fr)` domain. Here the baselines are in meters
and converted to wavelengths with the `Fr` lookup:

```@example md
dpf = StructuredDomain((Node(n), Fr([86.0e9, 230.0e9, 345.0e9])); u = 4.0e6 .* randn(n), v = 4.0e6 .* randn(n))
size(visibilitymap(cube, FourierDualDomain(gcube, dpf, NFFTAlg())))
```

[`FFTAlg`](@ref) interpolates a single 2D FFT and cannot transform an image with `Ti` or
`Fr` planes; use a nonuniform transform such as [`NFFTAlg`](@ref) or [`DFTAlg`](@ref).

In a sum whose components vary along different dims, evaluate each component on its own
grid, with a `FourierDualDomain` per grid built from the same visibility domain, and add the
visibilities.
