export MultiDomainImage, MultiDomainParams, build_param, build_param!
import ComradeBase: imagepixels, NoHeader, DomainParams, allocate_imgmap, build_param
export build_param, build_param!
include("poly_spectral.jl")

### general multidomain stuffs ###

@doc """
    MultiDomainImage(img::IntensityMap, kernel, domains...)
    MultiDomainImage(cimg::ContinuousImage, domains...)

Convenience constructor for a multidomain (e.g. multifrequency or multitime)
[`ContinuousImage`](@ref).

The spatial image `img` (a 2D `IntensityMap`) provides the base parameters and the
spatial `(X, Y)` grid, `kernel` is the image pulse, and `domains...` are one or more
`DomainParams` models (such as [`PolySpectral`](@ref)) describing how the image varies
across the extra domains.

The result is a `ContinuousImage` whose `params` field is a [`MultiDomainParams`](@ref).
When passed to `intensitymap` or `visibilitymap` over a grid with extra `Fr`/`Ti`
dimensions the spatial image cube is materialized by evaluating the domain models at
each frequency/time.

# Example
```julia
base = IntensityMap(rand(64, 64), imagepixels(10.0, 10.0, 64, 64))
dom  = PolySpectral((1.0,), 230.0e9)          # spectral index = 1
cimg = MultiDomainImage(base, BSplinePulse{3}(), dom)
```
"""
function MultiDomainImage(img::IntensityMap, kernel, domains...)
    mdp = MultiDomainParams(parent(img), domains)
    return ContinuousImage(mdp, spatialdims(img), kernel)
end

function MultiDomainImage(cimg::ContinuousImage, domains...)
    mdp = MultiDomainParams(cimg.params, domains)
    return ContinuousImage(mdp, spatialdims(cimg.grid), cimg.kernel)
end

struct MultiDomainParams{P, M<:Tuple{Vararg{<:DomainParams}}} <: DomainParams{P}
   params::P # base model parameters shared by all domains
   models::M  # tuple of domains: contains the domain-specific parameters

    function MultiDomainParams(params, domains...) # wrap trailing argument into a tuple
        return new{typeof(params), typeof(domains)}(params, domains)
    end

    function MultiDomainParams(params, domains::Tuple) # already pre-wrapped in a tuple
        return new{typeof(params), typeof(domains)}(params, domains)
    end
end

(md::MultiDomainParams)(p) = build_param(md, p)

@doc """
    build_param!(buffer, md::MultiDomainParams, p)

In-place form of [`build_param`](@ref): transforms `buffer` through each model in
`md.models` in turn and returns it.

!!! warning "buffer is overwritten and is the seed"
    The **first argument `buffer` is mutated in place** and is the *seed* of the
    transformation. The stored base `md.params` is **not** read by this mutating path —
    the caller is responsible for initializing `buffer` (e.g. with the base). In
    particular, do **not** pass `md.params` itself as `buffer`, or the model's stored
    base will be corrupted. Use the non-mutating `build_param(md, p)` (which seeds from
    `md.params`) when you want a fresh result.
"""
function build_param!(buffer, md::MultiDomainParams, p)
    build_param!(buffer, first(md.models), p)
    return build_param!(buffer, MultiDomainParams(buffer, Base.tail(md.models)), p)
end

# end the recursive loop
build_param!(buffer, ::MultiDomainParams{P, Tuple{}}, p) where {P} = buffer

# build_param!(md::MultiDomainParams, p) = build_param!(mp.params, md, p)
function ComradeBase.build_param(md::MultiDomainParams, p)
    return ComradeBase.build_param(md.params, md, p)
end

function build_param(params, md::MultiDomainParams, p)
    newparams = build_param(params, first(md.models), p)
    return build_param(newparams, MultiDomainParams(newparams, Base.tail(md.models)), p)
end

function build_param(params, ::MultiDomainParams{P, Tuple{}}, p) where {P}
    return params
end

# Convenience: pair a base value with a spectral model. The base (an image array for
# imaging, or a scalar for geometric modeling) is stored in the `MultiDomainParams`;
# `PolySpectral` itself stays spectral-only.
#
# Note on dispatch: a 3-argument all-`Number` call `PolySpectral(a, b, c)` is the
# spectral-only `(index, freq0, p0)` constructor (more specific, so it wins). A scalar
# base therefore needs the 4-argument form `PolySpectral(base, index, freq0, p0)` (or a
# tuple `index`); an array base is unambiguous in any arity.
function PolySpectral(base::Union{Number, AbstractArray}, index, freq0::Number, p0 = zero(base))
    return MultiDomainParams(base, PolySpectral(index, freq0, p0))
end

# extending image pizels to time AND frequency to build the multidomain RectiGrid
@doc """
    $(@doc ComradeBase.imagepixels)

    ---

    **VLBISkyParamss extension:**

        imagepixels(fovx, fovy, nx, ny, d1, d2, x0=0, y0=0; posang=0, executor=Serial(), header=NoHeader())

    Extends `imagepixels` for multidomain (multifrequency/multitime) image cubes.
    `d1` and `d2` are extra dimension lists appended to the spatial grid after X and Y.
    Their order determines the index ordering of the output cube.

    - A frequency list is created with `Fr([...])`
    - A time list is created with `Ti([...])`

    Both must be subtypes of `DimensionalData.Dimensions.Dimension`.

    # Arguments
    - `d1::D1`, `d2::D2`: extra dimensions (frequency or time lists)
    - `x0`, `y0`: optional image center offsets (default `0`)

    # Examples

    ```julia
    julia> frlist = Fr([5, 6, 7])
    julia> tlist  = Ti([8, 9, 0])

    julia> fr_ti_grid = imagepixels(1, 1, 10, 10, frlist, tlist)
    # Fr index comes before Ti

    julia> ti_fr_grid = imagepixels(1, 1, 10, 10, tlist, frlist)
    # Ti index comes before Fr

    julia> fr_ti_grid != ti_fr_grid
    true
    ```

    imagepixels(fovx, fovy, nx, ny, d1, x0=0, y0=0; posang=0, executor=Serial(), header=NoHeader())

    Extends `imagepixels` for multidomain (multifrequency/multitime) image cubes.
    `d1` is an extra dimension (time or frequency) appended to the spatial grid after X and Y.

    - A frequency list is created with `Fr([...])`
    - A time list is created with `Ti([...])`

    Must be a subtype of `DimensionalData.Dimensions.Dimension`.

    # Arguments
    - `d1::D1`: extra dimension (frequency or time list)
    - `x0`, `y0`: optional image center offsets (default `0`)

    # Examples

    ```julia
    julia> frlist = Fr([5, 6, 7])
    julia> tlist  = Ti([8, 9, 0])

    julia> fr_grid = imagepixels(1, 1, 10, 10, frlist)
    # adding frequency dimension

    julia> ti_grid = imagepixels(1, 1, 10, 10, tlist)
    # adding time dimension
    ```
    """
function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer,
        d1::D1, d2::D2,
        x0::Number = zero(fovx), y0::Number = zero(fovy);
        posang::Number = zero(fovx),
        executor = Serial(), header = NoHeader()
    ) where {D<:DimensionalData.Dimensions.Dimension, D1<:D, D2<:D}
    @assert (nx > 0) && (ny > 0) "Number of pixels must be positive"

    psizex = fovx / nx
    psizey = fovy / ny

    xitr = X(LinRange(-fovx / 2 + psizex / 2 - x0, fovx / 2 - psizex / 2 - x0, nx))
    yitr = Y(LinRange(-fovy / 2 + psizey / 2 - y0, fovy / 2 - psizey / 2 - y0, ny))
    d1itr = d1
    d2itr = d2
    grid = RectiGrid((xitr, yitr, d1itr, d2itr); executor, header, posang)
    return grid
end

# extending imagepixels to time OR frequency to build multidomain RectiGrid
function imagepixels(fovx::Real, fovy::Real, nx::Integer, ny::Integer,
        d1::D1,
        x0::Number = zero(fovx), y0::Number = zero(fovy);
        posang::Number = zero(fovx),
        executor = Serial(), header = NoHeader()
    ) where {D1<:DimensionalData.Dimensions.Dimension}
    @assert (nx > 0) && (ny > 0) "Number of pixels must be positive"

    psizex = fovx / nx
    psizey = fovy / ny

    xitr = X(LinRange(-fovx / 2 + psizex / 2 - x0, fovx / 2 - psizex / 2 - x0, nx))
    yitr = Y(LinRange(-fovy / 2 + psizey / 2 - y0, fovy / 2 - psizey / 2 - y0, ny))
    d1itr = d1
    grid = RectiGrid((xitr, yitr, d1itr); executor, header, posang)
    return grid
end

@inline _getindices(index::NTuple{N, <:AbstractArray}, i) where {N} = ntuple(n -> index[n][i], Val(N))
@inline _getindices(index::NTuple, i) = index