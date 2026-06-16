export MultiDomainImage, MultiDomainModel, build_param, build_param!
import ComradeBase: imagepixels, NoHeader, DomainParams, allocate_imgmap, build_param
export build_param, build_param!
include("poly_spectral.jl")

### general multidomain stuffs ###

@doc """
    MultiDomainImage(imgmodel, domain)
    MultiDomainImage(imgmodel, domain1, domain2, ...)

imgmodel is a ContinuousImage. The domains are user-defined domain models.
Represent an image model evaluated over one or more additional domains, such as
frequency or time.

Convenience function for multidomain images to get wrapped in MultiDomainModel.
"""
function MultiDomainImage(imgmodel, domains...) # convenience function, wrap trailing argument into a tuple
    return MultiDomainModel{typeof(imgmodel), typeof(domains)}(imgmodel, domains)
end

struct MultiDomainModel{P, D<:Tuple{Vararg{<:DomainParams}}} <: AbstractModel
   params::P # base model parameters shared by all domains
   domains::D  # tuple of domains: contains the domain-specific parameters

    function MultiDomainModel(params, domains...) # wrap trailing argument into a tuple
        return new{typeof(params), typeof(domains)}(params, domains)
    end

    function MultiDomainModel(params, domains::Tuple) # already pre-wrapped in a tuple
        return new{typeof(params), typeof(domains)}(params, domains)
    end
end

(md::MultiDomainModel)(p) = build_param(md, p)

function ComradeBase.build_param(model::MultiDomainModel, p)
    params = model.params
    return build_param!(params, model, p)
end

function build_param!(params, model::MultiDomainModel, p)
    for domain in model.domains # e.g. PolySpectral
        # recursively dispatch to the build_param! implementation for each domain
        params = build_param!(params, domain, p)
    end
    return params
end

# required model definitions
visanalytic(::MultiDomainModel{I}) where {I} = NotAnalytic()
imanalytic(::MultiDomainModel{I}) where {I} = imanalytic(I)
radialextent(::MultiDomainModel{I}) where {I} = radialextent(I)
flux(::MultiDomainModel{I}) where {I} = flux(I)
ispolarized(::MultiDomainModel{I}) where {I} = ispolarized(I)

function intensitymap_numeric(md::MultiDomainModel,imggrid::RectiGrid)
    mdimg = allocate_imgmap(md.params, imggrid) # allocate result: multidomain image cube

    # loop over all points in the multidomain grid
    @trace track_numbers=false for ind in CartesianIndices(mdimg)
        mdimg[ind] = ComradeBase.build_param(md, imggrid[ind]) 
    end
    return mdimg
end

function visibilitymap_numeric(md::MultiDomainModel{<:ContinuousImage},
                               grid::AbstractFourierDualDomain)
    checkspatialgrid(axisdims(md.imgmodel), grid.imgdomain) # compare image dimensions to spatial dimensions of data cube
    mdimg = intensitymap_numeric(md, grid.imgdomain) # apply the spectral and/or time models to the image data
    vis = applyft(forward_plan(grid), mdimg) # FT to visibilities
    return applypulse!(vis, md.imgmodel.kernel, grid)
end

function checkspatialgrid(imgdims, grid)
    return !(dims(imgdims) == dims(grid)[1:2]) &&
           throw(ArgumentError("The image dimensions in `ContinuousImage`\n" *
                               "and the spatial dimensions of the visibility grid passed to `visibilitymap`\n" *
                               "do not match. This is not currently supported."))
end

# extending image pizels to time AND frequency to build the multidomain RectiGrid
@doc """
    $(@doc ComradeBase.imagepixels)

    ---

    **VLBISkyModels extension:**

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