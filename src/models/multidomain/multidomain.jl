export MultiDomainImage, JointDomain
import ComradeBase: imagepixels, NoHeader, FrequencyParams, DomainParams, allocate_imgmap
include("poly_spectral.jl")

### general multidomain stuffs ###

@docs """
    MultiDomainImage(imgmodel, domain)
    MultiDomainImage(imgmodel, domainarr)

Represent an image model evaluated over one or more additional domains, such as
frequency or time.

`MultiDomainImage` stores one domain object per image pixel. During construction,
each domain object's parameter is set to the corresponding pixel value in
`parent(imgmodel)`. Thus, `imgmodel` acts as the reference image for the domain
model parameters.

If an array of domain models is passed in, its parameter values are still
replaced by the corresponding values from `parent(imgmodel)`.

For multiple domains, use `JointDomain`. The order of domains in a `JointDomain`
determines the evaluation order.
"""
struct MultiDomainImage{I<:ContinuousImage, D<:AbstractArray{<:DomainParams}} <: AbstractModel
    imgmodel::I
    domainarr::D

    # passing in arrays of domains with imagesizes equal to the image
    function MultiDomainImage(imgmodel::I, domainarr::D) where {I<:ContinuousImage, D<:AbstractArray{<:DomainParams}}
        domainarr = setdomainparam(domainarr, parent(imgmodel))
        return new{I, typeof(domainarr)}(imgmodel, domainarr)
    end
end

function MultiDomainImage(imgmodel::I, domain::D) where {I<:ContinuousImage, D<:DomainParams}
    # create an array of domain models:
    # each element of the array is either a Domain or a JointDomain
    # the parameter value of the Domain/JointDomain is set by the image value at that pixel
    # the other parameters are set by user input
    domainarr = setdomainparam(domain, parent(imgmodel))
    return MultiDomainImage{I, typeof(domainarr)}(imgmodel, domainarr)
end

# required model definitions
visanalytic(::MultiDomainImage{I}) where {I} = NotAnalytic()
imanalytic(::MultiDomainImage{I}) where {I} = imanalytic(I)
radialextent(::MultiDomainImage{I}) where {I} = radialextent(I)
flux(::MultiDomainImage{I}) where {I} = flux(I)
ispolarized(::MultiDomainImage{I}) where {I} = ispolarized(I)



# combine multiple domains into a joint domain to pass into build_param
struct JointDomain{D<:Tuple} <: DomainParams{Any}
    domains::D

    function JointDomain(domains...)
        return new{typeof(domains)}(domains)
    end
end

@doc """
    JointDomain(domain1, domain2, ...)
    JointDomain(domainarr1, domainarr2, ...)

Combine multiple domain models into a single domain model evaluated sequentially.

The order of the arguments determines the evaluation order. For example,

    JointDomain(d1, d2)

means that `d1` is evaluated first, and its result is used as the input
parameter for `d2`.

If the inputs are arrays of domain models with matching axes, `JointDomain`
constructs an array of `JointDomains` by combining the domains at each spatial
index. For example,

    jd = JointDomain(d1arr, d2arr)

returns an array such that

    jd[I] == JointDomain(d1arr[I], d2arr[I])

for each index `I`.

All input domain arrays must have the same axes.
"""
JointDomain
JointDomain(domainarrs::AbstractArray...) = JointDomain(domainarrs)

function JointDomain(domainarrs::Tuple{Vararg{<:AbstractArray}})
    axes0 = axes(first(domainarrs))

    all(arr -> axes(arr) == axes0, domainarrs) ||
        throw(DimensionMismatch("All domain arrays passed to JointDomain must have the same axes"))

    return map(CartesianIndices(first(domainarrs))) do ind
        JointDomain(getindex.(domainarrs, Ref(ind))...)
    end
end


# when applied to a single JointDomain, broadcast p to the individual domains, then wrap the final result in a JointDomain again
# then create array of JointDomains the size of the image - one JointDomain per pixel
setdomainparam(jointdomain::JointDomain, p) = JointDomain(setdomainparam.(jointdomain.domains, Ref(p))...)
function setdomainparam(domainarr::AbstractArray{<:DomainParams}, params::AbstractArray) # spatially varying spectral params
    axes(domainarr) == axes(params) ||
        throw(DimensionMismatch("domainarr and params must have the same axes"))

    return map(CartesianIndices(params)) do ind
        setdomainparam(domainarr[ind], params[ind])
    end
end


function intensitymap_numeric(md::MultiDomainImage,imggrid::RectiGrid)
    mdimg = allocate_imgmap(md.imgmodel, imggrid) # allocate result: multidomain image cube

    # apply domain model(s) to the image
    apply_domain!(mdimg, md.domainarr, imggrid)

    return mdimg
end

@inline function apply_domain!(mdimg::IntensityMap, domainarr::AbstractArray, grid::AbstractArray)
    # loop over spatial indices in the image
    @trace track_numbers=false for gridind in CartesianIndices(mdimg)
        ind = Tuple(gridind)
        spatialind = CartesianIndex(ind[1], ind[2])
        mdimg[gridind] = build_param(domainarr[spatialind], grid[gridind])
    end

    return mdimg
end

function build_param(jointdomain::JointDomain, p)
    domains = jointdomain.domains
    isempty(domains) && throw(ArgumentError("JointDomain cannot be empty"))
    
    val = build_param(first(domains), p) # evaluate first domain

    for domain in Base.tail(domains) # get all entries after the first
        val = build_param(setdomainparam(domain, val), p) # evaluate other domains at updated param value
    end

    return val
end

function visibilitymap_numeric(md::MultiDomainImage{<:ContinuousImage},
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