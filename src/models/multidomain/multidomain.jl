export MultiDomainImage, DomainList
import ComradeBase: imagepixels, NoHeader, FrequencyParams, DomainParams, allocate_imgmap
include("poly_spectral.jl")

### general multidomain stuffs ###

"""
    MultiDomainImage

Represents an image model evaluated over multiple domains (frequency, time).
Enables Multifrequency or Time Domain imaging.
Domain order here doesn't matter - the evaluation order is defined by the grid.
"""
struct MultiDomainImage{I<:ContinuousImage, D1<:DomainParams, D2<:DomainParams} <: AbstractModel
    imgmodel::I
    domain1::D1
    domain2::D2

    function MultiDomainImage(imgmodel::I, domain1::D1, domain2::D2) where {I<:ContinuousImage, D1<:DomainParams, D2<:DomainParams}
        d1 = setdomainparam(domain1, imgmodel)
        d2 = setdomainparam(domain2, imgmodel)
        return new{I, typeof(d1), typeof(d2)}(imgmodel, d1, d2)
    end
end

struct EmptyDomain <: ComradeBase.DomainParams{Nothing} end

setdomainparam(d::EmptyDomain, param) = d


function MultiDomainImage(imgmodel::I, domain::D) where {I<:ContinuousImage, D<:DomainParams}
    return MultiDomainImage(imgmodel, domain, EmptyDomain())
end

# required model definitions
visanalytic(::MultiDomainImage{I}) where {I} = NotAnalytic()
imanalytic(::MultiDomainImage{I}) where {I} = imanalytic(I)
radialextent(::MultiDomainImage{I}) where {I} = radialextent(I)
flux(::MultiDomainImage{I}) where {I} = flux(I)
ispolarized(::MultiDomainImage{I}) where {I} = ispolarized(I)

function intensitymap_numeric(md::MultiDomainImage{<:ContinuousImage},imggrid::RectiGrid)
    mdimg = allocate_imgmap(md.imgmodel, imggrid) # allocate result: multidomain image

    # apply domain models to the multidomain image
    apply_domain!(mdimg, md.domain1, imggrid) # apply first domain model
    apply_domain!(mdimg, md.domain2, imggrid) # apply second domain model

    return mdimg
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

# if 2nd domain doesn't exist, do nothing and return the input
function apply_domain!(mdimg, domain::EmptyDomain, imggrid::RectiGrid)
    return mdimg
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



### mfs specific ###


# construct the reference frequency parameterization
# dispatches to specific spectral model implementation
#@fastmath @inline function build_param(model::M, grid::RectiGrid) where {M<:FrequencyParams{<:Int}}
#    lf = build_reference_frequency(model, grid.Fr)
#    return build_spectral(model.param, model.index, lf, model.p0, typeof(model))
#end

# applying the spectral expansion to ContinuousImage
@fastmath @inline function apply_domain!(mdimg::IntensityMap, specmodel::S, imggrid::RectiGrid) where {S<:FrequencyParams}
    mp0 = specmodel.p0 # initial spectral model parameters

    # builds a N-length tuple holding the reference frequency parameterization for all frequencies
    ref_freqs = build_reference_frequency(specmodel, imggrid.Fr)

    frdim = findfirst(typeof.(dims(mdimg)) .<: Fr) # get which dimension corresponds to frequency

    spatialinds = CartesianIndices((axes(mdimg, 1), axes(mdimg, 2))) # getting spatial indices of image

    # loop over frequencies
    @trace track_numbers=false for i in axes(mdimg, frdim)
        # view the data associated with each frequency
        frslice = selectdim(mdimg, frdim, i) # axes are (X,Y,Ti) or (X,Y)
        ref_freq = ref_freqs[i] # get reference frequency parameterization

        # loop over spatial indices
        for pixind in spatialinds
            index = _getindices(specmodel.index, pixind) # for each pixel, grab the corresponding spectral parameters
            pixfrslice = @view frslice[pixind, :] # grabbing the image values at that pixel & frequency
            # loop over time dimension (if it exists) to calculate spectral expansion on the image
            map!(val ->  @inline build_spectral(val, index, ref_freq, mp0), pixfrslice, pixfrslice) # dispatch to apply the spectral model
        end
    end

    return mdimg
end

@inline _getindices(index::NTuple{N, <:AbstractArray}, i) where {N} = ntuple(n -> index[n][i], Val(N))
@inline _getindices(index::NTuple, i) = index