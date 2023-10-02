export polimage, polimage!, imageviz

"""
    polimage(img::IntensityMap{<:StokesParams};
                colormap = :bone,
                colorrange = Makie.automatic,
                pcolorrange=Makie.automatic,
                pcolormap=Reverse(:jet1),
                nvec = 30,
                min_frac = 0.1,
                min_pol_frac=0.2,
                length_norm=1.0,
                plot_total=true)

Plot a polarized intensity map using the image `img`.

The plot follows the conventions from [EHTC M87 Paper VII](https://iopscience.iop.org/article/10.3847/2041-8213/abe71d).

The stokes `I` image will be plotted with the attributes
  - `colormap`
  - `colorrange`
  - `alpha`
  - `colorscale`

The polarized image will consist of a set. The attribute `plot_total` changes what polarized
quantities are considered.

**If `plot_total = true`**

 - The total polarization will be considered and the markers will be given by ellipses.
 - The orientation of the ellipse is equal to the EVPA.
 - The area of the ellipse is proportional to `|V|²`.
 - The semi-major axis is related to the total polarized intensity times the `length_norm`.
 - The color of the ellipse is given by the fractional total polarization times the
sign of Stokes `V`.

**If `plot_total = false`**

 - Only the linear polarization is considered and the markers will be ticks.
 - The orientation of the ticks is equal to the EVPA.
 - The length of the ticks is equal to the total linear polarized intensity,
   i.e. `√(Q² + U²)` times the `length_norm`.
 - The color of the tick is given by the fractional linear polarization.

## Attributes
  - `colormap`: The colormap of the stokes `I` image. The default is `:bone`.
  - `colorrange`: The color range of the stokes `I` image. The default is `(0, maximum(stokes(img, :I)))`
  - `pcolorrange:` The color range for the polarized image
  - `pcolormap`: The colormap used for fractional total/linear polarization markers.
  - `nvec`: The number of polarization vectors to plot
  - `min_frac`: Any markers with `I < min_frac*maximum(I))` will not be plotted
  - `min_pol_frac`: Any markers with `P < min_frac*maximum(P))` where `P` is the total/linear polarization flux
                will not be plotted.
  - `length_norm`: Specifies the normalization used for the ticks. The default is that the pixel
                    with the largest polarization intensity will have a tick length = 10x the
                    pixel separation. For an image with a maximum polarized intensity of 10Jy/μas²
                    and pixel spacing of 1μas the marker length will be equal to 10μas.
  - `plot_total`: If true plot the total polarization. If false only plot the linear polarization.




!!! warning
    The polarized plotting is intrinsically defined using astronomer/EHT polarization conventions
    This means that in order to have the polarization ticks plotted in a way that makes sense
    you need to have `xreversed=true` when defining your axis.

"""
function polimage end
function polimage! end

"""
    imageviz(img::IntensityMap; scale_length = fieldofview(img.X/4), kwargs...)

A default image visualization for a `IntensityMap`.

**If `eltype(img) <: Real`** i.e. an image of a single stokes parameter
this will plot the image with a colorbar in units of Jy/μas². The plot will
accept any `kwargs` that are a supported by the `Makie.Image` type an can
be queried by typing `?image` in the REPL

**If `eltype(img) <: StokesParams`** i.e. full polarized image
this will use `polimage`. The plot will
accept any `kwargs` that are a supported by the `PolImage` type an can
be queried by typing `?polimage` in the REPL.

!!! tip
    To customize the image, i.e. specify a specific axis we recommend to use
    `image` and `polimage` directly.

!!! warn
    To load this function definition you need to import `CairoMakie` first

"""
function imageviz end
