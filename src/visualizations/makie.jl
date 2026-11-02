export polimage, polimage!, imageviz


function polimage end
function polimage! end

"""
    imageviz(img::IntensityMap; scale_length = rad2μas(fieldofview(img).X / 4), backgroundcolor = nothing, kwargs...)

A default image visualization for a `IntensityMap`. The scale bar has length `scale_length`
in μas, and `backgroundcolor = nothing` uses the first color of the colormap.

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
