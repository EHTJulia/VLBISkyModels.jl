export load_fits, save_fits

"""
    load_fits(fitsfile::String, T::Type{<:IntensityMap}))

Loads an image from a FITS file, placing it on the sky with the FITS world coordinate system
(the `CDi_j`, `PCi_j`/`CDELTi`, or legacy `CROTA2` forms) relative to the phase center
`OBSRA`/`OBSDEC`, or `CRVAL1`/`CRVAL2` when those are absent.
The function returns an `T<:IntensityMap`. By default if `T===IntensityMap` then we only return the Stokes I image. If `T<:IntensityMap{StokesParams}` then we return the full Stokes image.
Stokes planes are read from a `STOKES` axis of the primary HDU, identified by their FITS Stokes
codes, and otherwise from HDUs named `Q`, `U` and `V`.
"""
function load_fits(file, T::Type{<:IntensityMap})
    if !endswith(file, ".fits")
        @warn "File does not end with FITS trying to load anyways"
    end
    return _load_fits(file, T)
end

function _load_fits(fname, ::Type{IntensityMap})
    img = FITS(fname, "r") do f
        planes = _fits_planes(f[1], :I)
        if length(f) > 1 || length(planes) > 1
            @warn "Currently only loading stokes I. To load polarized quantities\n" *
                "please call `load_fits(filename, IntensityMap{StokesParams})`"
        end
        return _stokes_i_plane(planes, fname)
    end
    return img
end

function _stokes_i_plane(planes, fname)
    haskey(planes, :I) || throw(ArgumentError("$fname has no Stokes I plane in its primary HDU"))
    return planes[:I]
end

function _load_stokes_hdu(f, stokes::Symbol, imgI)
    name = string(stokes)
    if !haskey(f, name)
        @warn "No stokes $(stokes) found creating a zero array"
        return zero(imgI)
    end
    planes = _fits_planes(f[name], stokes)
    haskey(planes, stokes) ||
        throw(ArgumentError("HDU $name holds the Stokes planes $(collect(keys(planes))), not $stokes"))
    img = planes[stokes]
    img.X ≈ imgI.X && img.Y ≈ imgI.Y && posang(axisdims(img)) ≈ posang(axisdims(imgI)) ||
        throw(ArgumentError("HDU $name is on a different pixel grid than Stokes I"))
    return img
end

function _load_fits(fname, ::Type{<:IntensityMap{<:StokesParams}})
    img = FITS(fname, "r") do f
        planes = _fits_planes(f[1], :I)
        imgI = _stokes_i_plane(planes, fname)
        imgQ, imgU, imgV = map((:Q, :U, :V)) do s
            haskey(planes, s) ? planes[s] : _load_stokes_hdu(f, s, imgI)
        end
        return IntensityMap(
            StructArray{StokesParams{eltype(imgI)}}(
                (
                    I = baseimage(imgI),
                    Q = baseimage(imgQ),
                    U = baseimage(imgU),
                    V = baseimage(imgV),
                )
            ),
            axisdims(imgI)
        )
    end
    return img
end

# Stokes and polarization codes of a FITS STOKES axis (FITS 4.0, Table 29).
const FITS_STOKES_CODES = (
    I = 1, Q = 2, U = 3, V = 4,
    RR = -1, LL = -2, RL = -3, LR = -4,
    XX = -5, YY = -6, XY = -7, YX = -8,
)

_fits_ctype(header, n) = strip(string(get(header, "CTYPE$n", "")))

_fits_axis(header, ctype, nd) = findfirst(n -> _fits_ctype(header, n) == ctype, 1:nd)

function _fits_linear_world(header, n, p)
    delta = if haskey(header, "CD$(n)_$(n)")
        float(header["CD$(n)_$(n)"])
    else
        float(get(header, "CDELT$n", 1.0)) * float(get(header, "PC$(n)_$(n)", 1.0))
    end
    return float(get(header, "CRVAL$n", 0.0)) + delta * (p - float(get(header, "CRPIX$n", 0.0)))
end

"""
    _fits_planes(hdu::FITSIO.ImageHDU, label::Symbol)

Returns the images in `hdu` keyed by Stokes parameter. The planes of a `STOKES` axis are keyed
by their FITS Stokes codes; an HDU without one holds a single image, keyed by `label`. Any
other axis beyond the two celestial ones must have length one.
"""
function _fits_planes(hdu::FITSIO.ImageHDU, label::Symbol)
    header = read_header(hdu)
    data = read(hdu)
    nd = ndims(data)
    nd >= 2 || throw(ArgumentError("FITS image HDU has $nd axes; an image needs two celestial axes"))
    sax = _fits_axis(header, "STOKES", nd)
    for n in 3:nd
        n == sax || size(data, n) == 1 || throw(
            ArgumentError(
                "FITS axis $n ($(_fits_ctype(header, n))) has length $(size(data, n)); " *
                    "only a STOKES axis may have more than one plane"
            )
        )
    end
    nx, ny = size(data, 1), size(data, 2)
    isnothing(sax) && return Dict(label => _extract_fits_image(reshape(data, nx, ny), header))
    codes = map(k -> _fits_linear_world(header, sax, k), axes(data, sax))
    stokes = map(codes) do c
        s = findfirst(==(c), FITS_STOKES_CODES)
        isnothing(s) && throw(ArgumentError("FITS Stokes code $c is not defined; codes on this axis: $codes"))
        s in (:I, :Q, :U, :V) || throw(
            ArgumentError("FITS $s planes are not supported; only Stokes I, Q, U, V planes can be loaded")
        )
        s
    end
    allunique(stokes) || throw(ArgumentError("repeated FITS Stokes codes $codes"))
    return Dict(
        s => _extract_fits_image(reshape(selectdim(data, sax, k), nx, ny), header)
            for (k, s) in zip(axes(data, sax), stokes)
    )
end

"""
    _fits_celestial_jacobian(header)

The linear part of the celestial WCS: the derivative of the intermediate world coordinates
(l, m) with respect to the pixel coordinates (p₁, p₂), in radians per pixel. It is taken from
`CDi_j`, from `PCi_j` and `CDELTi`, or from `CDELTi` and the legacy `CROTA2`, in that order of
precedence.
"""
function _fits_celestial_jacobian(header)
    for n in 1:2
        unit = strip(string(get(header, "CUNIT$n", "deg")))
        unit == "deg" || throw(ArgumentError("FITS celestial axis $n has CUNIT$n = $unit; expected deg"))
    end
    has(prefix) = any(((i, j),) -> haskey(header, "$prefix$(i)_$(j)"), Iterators.product(1:2, 1:2))
    cdelt = [float(get(header, "CDELT$i", 1.0)) for i in 1:2]
    cd = if has("CD")
        [float(get(header, "CD$(i)_$(j)", 0.0)) for i in 1:2, j in 1:2]
    elseif has("PC")
        cdelt .* [float(get(header, "PC$(i)_$(j)", i == j ? 1.0 : 0.0)) for i in 1:2, j in 1:2]
    else
        s, c = sincosd(float(get(header, "CROTA2", 0.0)))
        [cdelt[1] * c -cdelt[2] * s; cdelt[1] * s cdelt[2] * c]
    end
    return deg2rad.(cd)
end

function _fits_phase_center(header)
    haskey(header, "OBSRA") && haskey(header, "OBSDEC") &&
        return float(header["OBSRA"]), float(header["OBSDEC"])
    _fits_is_radec(header) && return float(get(header, "CRVAL1", 0.0)), float(get(header, "CRVAL2", 0.0))
    @warn "No OBSRA/OBSDEC or RA/DEC reference value in header setting to 180.0, 0.0"
    return 180.0, 0.0
end

_fits_is_radec(header) = startswith(_fits_ctype(header, 1), "RA") && startswith(_fits_ctype(header, 2), "DEC")

# Direction cosines (l, m) of the WCS reference point (CRVAL1, CRVAL2) about the phase center.
function _fits_reference_offset(header, ra0, dec0)
    _fits_is_radec(header) || return SVector(0.0, 0.0)
    αr, δr = deg2rad(float(get(header, "CRVAL1", ra0))), deg2rad(float(get(header, "CRVAL2", dec0)))
    α0, δ0 = deg2rad(ra0), deg2rad(dec0)
    l = cos(δr) * sin(αr - α0)
    m = sin(δr) * cos(δ0) - cos(δr) * sin(δ0) * cos(αr - α0)
    return SVector(l, m)
end

function _fits_frequency(header)
    haskey(header, "FREQ") && return float(header["FREQ"])
    n = _fits_axis(header, "FREQ", Int(get(header, "NAXIS", 0)))
    return isnothing(n) ? 0.0 : _fits_linear_world(header, n, 1)
end

"""
    _extract_fits_image(data::AbstractMatrix, header)

Places the celestial plane `data` on a `RectiGrid` following the FITS WCS (FITS 4.0, §8).
Pixel coordinate p lies at the intermediate world coordinates (l, m) = J (p - CRPIX), with `J`
from [`_fits_celestial_jacobian`](@ref), offset by the position of (CRVAL1, CRVAL2) relative to
the phase center (OBSRA, OBSDEC). For the SIN projection (l, m) are the direction cosines of the
image plane. The pixel axes are reversed as needed so that X and Y increase along the grid, and
any rotation of the pixel axes on the sky becomes the grid's `posang`.
"""
function _extract_fits_image(data::AbstractMatrix, header)
    nx, ny = size(data)
    J = _fits_celestial_jacobian(header)
    dJ = det(J)
    iszero(dJ) && throw(ArgumentError("singular FITS celestial pixel transform $J"))
    f1 = J[1, 1] < 0 ? -1 : 1
    f2 = dJ < 0 ? -f1 : f1
    A = J .* [f1 f2]
    dx, dy = norm(A[:, 1]), norm(A[:, 2])
    abs(dot(A[:, 1], A[:, 2])) <= sqrt(eps()) * dx * dy || throw(
        ArgumentError("FITS celestial pixel axes are not perpendicular on the sky (transform $J)")
    )
    pa = atan(A[1, 2] / dy, A[1, 1] / dx)

    ra, dec = _fits_phase_center(header)
    rot = SMatrix{2, 2}(cos(pa), -sin(pa), sin(pa), cos(pa))
    o = transpose(rot) * _fits_reference_offset(header, ra, dec)
    r1 = float(get(header, "CRPIX1", 0.0))
    r2 = float(get(header, "CRPIX2", 0.0))
    r1 = f1 == 1 ? r1 : nx + 1 - r1
    r2 = f2 == 1 ? r2 : ny + 1 - r2
    xs = LinRange(dx * (1 - r1) + o[1], dx * (nx - r1) + o[1], nx)
    ys = LinRange(dy * (1 - r2) + o[2], dy * (ny - r2) + o[2], ny)

    i1 = f1 == 1 ? axes(data, 1) : reverse(axes(data, 1))
    i2 = f2 == 1 ? axes(data, 2) : reverse(axes(data, 2))
    image = data[i1, i2]
    T = eltype(image)

    if lowercase(strip(string(get(header, "BUNIT", "")))) == "jy/beam"
        @info "Converting Jy/Beam => Jy/pixel"
        bmaj = deg2rad(float(header["BMAJ"]))
        bmin = deg2rad(float(header["BMIN"]))
        beamarea = (2 * π * bmaj * bmin) / (8 * log(2))
        image .= image .* T(dx * dy / beamarea)
    end

    mjd = float(get(header, "MJD-OBS", get(header, "MJD", 0)))
    source = haskey(header, "OBJECT") ? string(header["OBJECT"]) : "NA"
    info = ComradeBase.MinimalHeader(source, ra, dec, mjd, _fits_frequency(header))
    g = RectiGrid((X = xs, Y = ys); header = info, posang = pa)
    return IntensityMap(image, g)
end

"""
    save_fits(file::String, img::IntensityMap, obs)

Saves an image to a fits file. You can optionally pass an EHTObservation so that ancillary information
will be added.
"""
function save_fits(fname::String, img::IntensityMap)
    return _save_fits(fname, img)
end

function make_header(img)
    head = header(img)
    if head isa ComradeBase.NoHeader
        return (source = "Unknown", RA = 180.0, DEC = 0.0, mjd = 0, F = 230.0e9)
    else
        return (
            source = head.source, RA = head.ra, DEC = head.dec, mjd = head.mjd,
            F = head.frequency,
        )
    end
end

function _prepare_header(image, stokes = "I")
    head = make_header(image)
    headerkeys = [
        "SIMPLE",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "EXTEND",
        "OBJECT",
        "CTYPE1",
        "CTYPE2",
        "CDELT1",
        "CDELT2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "CROTA2",
        "OBSRA",
        "OBSDEC",
        "FREQ",
        "CRPIX1",
        "CRPIX2",
        "MJD",
        "TELESCOP",
        "BUNIT",
        "STOKES",
    ]

    x0c, y0c = phasecenter(image)
    psizex, psizey = pixelsizes(image)
    rmat = rotmat(axisdims(image))
    CD1 = -rad2deg(psizex)
    CD2 = rad2deg(psizey)
    values = [
        true,
        -64,
        2,
        size(image, 1),
        size(image, 2),
        true,
        head.source,
        "RA---SIN",
        "DEC--SIN",
        CD1,
        CD2,
        CD1 * rmat[1, 1],
        CD2 * rmat[1, 2],
        CD1 * rmat[2, 1],
        CD2 * rmat[2, 2],
        -rad2deg(posang(axisdims(image))),
        head.RA,
        head.DEC,
        head.F,
        size(image, 1) / 2 + 0.5 - x0c / psizex,
        size(image, 2) / 2 + 0.5 + y0c / psizey,
        head.mjd,
        "VLBI",
        "JY/PIXEL",
        stokes,
    ]
    comments = [
        "conforms to FITS standard",
        "array data type",
        "number of array dimensions",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ]

    return headerkeys, values, comments
end

function _save_fits(fname::String, image::IntensityMap{T}) where {T <: Number}
    return FITS(fname, "w") do hdu
        return write_stokes(hdu, image)
    end
end

function write_stokes(f, image, stokes = "I", innername = "")
    headerkeys, values, comments = _prepare_header(image, stokes)
    hdeheader = FITSHeader(headerkeys, values, comments)
    img = parent(image[end:-1:1, :])
    return FITSIO.write(f, img; header = hdeheader, name = innername)
end

function _save_fits(fname::String, image::IntensityMap{T}) where {T <: StokesParams}
    return FITS(fname, "w") do fits
        write_stokes(fits, stokes(image, :I), "I")
        write_stokes(fits, stokes(image, :Q), "Q", "Q")
        write_stokes(fits, stokes(image, :U), "U", "U")
        return write_stokes(fits, stokes(image, :V), "V", "V")
    end
end
