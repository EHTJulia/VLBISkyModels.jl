export load_fits, save_fits

"""
    load_fits(fitsfile::String, IntensityMap)

This loads in a fits file that is more robust to the various imaging algorithms
in the EHT, i.e. is works with clean, smili, eht-imaging.
The function returns an tuple with an intensitymap and a second named tuple with ancillary
information about the image, like the source name, location, mjd, and radio frequency.
"""
function load_fits(file, T::Type{<:IntensityMap})
    if !endswith(file, ".fits")
        @warn "File does not end with FITS trying to load anyways"
    end
    return _load_fits(file, T)
end

function _load_fits(fname, ::Type{IntensityMap})
    img = FITS(fname, "r") do f
        if length(f) > 1
            @warn "Currently only loading stokes I. To load polarized quantities\n" *
                  "please call `load_fits(filename, IntensityMap{StokesParams})`"
        end
        # assume that the first element is stokes I
        return _extract_fits_image(f[1])
    end
    return img
end

function try_loading(f, stokes, imgI)
    try
        return _extract_fits_image(f[stokes])
    catch
        @warn "No stokes $(stokes) found creating a zero array"
        imgQ = zero(imgI)
        return imgQ
    end
end

function _load_fits(fname, ::Type{<:IntensityMap{<:StokesParams}})
    img = FITS(fname, "r") do f
        # assume that the first element is stokes I
        imgI = _extract_fits_image(f[1])
        imgQ = try_loading(f, "Q", imgI)
        imgU = try_loading(f, "U", imgI)
        imgV = try_loading(f, "V", imgI)
        return IntensityMap(StructArray{StokesParams{eltype(imgI)}}((I=baseimage(imgI),
                                                                     Q=baseimage(imgQ),
                                                                     U=baseimage(imgU),
                                                                     V=baseimage(imgV))),
                            axisdims(imgI))
    end
    return img
end

function _extract_fits_image(f::FITSIO.ImageHDU{T}) where {T}
    image = read(f)[end:-1:begin, :]
    header = read_header(f)
    nx = Int(header["NAXIS1"])
    ny = Int(header["NAXIS2"])

    psizex = abs(float(header["CDELT1"])) * π / 180
    psizey = abs(float(header["CDELT2"])) * π / 180
    # We assume that the pixel center is in the middle of the image
    x0c = float(header["CRPIX1"]) - nx / 2 - 0.5
    y0c = float(header["CRPIX2"]) - ny / 2 - 0.5
    ra = (180)
    dec = zero(T)
    try
        ra = float(header["OBSRA"])
        dec = float(header["OBSDEC"])
    catch
        @warn "No OBSRA or OBSDEC in header setting to 180.0, 0.0"
    end

    #Get frequency
    freq = zero(T)
    if haskey(header, "FREQ")
        freq = parse(T, string(header["FREQ"]))
    elseif "CRVAL3" in keys(header)
        freq = float(header["CRVAL3"])
    end
    mjd = 0
    if haskey(header, "MJD")
        mjd = parse(T, string(header["MJD"]))
    end
    source = "NA"
    if haskey(header, "OBJECT")
        source = string(header["OBJECT"])
    end
    stokes = "NA"
    if haskey(header, "STOKES")
        stokes = Symbol(header["STOKES"])
    end
    bmaj = one(T) # Nominal values
    bmin = one(T)
    if haskey(header, "BUNIT")
        if header["BUNIT"] == "JY/BEAM"
            @info "Converting Jy/Beam => Jy/pixel"
            bmaj = header["BMAJ"] * π / 180
            bmin = header["BMIN"] * π / 180
            beamarea = (2 * T(π) * bmaj * bmin) / (8 * log(T(2)))
            image .= image .* (psizex * psizey / beamarea)
        end
    end
    info = ComradeBase.MinimalHeader(source, ra, dec, mjd, freq)
    g = imagepixels(psizex * nx, psizey * ny, nx, ny, x0c * psizex, y0c * psizey;
                    header=info)
    imap = IntensityMap(image, g)
    return imap
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
        return (source="Unknown", RA=180.0, DEC=0.0, mjd=0, F=230e9)
    else
        return (source=head.source, RA=head.ra, DEC=head.dec, mjd=head.mjd,
                F=head.frequency)
    end
end

function _prepare_header(image, stokes="I")
    head = make_header(image)
    headerkeys = ["SIMPLE",
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
                  "OBSRA",
                  "OBSDEC",
                  "FREQ",
                  "CRPIX1",
                  "CRPIX2",
                  "MJD",
                  "TELESCOP",
                  "BUNIT",
                  "STOKES"]

    x0c, y0c = phasecenter(image)
    psizex, psizey = pixelsizes(image)
    values = [true,
              -64,
              2,
              size(image, 1),
              size(image, 2),
              true,
              head.source,
              "RA---SIN",
              "DEC---SIN",
              rad2deg(psizex),
              rad2deg(psizey),
              head.RA,
              head.DEC,
              head.F,
              size(image, 1) / 2 + 0.5 + x0c / psizex,
              size(image, 2) / 2 + 0.5 + y0c / psizey,
              head.mjd,
              "VLBI",
              "JY/PIXEL",
              stokes]
    comments = ["conforms to FITS standard",
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
                ""]

    return headerkeys, values, comments
end

function _save_fits(fname::String, image::IntensityMap{T}) where {T<:Number}
    FITS(fname, "w") do hdu
        return write_stokes(hdu, image)
    end
end

function write_stokes(f, image, stokes="I", innername="")
    headerkeys, values, comments = _prepare_header(image, stokes)
    hdeheader = FITSHeader(headerkeys, values, comments)
    img = parent(image[end:-1:1, :])
    return FITSIO.write(f, img; header=hdeheader, name=innername)
end

function _save_fits(fname::String, image::IntensityMap{T}) where {T<:StokesParams}
    FITS(fname, "w") do fits
        write_stokes(fits, stokes(image, :I), "I")
        write_stokes(fits, stokes(image, :Q), "Q", "Q")
        write_stokes(fits, stokes(image, :U), "U", "U")
        return write_stokes(fits, stokes(image, :V), "V", "V")
    end
end
