using Pyehtim

@testset "io.jl" begin
    fname = Downloads.download(
        "http://www.cv.nrao.edu/2cmVLBA/data/0003-066/2012_03_04/0003-066.u.2012_03_04.icn.fits.gz",
        joinpath(@__DIR__, "test.fits")
    )

    imc = load_fits(fname, IntensityMap)
    load_fits(fname, IntensityMap{StokesParams})
    ime = ehtim.image.load_image(fname)
    @test pyconvert(Tuple, ime.imarr("I").shape) == size(imc)
    @test flux(imc) ≈ pyconvert(Float64, ime.total_flux())
    fov = fieldofview(imc)
    @test fov.X ≈ pyconvert(Float64, ime.fovx())
    @test fov.Y ≈ pyconvert(Float64, ime.fovx())
    save_fits("test.fits", imc)
    rm("test.fits")

    x = X(range(-10.0, 10.0; length = 64))
    y = Y(range(-5.0, 5.0; length = 32))
    t = Ti([0.0, 0.5, 0.8])
    f = Fr([86.0e9, 230.0e9, 345.0e9])

    imgI = rand(64, 32, 3, 3)
    imgQ = rand(64, 32, 3, 3)
    imgU = rand(64, 32, 3, 3)
    imgV = rand(64, 32, 3, 3)

    imgP = StructArray{StokesParams}(; I = imgI, Q = imgQ, U = imgU, V = imgV)
    img1 = IntensityMap(imgP[:, :, 1, 1], RectiGrid((; X = x, Y = y)))
    save_fits("ptest.fits", img1)
    load_fits("ptest.fits", IntensityMap)
    img2 = load_fits("ptest.fits", IntensityMap{StokesParams})
    rm("ptest.fits")
    @test parent(img1) ≈ parent(img2)
    @test parent(img1.X) ≈ parent(img2.X)
    @test parent(img1.Y) ≈ parent(img2.Y)

    # Now test shift
    Xnew = imc.X .+ μas2rad(50.0)
    Ynew = imc.Y .- μas2rad(75.0)

    imgc2 = IntensityMap(baseimage(imc), RectiGrid((; X = Xnew, Y = Ynew); header = header(imc)))
    save_fits("etest.fits", imgc2)
    imgc2l = load_fits("etest.fits", IntensityMap)
    @test parent(imgc2) ≈ parent(imgc2l)
    @test parent(imgc2.X) ≈ parent(imgc2l.X)
    @test parent(imgc2.Y) ≈ parent(imgc2l.Y)
    rm("etest.fits")
end

@testset "io.jl FITS WCS" begin
    nx, ny = 16, 12
    function cubeheader(ctype3; extra = ())
        hdr = VLBISkyModels.FITSIO.FITSHeader(
            ["CTYPE1", "CTYPE2", "CTYPE3", "CDELT1", "CDELT2", "CDELT3", "CRPIX1", "CRPIX2", "CRPIX3", "CRVAL1", "CRVAL2", "CRVAL3", "OBSRA", "OBSDEC", "FREQ"],
            Any["RA---SIN", "DEC--SIN", ctype3, -1.0e-10, 2.0e-10, 1.0, 4.25, 9.5, 1.0, 187.7, 12.4, 1.0, 187.7, 12.4, 230.0e9],
            fill("", 15)
        )
        for (key, value) in extra
            hdr[key] = value
        end
        return hdr
    end
    fname = joinpath(mktempdir(), "cube.fits")

    # A single bright pixel at FITS pixel (p1, p2) lies at (l, m) = J (p - CRPIX).
    p1, p2 = 3, 7
    plane = zeros(nx, ny, 1)
    plane[p1, p2, 1] = 1
    VLBISkyModels.FITSIO.FITS(f -> write(f, plane; header = cubeheader("FREQ")), fname, "w")
    img = load_fits(fname, IntensityMap)
    k = argmax(parent(img))
    @test img.X[k[1]] ≈ deg2rad(-1.0e-10) * (p1 - 4.25)
    @test img.Y[k[2]] ≈ deg2rad(2.0e-10) * (p2 - 9.5)
    @test pixelsizes(img).X ≈ deg2rad(1.0e-10)
    @test pixelsizes(img).Y ≈ deg2rad(2.0e-10)
    @test header(img).frequency == 230.0e9

    ρ = 30.0
    cd = deg2rad.([-1.0e-10*cosd(ρ) -2.0e-10*sind(ρ); -1.0e-10*sind(ρ) 2.0e-10*cosd(ρ)])
    rotated = cubeheader("FREQ"; extra = ("CD1_1", "CD1_2", "CD2_1", "CD2_2") .=> rad2deg.((cd[1, 1], cd[1, 2], cd[2, 1], cd[2, 2])))
    VLBISkyModels.FITSIO.FITS(f -> write(f, plane; header = rotated), fname, "w")
    img = load_fits(fname, IntensityMap)
    k = argmax(parent(img))
    @test collect(values(domainpoints(axisdims(img))[k])) ≈ cd * [p1 - 4.25, p2 - 9.5]

    offset = cubeheader("FREQ"; extra = ("OBSRA", "OBSDEC") .=> (187.7 + 1.0e-9 / cosd(12.4), 12.4 + 2.0e-9))
    VLBISkyModels.FITSIO.FITS(f -> write(f, plane; header = offset), fname, "w")
    img = load_fits(fname, IntensityMap)
    k = argmax(parent(img))
    @test img.X[k[1]] ≈ deg2rad(-1.0e-10) * (p1 - 4.25) - deg2rad(1.0e-9) atol = deg2rad(1.0e-11)
    @test img.Y[k[2]] ≈ deg2rad(2.0e-10) * (p2 - 9.5) - deg2rad(2.0e-9) atol = deg2rad(1.0e-11)

    cube = rand(Float32, nx, ny, 4)
    VLBISkyModels.FITSIO.FITS(f -> write(f, cube; header = cubeheader("STOKES")), fname, "w")
    imgP = load_fits(fname, IntensityMap{StokesParams})
    for (k, s) in pairs((:I, :Q, :U, :V))
        @test parent(stokes(imgP, s)) == cube[end:-1:begin, :, k]
    end
    @test parent(load_fits(fname, IntensityMap)) == cube[end:-1:begin, :, 1]
    @test header(load_fits(fname, IntensityMap)).frequency == 230.0e9

    VLBISkyModels.FITSIO.FITS(f -> write(f, cube; header = cubeheader("STOKES"; extra = ("CRVAL3" => -1.0,))), fname, "w")
    @test_throws "FITS RR planes are not supported" load_fits(fname, IntensityMap)

    VLBISkyModels.FITSIO.FITS(f -> write(f, cube; header = cubeheader("FREQ")), fname, "w")
    @test_throws "only a STOKES axis may have more than one plane" load_fits(fname, IntensityMap)

    single = reshape(cube[:, :, 1], nx, ny, 1, 1)
    VLBISkyModels.FITSIO.FITS(f -> write(f, single; header = cubeheader("FREQ"; extra = ("CTYPE4" => "STOKES", "CRVAL4" => 1.0, "CRPIX4" => 1.0))), fname, "w")
    imgI = load_fits(fname, IntensityMap{StokesParams})
    @test parent(stokes(imgI, :I)) == cube[end:-1:begin, :, 1]
    @test all(iszero, parent(stokes(imgI, :Q)))
end
