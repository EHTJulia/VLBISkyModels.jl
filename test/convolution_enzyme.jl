function reference_convolve(img::IntensityMap{<:Real}, m::VLBISkyModels.AbstractModel)
    reference = copy(img)
    image = baseimage(reference)
    shape = size(image)
    plan = plan_rfft(image, 1:2)
    points = VLBISkyModels._convolution_uvpoints(axisdims(reference), shape)
    spectrum = plan * image
    spectrum .*= conj.(VLBISkyModels.visibility_point.(Ref(m), points))
    inverse_plan = plan_irfft(spectrum, shape[1], 1:2)
    mul!(image, inverse_plan, spectrum)
    return reference
end

function convolution_loss(img, scale)
    model = modify(Gaussian(), Stretch(scale[1]))
    return sum(abs2, baseimage(convolve(img, model)))
end

@testset "Enzyme FFT convolution" begin
    for shape in ((5, 4), (6, 5))
        grid = imagepixels(1.0, 1.0, shape...)
        image = IntensityMap(reshape(collect(1.0:prod(shape)), shape), grid)
        scale = [0.13]
        model = modify(Gaussian(), Stretch(only(scale)))

        @test baseimage(convolve(image, model)) ≈ baseimage(reference_convolve(image, model))

        image_gradient = Enzyme.make_zero(image)
        scale_gradient = zero(scale)
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(convolution_loss),
            Enzyme.Active,
            Enzyme.Duplicated(image, image_gradient),
            Enzyme.Duplicated(scale, scale_gradient),
        )

        fdm = central_fdm(5, 1)
        finite_image_gradient = grad(
            fdm,
            values -> convolution_loss(IntensityMap(values, grid), scale),
            baseimage(image),
        )[1]
        finite_scale_gradient = grad(fdm, scale -> convolution_loss(image, scale), scale)[1]

        @test baseimage(image_gradient) ≈ finite_image_gradient rtol = 1.0e-6
        @test scale_gradient ≈ finite_scale_gradient rtol = 1.0e-6
    end
end
