using Random

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

function complex_transfer_loss(values, parameters)
    shape = size(values)
    forward_plan = VLBISkyModels._rfft_plan(eltype(values), shape)
    reverse_plan = VLBISkyModels._irfft_plan(eltype(values), shape)
    u = rfftfreq(shape[1], 1.0)
    v = fftfreq(shape[2], 1.0)'
    transfer = @. exp(-parameters[1] * (u^2 + v^2)) * cis(parameters[2] * (u + 0.37 * v))
    spectrum = similar(transfer)
    VLBISkyModels._apply_rfft!(spectrum, forward_plan, values)
    spectrum .*= transfer
    output = zeros(eltype(values), shape)
    VLBISkyModels._apply_irfft!(output, reverse_plan, spectrum)
    return sum(abs2, output)
end

function directional_central_difference(f, x, direction; step = 1.0e-6)
    return (f(x .+ step .* direction) - f(x .- step .* direction)) / (2 * step)
end

@testset "RFFT and IRFFT adjoints" begin
    rng = MersenneTwister(20261027)
    # The real inner product is <a, b>_R = real(sum(conj.(a) .* b)).
    # The tolerance is conservative relative to Float64 roundoff on these small arrays.
    for n in (5, 6), m in (3, 4)
        image_direction = randn(rng, n, m)
        spectrum_direction = randn(rng, div(n, 2) + 1, m) .+
                             im .* randn(rng, div(n, 2) + 1, m)

        rfft_adjoint = zeros(n, m)
        VLBISkyModels._rfft_adjoint_add!(rfft_adjoint, spectrum_direction)
        @test real(sum(conj.(spectrum_direction) .* rfft(image_direction, 1:2))) ≈
              sum(rfft_adjoint .* image_direction) atol = 5.0e-13 rtol = 5.0e-13

        irfft_adjoint = zeros(ComplexF64, div(n, 2) + 1, m)
        VLBISkyModels._irfft_adjoint_add!(irfft_adjoint, image_direction)
        @test sum(image_direction .* irfft(spectrum_direction, n, 1:2)) ≈
              real(sum(conj.(irfft_adjoint) .* spectrum_direction)) atol = 5.0e-13 rtol = 5.0e-13
    end
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

        constant_image_scale_gradient = zero(scale)
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.Reverse),
            Enzyme.Const(convolution_loss),
            Enzyme.Active,
            Enzyme.Const(image),
            Enzyme.Duplicated(scale, constant_image_scale_gradient),
        )
        @test constant_image_scale_gradient ≈ finite_scale_gradient rtol = 1.0e-6
    end
end

@testset "Complex Fourier transfer Enzyme gradient" begin
    values = reshape(collect(1.0:20.0), 5, 4)
    parameters = [0.17, 0.29]
    parameter_gradient = zero(parameters)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(complex_transfer_loss),
        Enzyme.Active,
        Enzyme.Const(values),
        Enzyme.Duplicated(parameters, parameter_gradient),
    )

    for direction in ([0.8, -0.3], [-0.25, 1.1], [0.6, 0.4])
        finite_difference = directional_central_difference(
            p -> complex_transfer_loss(values, p), parameters, direction
        )
        @test dot(parameter_gradient, direction) ≈ finite_difference rtol = 1.0e-6
    end
end
