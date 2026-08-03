module VLBISkyModelsReactantExt

using VLBISkyModels
using AbstractFFTs
using ComradeBase
using Reactant
using NFFT
using NFFT: AbstractNFFTs
using VLBISkyModels: ReactantNUFFTAlg
using LinearAlgebra

include("nufft/ReactantNUFFT.jl")

function VLBISkyModels.applyphases!(vis::Reactant.AbstractArray, phases::Reactant.AnyTracedRArray)
    vout = vis .* phases
    return copyto!(vis, vout)
end

# The traced version of the multidomain slice loop. The CPU path evaluates each output
# pixel over the pulse's support window, but those data-dependent loops cannot be traced
# through the executor's broadcast. A `Pulse` is separable — `intensity_point` factors as
# `κ(ΔX)κ(ΔY)` — so the kernel resampling of a slice is instead two dense matmuls,
# `Mx * slice * Myᵀ`, which XLA handles natively. κ vanishes outside the pulse support, so
# using the full matrices computes exactly the CPU path's window sums. The matrices depend
# only on the (static) grids, so they are built once on the host outside the loop.
#
# A traced cube also cannot be viewed at a traced slice index, so slices are fetched with
# `getindex` (a `dynamic_slice`) and written back with `setindex!` (a
# `dynamic_update_slice`), landing in `pimg` in one final copy. `track_numbers = false`
# keeps the numbers captured by the loop from being promoted along with the index, which
# would put a traced value into the `LinRange` of grid coordinates.
function VLBISkyModels._resample_slices!(
        pimg::Reactant.AnyTracedRArray, datacube, m::ContinuousImage, gspat
    )
    kernel = m.kernel
    kernel isa VLBISkyModels.Pulse || throw(
        ArgumentError(
            "Only separable `Pulse` kernels are supported for traced image " *
                "resampling; got $(nameof(typeof(kernel)))."
        )
    )
    gsrc = VLBISkyModels.spatialdims(m.grid)
    ComradeBase.posang(gsrc) == ComradeBase.posang(gspat) || throw(
        ArgumentError(
            "The image and output grids of a traced image must share a position angle."
        )
    )
    dx, dy = pixelsizes(gsrc)
    dxo, dyo = pixelsizes(gspat)
    X = gsrc.X
    Y = gsrc.Y
    Xo = gspat.X
    Yo = gspat.Y
    Mx = VLBISkyModels.κ.(Ref(kernel), (Xo .- X') ./ dx) .* (dxo / dx)
    Myt = VLBISkyModels.κ.(Ref(kernel), (Yo' .- Y) ./ dy) .* (dyo / dy)

    cube = reshape(datacube, size(gsrc)..., :)
    nsl = size(cube, 3)
    out = similar(datacube, length(Xo), length(Yo), nsl)
    Reactant.@trace track_numbers = false for k in axes(cube, 3)
        out[:, :, k] = Mx * cube[:, :, k] * Myt
    end
    if nsl == 1 && ndims(pimg) > 2
        # A single stored slice evaluated over a larger grid replicates across the
        # trailing dimensions, matching per-point evaluation of a spatial image.
        pimg .= reshape(out, size(out, 1), size(out, 2), ntuple(_ -> 1, ndims(pimg) - 2)...)
    else
        pimg .= reshape(out, size(pimg))
    end
    return nothing
end

function VLBISkyModels.PolExp2Map!(
        a::Reactant.AnyTracedRArray,
        b::Reactant.AnyTracedRArray,
        c::Reactant.AnyTracedRArray,
        d::Reactant.AnyTracedRArray,
        grid::ComradeBase.AbstractRectiGrid
    )

    # TODO figure out why the regular looped version isn't getting
    # raised nicely? Looks like some dus is getting in the way?
    p = sqrt.(b .^ 2 .+ c .^ 2 .+ d .^ 2)
    pimgI = exp.(a) .* cosh.(p)
    tmp = exp.(a) .* sinh.(p) ./ p
    pimgQ = tmp .* b
    pimgU = tmp .* c
    pimgV = tmp .* d

    copyto!(a, pimgI)
    copyto!(b, pimgQ)
    copyto!(c, pimgU)
    copyto!(d, pimgV)

    return stokes_intensitymap(a, b, c, d, grid)
end


# function VLBISkyModels.FourierDualDomain(
#         imgdomain::ComradeBase.AbstractRectiGrid, visdomain::ComradeBase.UnstructuredDomain,
#         algorithm::ReactantNUFFTAlg
#     )
#     plan_forward, plan_reverse = VLBISkyModels.create_plans(algorithm, imgdomain, visdomain)
#     return FourierDualDomain(imgdomain, Reactant.to_rarray(visdomain), algorithm, plan_forward, plan_reverse)
# end

end
