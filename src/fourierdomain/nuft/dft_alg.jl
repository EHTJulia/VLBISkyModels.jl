"""
    DFTAlg
Uses a discrete fourier transform. This is not very efficient for larger images. In those cases
 NFFTAlg or FFTAlg are more reasonable. For small images this is a reasonable choice especially
since it's easy to define derivatives.
"""
struct DFTAlg <: NUFT end


# internal function that creates an DFT matrix/plan to use used for the img.
function plan_nuft(::DFTAlg, imagegrid::AbstractRectiGrid, visdomain::UnstructuredDomain)
    visp = domainpoints(visdomain)
    (;X, Y) = imagegrid
    dft = similar(similar(uv), Complex{eltype(uv)}, size(uv,2), size(grid)...)
    @fastmath for i in eachindex(Y), j in eachindex(X), k in eachindex(visp)
        u = uv[1,k]
        v = uv[2,k]
        # - sign is taken care of in _visibilities
        dft[k, j, i] = cispi(2(u*X[j] + v*Y[i]))
    end
    # reshape to a matrix so we can take advantage of an easy BLAS call
    return reshape(dft, length(visp), :)
end

# internal function to make the phases to phase center the image.
function make_phases(::DFTAlg, imagedomain::AbstractRectiGrid, visdomain::UnstructuredDomain, pulse::Pulse=DeltaPulse())
    visp = domainpoints(visdomain)
    dx, dy = pixelsizes(imagedomain)
    return visibilities_analytic(stretched(pulse, dx, dy), visp)
end

"""
    create_cache(alg::ObservedNUFT, plan , phases, img)

Create a cache for the DFT algorithm with precomputed `plan`, `phases` and `img`.
This is an internal version.
"""
function create_cache(alg::ObservedNUFT{<:DFTAlg}, plan, phases, grid, pulse::Pulse)
    return NUFTCache(alg, plan, phases, pulse, grid)
end

function _nuft(A::AbstractMatrix, b::AbstractArray)
    return A*reshape(b, :)
end
