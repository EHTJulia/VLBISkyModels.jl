export DFTAlg

"""
    DFTAlg

Uses a discrete fourier transform. This is not very efficient for larger images. In those cases
NFFTAlg or FFTAlg are more reasonable. For small images this is a reasonable choice especially
since it's easy to define derivatives.
"""
struct DFTAlg <: NUFT end

# internal function that creates an DFT matrix/plan to use used for the img.
function plan_nuft_spatial(::DFTAlg, imagegrid::AbstractRectiGrid,
                           visdomain::UnstructuredDomain)
    visp = domainpoints(visdomain)
    (; X, Y) = imagegrid
    uv = domainpoints(visdomain)
    dft = similar(Array{Complex{eltype(imagegrid)}}, length(visdomain), size(imagegrid)...)
    @fastmath for i in eachindex(Y), j in eachindex(X), k in eachindex(visp)
        u = uv.U[k]
        v = uv.V[k]
        # - sign is taken care of in _visibilitymap
        dft[k, j, i] = cispi(2(u * X[j] + v * Y[i]))
    end
    # reshape to a matrix so we can take advantage of an easy BLAS call
    return reshape(dft, length(visp), :)
end

# internal function to make the phases to phase center the image.
function make_phases(::DFTAlg, imgdomain::AbstractRectiGrid, visdomain::UnstructuredDomain)
    return one(Complex{eltype(visdomain.U)})
end

function _nuft(A::AbstractMatrix, b::AbstractArray)
    return A * reshape(b, :)
end
