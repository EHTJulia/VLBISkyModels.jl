export DFTAlg

"""
    DFTAlg

Uses a discrete fourier transform. This is not very efficient for larger images. In those cases
NFFTAlg or FFTAlg are more reasonable. For small images this is a reasonable choice especially
since it's easy to define derivatives.
"""
struct DFTAlg <: NUFT end

# internal function that creates an DFT matrix/plan to use used for the img.
_rot(u, v, c, s) = (c * u - s * v, s * u + c * v) # inverse rotation
function plan_nuft_spatial(::DFTAlg, imagegrid::AbstractRectiGrid,
                           visdomain::UnstructuredDomain)
    visp = domainpoints(visdomain)
    (; X, Y) = imagegrid
    uv = domainpoints(visdomain)
    rmat = ComradeBase.rotmat(imagegrid)' # adjoint because we need to move into rotated frame
    dft = similar(Array{Complex{eltype(imagegrid)}}, length(visdomain),
                  size(imagegrid)[1:2]...)
    @fastmath for i in eachindex(Y), j in eachindex(X), k in eachindex(visp)
        uvr = rmat * SVector(uv.U[k], uv.V[k])
        u = uvr[1]
        v = uvr[2]
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

function _nuft!(out::AbstractArray, p::AbstractArray, b::AbstractArray{<:Real})
    return mul!(out, p, reshape(b, :))
end
