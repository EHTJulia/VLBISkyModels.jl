padfac(::DFTAlg) = 1

# We don't pad a DFT since it is already correct
padimage(::DFTAlg, img::IntensityMap) = img

"""
    DFTAlg(u::AbstractArray, v::AbstractArray)

Create an algorithm object using the direct Fourier transform object using the uv positions
`u`, `v` allowing for a more efficient transform.
"""
function DFTAlg(u::AbstractArray, v::AbstractArray)
    @argcheck length(u) == length(v)
    uv = Matrix{eltype(u)}(undef, 2, length(u))
    uv[1,:] .= u
    uv[2,:] .= v
    return ObservedNUFT(DFTAlg(), uv)
end


# internal function that creates an DFT matrix/plan to use used for the img.
function plan_nuft(alg::ObservedNUFT{<:DFTAlg}, grid::AbstractGrid)
    uv = alg.uv
    (;X, Y) = grid
    dft = similar(similar(uv), Complex{eltype(uv)}, size(uv,2), size(grid)...)
    @fastmath for i in eachindex(Y), j in eachindex(X), k in axes(uv,2)
        u = uv[1,k]
        v = uv[2,k]
        # - sign is taken care of in _visibilities
        dft[k, j, i] = cispi(-2(u*X[j] + v*Y[i]))
    end
    # reshape to a matrix so we can take advantage of an easy BLAS call
    return reshape(dft, size(uv,2), :)
end

# internal function to make the phases to phase center the image.
function make_phases(alg::ObservedNUFT{<:DFTAlg}, grid::AbstractGrid, pulse::Pulse)
    u = @view alg.uv[1,:]
    v = @view alg.uv[2,:]
    # We don't need to correct for the phase offset here since that
    # is taken care of in plan_nuft for DFTAlg
    dx, dy = pixelsizes(grid)
    return visibilities_analytic(stretched(pulse, dx, dy), u, v, zero(u), zero(u))
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
