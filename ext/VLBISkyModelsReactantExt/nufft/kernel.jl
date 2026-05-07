#==============================================================================
ES (exp-sqrt / "exponential of semicircle") kernel + Horner polynomial machinery.

Math:
    phi(z) = exp(beta * (sqrt(1 - z^2) - 1)),  |z| <= 1, else 0
with z = (cell-centered offset) / (w/2). Beta is set as in FINUFFT:
    beta = 2.30 * w * (2 / sigma).

Per-cell Horner expansion. For a NU point with fractional position
frac in [0, 1) inside its base cell (the leftmost of the w-cell stencil),
the kernel value at stencil cell k = 0..w-1 is

    weight[k] = phi(2*(frac + (k - (w-1)/2)) / w),

i.e. a function of `frac` for each fixed `k`. We fit each as a degree-`deg`
polynomial in `t = 2*frac - 1 ∈ [-1, 1]`, giving a coefficient matrix
`horner_coefs :: (w, deg+1)` ordered as monomial coefficients
`p_k(t) = sum_p coefs[k, p] * t^p`.

phi_hat (deconvolution table) is the discrete Fourier transform of the
sampled spreading kernel. We compute it by Gauss–Legendre quadrature of
the continuous `phi`, matching FINUFFT's `onedim_fseries_kernel`. This is
the principled choice — analytic FT of the ES kernel has no closed form.
==============================================================================#

# ----------------- ES kernel (pure host) -----------------

@inline function expsemicircle_phi(z::T, beta::T) where {T<:Real}
    az = abs(z)
    az >= one(T) && return zero(T)
    return exp(beta * (sqrt(one(T) - az * az) - one(T)))
end

# FINUFFT's beta scaling for ES kernel.
@inline function expsemicircle_beta(::Type{T}, w::Integer, sigma::Real) where {T<:Real}
    gamma = T(2.30) * (T(2) / T(sigma))
    return gamma * T(w)
end

# ----------------- Per-cell polynomial fit -----------------
#
# For each cell k in 0..w-1, fit the function
#     f_k(t) = phi( 2*(frac(t) + k - (w-1)/2) / w )
# where `frac(t) = (t + 1)/2` so `frac ∈ [0, 1]` as `t ∈ [-1, 1]`.
#
# We use Chebyshev interpolation at the (deg+1) Chebyshev–Gauss nodes
# (cosine spacing), then convert to monomial coefficients via Clenshaw-style
# evaluation back at sampled t. This is robust and entirely host-side.

function _chebyshev_nodes(::Type{T}, n::Integer) where {T<:Real}
    return [cos(T(pi) * (T(2k - 1) / T(2n))) for k in 1:n]
end

# Monomial-basis coefficients (length n) such that p(t) = sum c[i+1] * t^i
# minimizes max-norm error on [-1, 1]; computed via Chebyshev coefs +
# basis change. We keep this simple by sampling at (n) Chebyshev nodes,
# building a Vandermonde-like system in the monomial basis, and solving.
function _fit_monomial(::Type{T}, fvals::AbstractVector{T}, nodes::AbstractVector{T}) where {T<:Real}
    n = length(nodes)
    @assert length(fvals) == n
    # Vandermonde: V[i, j] = nodes[i]^(j-1)
    V = Matrix{T}(undef, n, n)
    @inbounds for i in axes(V, 2)
        x = nodes[i]
        v = one(T)
        for j in axes(V, 1)
            V[i, j] = v
            v *= x
        end
    end
    return V \ fvals
end

"""
    horner_coefficients(T, w, sigma; deg=w+2)

Return a `(w, deg+1)` matrix of monomial coefficients. Row `k` (1-based) gives
the coefficients for stencil cell `k-1`'s polynomial in `t = 2*frac - 1`.
"""
function horner_coefficients(::Type{T}, w::Integer, sigma::Real; deg::Integer=w + 2) where {T<:Real}
    beta = expsemicircle_beta(T, w, sigma)
    n = deg + 1
    nodes = _chebyshev_nodes(T, n)               # (n,) in [-1, 1]
    coefs = Matrix{T}(undef, w, n)
    halfw = T(w) / T(2)
    half_w_offset = (w ÷ 2) - 1                  # matches setpts: base = floor(s) - half_w_offset
    for k in 0:(w - 1)
        shift = T(k - half_w_offset)             # cell offset relative to floor(s)
        # weight[k] = phi( (shift - frac(t)) / (w/2) )
        # frac(t) = (t + 1)/2
        f = T[
            expsemicircle_phi((shift - (nodes[i] + one(T)) / T(2)) / halfw, beta)
            for i in 1:n
        ]
        coefs[k + 1, :] .= _fit_monomial(T, f, nodes)
    end
    return coefs
end

# ----------------- phi_hat via Gauss–Legendre quadrature -----------------
#
# phi_hat[m] = ∫_{-1}^{1} phi(z) * cos(pi*m*z*w/Nf) dz   (FINUFFT convention)
# This is symmetric so we use only the positive-z half: 2 * ∫_0^1.
# We sample phi at GL nodes scaled to [0,1].
#
# Implementation note: nquad ~ max(2*w + 2, ceil(0.5 * Nf / w)) suffices for
# eps near the kernel's nominal accuracy. We cap at MAX_NQUAD=100.

const MAX_NQUAD = 100

# Golub–Welsch via the Jacobi matrix. n nodes/weights on [-1, 1].
function _gauss_legendre(::Type{T}, n::Integer) where {T<:Real}
    if n <= 1
        return T[zero(T)], T[T(2)]
    end
    # Jacobi tridiagonal matrix entries: beta_k = k / sqrt(4k^2 - 1) for k=1..n-1
    bsub = [T(k) / sqrt(T(4 * k * k - 1)) for k in 1:(n - 1)]
    A = zeros(T, n, n)
    for k in 1:(n - 1)
        A[k, k + 1] = bsub[k]
        A[k + 1, k] = bsub[k]
    end
    F = LinearAlgebra.eigen(LinearAlgebra.Symmetric(A))
    nodes = F.values
    weights = T(2) .* (F.vectors[1, :] .^ 2)
    return nodes, weights
end

"""
    phi_hat_1d(T, w, sigma, Nf, nmodes_d) -> Vector{T} of length nmodes_d

Compute the deconvolution table for a single dimension, mapping logical mode
indices m ∈ {-nmodes_d/2, ..., (nmodes_d-1)/2} to phi_hat values on the
oversampled grid of size Nf.
"""
function phi_hat_1d(
    ::Type{T}, w::Integer, sigma::Real, Nf::Integer, nmodes_d::Integer
) where {T<:Real}
    beta = expsemicircle_beta(T, w, sigma)
    nquad = clamp(2 * w + 4, 8, MAX_NQUAD)
    nodes, weights = _gauss_legendre(T, nquad)   # on [-1, 1]
    # Use only positive half for symmetric integrand.
    pos_idx = nodes .> 0
    z = nodes[pos_idx]
    wq = weights[pos_idx]
    phi_z = [expsemicircle_phi(zz, beta) for zz in z]

    # Mode axis: centered around 0.
    #
    # Continuous-FT derivation: the spread kernel as a function of grid units
    # is `ker(g) = phi(2g/w)`, and its DFT under the change of variable
    # z = 2g/w gives a (w/2) Jacobian. Hence
    #
    #     phi_hat[m] = (w/2) * ∫_{-1}^{1} phi(z) cos(pi*m*w*z/Nf) dz
    #               = w   * ∫_{0}^{1}    phi(z) cos(pi*m*w*z/Nf) dz       (symmetric)
    #
    # The cos integrand uses the half-domain weights `wq` and a factor of 2
    # for the symmetric extension; final prefactor below is the (w/2).
    half = nmodes_d ÷ 2
    out = Vector{T}(undef, nmodes_d)
    inv_alpha = T(pi) * T(w) / T(Nf)            # = pi*w/Nf
    prefactor = T(w) / T(2)
    @inbounds for i in eachindex(out)
        m = T(i - 1 - half)
        s = zero(T)
        ang = inv_alpha * m
        for q in eachindex(z)
            s += T(2) * wq[q] * phi_z[q] * cos(ang * z[q])
        end
        out[i] = prefactor * s
    end
    return out
end

# ----------------- Traced Horner evaluator -----------------
#
# Evaluate w polynomials of degree `deg` simultaneously at M arguments t.
# coefs :: (w, deg+1) — host (or traced); t :: (M,) traced.
# Returns weights :: (M, w) traced.
#
# Uses Horner recurrence with batched broadcasts. The Julia loop over
# coefficient columns is unrolled in tracing because `deg` is statically
# known from the coef matrix shape.
function horner_eval(coefs::AbstractMatrix, t::AbstractVector)
    T = eltype(t)
    w_count = size(coefs, 1)
    n = size(coefs, 2)            # = deg + 1
    M = length(t)
    # Initialize accumulator with leading coefficient (degree `deg`).
    last_col = reshape(coefs[:, n], 1, w_count)         # (1, w)
    acc = ones(T, M, 1) .* last_col                      # (M, w)
    @inbounds for p in (n - 1):-1:1
        col = reshape(coefs[:, p], 1, w_count)          # (1, w)
        acc = acc .* reshape(t, M, 1) .+ col
    end
    return acc
end
