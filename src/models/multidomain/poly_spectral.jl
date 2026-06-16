export PolySpectral, TaylorSpectral, build_param, build_param!
import ComradeBase: build_param

@doc """
    PolySpectral(index::NTuple{N}, freq0::Number, p0=0.0)
    PolySpectral(index::Number, freq0::Number, p0=0.0)
    PolySpectral(params::Number, index, freq0::Number, p0=zero(params))
    PolySpectral(params::AbstractArray, index, freq0::Number, p0=zero(params))

A frequency-dependent domain model that expands a parameter in a polynomial series:

    params * exp(∑ₙ index[n] * log(Fr / freq0)^n) + p0

where `Fr` is the observation frequency, `freq0` is the reference frequency,
and `index` defines the polynomial coefficients in log-frequency space.

# Arguments
- `index`: polynomial coefficients. An `NTuple{N}` for order-N expansion, or a
  single `Number` for order-1 (spectral index). Can also be an `NTuple` of
  `AbstractArray`s for spatially varying coefficients.
- `freq0`: reference frequency. The model evaluates to `params` at `Fr = freq0`.
- `params` (optional): base value at the reference frequency. 
- `p0` (optional): additive offset term. Defaults to `zero(params)`.

# Constructor variants

**Geometric modeling — no base param (returns raw `PolySpectral`):**
Used directly as a frequency-dependent argument to modifiers like `Stretch`, `Rotate`, `Shift`.
```julia
ps = PolySpectral(1.0, 230.0e9)        # spectral index=1, ref freq=230 GHz
ps = PolySpectral(1.0, 230.0e9, -1.0)  # with offset p0=-1
ps = PolySpectral((1.0, 0.5), 230.0e9) # order-2: index + curvature
modify(Gaussian(), Stretch(ps, 1.0))    # frequency-dependent stretch
```

**Geometric modeling — with scalar base param (returns raw `PolySpectral`):**
Like above but with an explicit base value baked in.
```julia
ts = PolySpectral(2.0, 1.0, 230.0e9)  # base value=2, spectral index=1
```

**Imaging — with array base param (returns `MultiDomainParams`):**
Used for frequency-dependent image cubes where `params` is the reference image.
```julia
ps = PolySpectral(base_image, (α, β), 230.0e9)  # spatially varying index+curvature
ps = PolySpectral(base_image, 1.0, 230.0e9)      # uniform spectral index
ComradeBase.build_param(ps, (; Fr = 345.0e9))    # returns image at 345 GHz
```

# Polynomial expansion
- `N=1`: `exp(index[1] * log(Fr/freq0))` — spectral index only.
- `N=2`: `exp(index[1] * log(Fr/freq0) + index[2] * log(Fr/freq0)²)` — spectral index + spectral curvure
- Higher orders include even higher order structure

`TaylorSpectral` is an alias for `PolySpectral` for backwards compatability.
"""
struct PolySpectral{N, P, T <: NTuple{N}, F <: Number, P0} <: ComradeBase.FrequencyParams{Any}
    params::P # to remove in the future when build_param -> build_param(param, model, p)
    index::T
    freq0::F
    p0::P0
end

# feed in a single params: geometric modeling
function PolySpectral(params, index, freq0, p0 = zero(params)) where {N}
    return PolySpectral(params, (index,), freq0, p0)
end

# don't feed in params: imaging
function PolySpectral(index::NTuple{N}, freq0::Number, p0 = 0.) where {N}
    return PolySpectral{N, Nothing, typeof(index), typeof(freq0), typeof(p0)}(nothing, index, freq0, p0)
end

# wrap index in tuples
#PolySpectral(params, index, freq0, p0 = zero(params)) = PolySpectral(params, (index,), freq0, p0)
#PolySpectral(index, freq0, p0 = 0.0) = PolySpectral((index,), freq0, p0)

# spectral model expansion
function arg_expand(domain::PolySpectral{N}, p) where {N}
    x = log(p.Fr / domain.freq0)
    return reduce(+, ntuple(n -> @inbounds(domain.index[n]) .* x^n, Val(N)))
end

# 2 -> 3 argument conversion
function ComradeBase.build_param(domain::PolySpectral, p)
    return ComradeBase.build_param(domain.params, domain, p)
end

# if param doesn't exist and build_param is called on PolySpectral
# just evaluate the argument and add p0
function ComradeBase.build_param(params::Nothing, domain::PolySpectral, p)
    arg = arg_expand(domain, p)
    return @. exp(arg) .+ domain.p0 
end

# if param exists and build_param is called on PolySpectral
# do the full expansion
function ComradeBase.build_param(params, domain::PolySpectral, p)
    arg = arg_expand(domain, p)
    return @. params * exp(arg) + domain.p0 
end

### MUTATING VER. ###

# if unrolled via MultiDomainParams for multidomain imaging
# array version
function build_param!(param::AbstractArray, domain::PolySpectral, p)
    arg = arg_expand(domain, p)
    param  .= param .* exp.(arg) .+ domain.p0
    return param
end

# number version
function build_param!(param::Number, domain::PolySpectral, p)
    arg = arg_expand(domain, p)
    return param * exp(arg) + domain.p0
end

@doc """
    TaylorSpectral

Deprecated alias for [`PolySpectral`](@ref). Use `PolySpectral` instead.
"""
TaylorSpectral
const TaylorSpectral = PolySpectral