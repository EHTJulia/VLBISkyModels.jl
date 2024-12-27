export stretched, shifted, rotated, renormed, modify, Stretch, Renormalize, Shift, Rotate

"""
    $(TYPEDEF)

General type for a model modifier. These transform any model
using simple Fourier transform properties. To modify a model
you can use the [`ModifiedModel`](@ref) constructor or the [`modify`](@ref)
function.

```julia-repl
julia> visanalytic(stretched(Disk(), 2.0, 2.0)) == visanalytic(Disk())
true
```



To implement a model transform you need to specify the following methods:
- [`transform_uv`](@ref)
- [`transform_image`](@ref)
- [`scale_uv`](@ref)
- [`scale_image`](@ref)
- [`radialextent`](@ref)
See thee docstrings of those methods for guidance on implementation details.

Additionally these methods assume the modifiers are of the form

I(x,y) -> fᵢ(x,y)I(gᵢ(x,y))
V(u,v) -> fᵥ(u,v)V(gᵥ(u,v))

where `g` are the transform_image/uv functions and `f` are the scale_image/uv
function.

"""
abstract type ModelModifier{T} end

function ComradeBase.getparam(m::ModelModifier, s::Symbol, p)
    return ComradeBase.build_param(getproperty(m, s), p)
end
function ComradeBase.getparam(m::ModelModifier, ::Val{s}, p) where {s}
    return ComradeBase.build_param(getproperty(m, s), p)
end

"""
    scale_image(model::AbstractModifier, x, y)

Returns a number of how to to scale the image intensity at `x` `y` for an modified `model`
"""
function scale_image end

"""
    transform_image(model::AbstractModifier, x, y)

Returns a transformed `x` and `y` according to the `model` modifier
"""
function transform_image end

"""
    scale_image(model::AbstractModifier, u, u)

Returns a number on how to scale the image visibility at `u` `v` for an modified `model`
"""
function scale_uv end

"""
    transform_uv(model::AbstractModifier, u, u)

Returns a transformed `u` and `v` according to the `model` modifier
"""
function transform_uv end

unitscale(T, ::NotPolarized) = one(T)
unitscale(T, ::IsPolarized) = I

"""
    $(TYPEDEF)

Container type for models that have been transformed in some way.
For a list of potential modifiers or transforms see `subtypes(ModelModifiers)`.

# Fields
$(FIELDS)
"""
struct ModifiedModel{M,MT<:Tuple} <: AbstractModel
    """base model"""
    model::M
    """model transforms"""
    transform::MT
end

function Base.show(io::IO, mi::ModifiedModel)
    #io = IOContext(io, :compact=>true)
    #s = summary(mi)
    println(io, "ModifiedModel")
    println(io, "  base model: ", mi.model)
    println(io, "  Modifiers:")
    for i in eachindex(mi.transform)[1:(end - 1)]
        println(io, "    $i. ", summary(mi.transform[i]))
    end
    return print(io, "    $(length(mi.transform)). ", summary(mi.transform[end]))
end

"""
    unmodified(model::ModifiedModel)

Returns the un-modified model

### Example
```julia-repl
julia> m = stretched(rotated(Gaussian(), π/4), 2.0, 1.0)
julia> umodified(m) == Gaussian()
true
```
"""
unmodified(model::ModifiedModel) = model.model
unmodified(model::AbstractModel) = model

"""
    basemodel(model::ModifiedModel)

Returns the ModifiedModel with the last transformation stripped.

# Example
```julia-repl
julia> basemodel(stretched(Disk(), 1.0, 2.0)) == Disk()
true
```
"""
basemodel(model::ModifiedModel) = ModifiedModel(model.model, Base.front(model.transform))
basemodel(model::ModifiedModel{M,Tuple{}}) where {M} = model

flux(m::ModifiedModel) = flux(m.model) * mapreduce(flux, Base.:*, m.transform)
flux(::ModelModifier{T}) where {T} = one(T)

radialextent(m::ModifiedModel) = radialextent_modified(radialextent(m.model), m.transform)

@inline visanalytic(::Type{ModifiedModel{M,T}}) where {M,T} = visanalytic(M)
@inline imanalytic(::Type{ModifiedModel{M,T}}) where {M,T} = imanalytic(M)
@inline ispolarized(::Type{ModifiedModel{M,T}}) where {M,T} = ispolarized(M)

@inline function ModifiedModel(m, t::ModelModifier)
    return ModifiedModel(m, (t,))
end

@inline function ModifiedModel(m::ModifiedModel, t::ModelModifier)
    model = m.model
    t0 = m.transform
    return ModifiedModel(model, (t0..., t))
end

@inline doesnot_uv_modify(t::Tuple) = doesnot_uv_modify(Base.front(t)) *
                                      doesnot_uv_modify(last(t))
@inline doesnot_uv_modify(::Tuple{}) = true

function modify_uv(model, t::Tuple, p, scale)
    pt = transform_uv(model, last(t), p)
    scalet = scale_uv(model, last(t), p)
    return modify_uv(model, Base.front(t), pt, scalet * scale)
end
modify_uv(model, ::Tuple{}, p, scale) = p, scale

function modify_image(model, t::Tuple, p, scale)
    pt = transform_image(model, last(t), p)
    scalet = scale_image(model, last(t), p)
    return modify_image(model, Base.front(t), pt, scalet * scale)
end
modify_image(model, ::Tuple{}, p, scale) = p, scale

# @inline function transform_uv(model, t::Tuple, u, v)
#     ut, vt = transform_uv(model, last(t), u, v)
#     return transform_uv(model, Base.front(t), ut, vt)
# end
# @inline transform_uv(model, ::Tuple{}, u, v) = u, v

# @inline function scale_uv(model, t::Tuple, u, v)
#     scale = scale_uv(model, last(t), u, v)
#     return scale*scale_uv(model, Base.front(t), u, v)
# end
# @inline scale_uv(::M, t::Tuple{}, u, v) where {M} = unitscale(eltype(u), M)

# @inline function transform_image(model, t::Tuple, x, y)
#     xt, yt = transform_image(model, last(t), x, y)
#     return transform_image(model, Base.front(t), xt, yt)
# end
# @inline transform_image(model, ::Tuple{}, x, y) = x, y

# @inline function scale_image(model, t::Tuple, x, y)
#     scale = scale_image(model, last(t), x, y)
#     return scale*scale_image(model, Base.front(t), x, y)
# end
# @inline scale_image(::M, t::Tuple{}, x, y) where {M} = unitscale(eltype(x), M)

@inline radialextent_modified(r::Real, t::Tuple) = radialextent_modified(radialextent_modified(r,
                                                                                               last(t)),
                                                                         Base.front(t))
@inline radialextent_modified(r::Real, ::Tuple{}) = r

"""
    modify(m::AbstractModel, transforms...)

Modify a given `model` using the set of `transforms`. This is the most general
function that allows you to apply a sequence of model transformation for example

```julia-repl
modify(Gaussian(), Stretch(2.0, 1.0), Rotate(π/4), Shift(1.0, 2.0), Renorm(2.0))
```
will create a asymmetric Gaussian with position angle `π/4` shifted to the position
(1.0, 2.0) with a flux of 2 Jy. This is similar to Flux's chain.
"""
function modify(m::AbstractModel, transforms...)
    return ModifiedModel(m, transforms)
end

# @inline function apply_uv_transform(m::AbstractModifier, t::TransformState)
#     ut, vt = transform_uv(m, t.u, t.v)
#     scale = t.scale*scale_uv(m, t.u, t.v)
#     return apply_uv_transform(basemodel(m), TransformState(ut, vt, scale))
# end

# @inline function apply_uv_transform(::AbstractModel, t::TransformState)
#     return t
# end

# @inline function apply_uv_transform(m::AbstractModifier, u, v, scale)
#     ut, vt = transform_uv(m, u, v)
#     scale = scale*scale_uv(m, u, v)
#     return apply_uv_transform(basemodel(m), ut, vt, scale)
# end

# @inline function apply_uv_transform(::AbstractModel, u, v, scale)
#     return (u, v), scale
# end

# @inline function _visibilitymap(m::AbstractModifier, u, v, time, freq)
#     uv, scale = apply_uv_transform(m, u, v)
#     ut = first.(uv)
#     vt = last.(uv)
#     scale.*_visibilitymap(unmodified(m), ut, vt, time, freq)
# end

# # function visibilitymap(m, p::NamedTuple)
# #     m = Base.Fix1(m∘NamedTuple{keys(p)})
# #     return visibilitymap(m, NamedTuple{keys(p)}(p))
# # end

# function apply_uv_transform(m::AbstractModifier, u::AbstractVector, v::AbstractVector)
#     res = apply_uv_transform.(Ref(m), u, v, 1.0)
#     return first.(res), last.(res)
# end

# function apply_uv_transform(m::AbstractModifier, u::AbstractVector, v::AbstractVector)
#     res = apply_uv_transform.(Ref(m), u, v, 1.0)
#     return getindex.(res,1), getindex.(res,2), getindex.(res,3)
#     res = apply_uv_transform.(Ref(m), u, v, 1.0)
#     return getindex.(res,1), getindex.(res,2), getindex.(res,3)
# end
# @inline function _visibilitymap(m::M, p) {M<:AbstractModifier}

# @inline function _visibilitymap(m::M, p) {M<:AbstractModifier}

#     return _visibilitymap(visanalytic(M), m, u, v, args...)
# end

# @inline function _visibilitymap(m::AbstractModifier{M}, p) where {M}
#     return _visibilitymap(ispolarized(M), m, p)
# end

# @inline function _visibilitymap(m::AbstractModifier, p)
#     (;U, V) = p
#     st = StructArray{TransformState{eltype(U), Complex{eltype(U)}}}(u=U, v=V, scale=fill(one(Complex{eltype(U)}), length(U)))
#     auv = Base.Fix1(apply_uv_transform, m)
#     mst = map(auv, st)
#     mst.scale.*visibilitymap(unmodified(m), (U=mst.u, V=mst.v))
# end

function update_uv(p::NamedTuple, uv)
    p1 = @set p.U = uv.U
    p2 = @set p1.V = uv.V
    return p2
end

function update_xy(p::NamedTuple, xy)
    p1 = @set p.X = xy.X
    p2 = @set p1.Y = xy.Y
    return p2
end

# @inline function _visibilitymap(::IsPolarized, m::AbstractModifier, p)
#     (;U, V) = p

#     S = eltype(U)
#     unit = StokesParams(complex(one(S)), complex(one(S)), complex(one(S)),complex(one(S)))
#     st = StructArray{TransformState{eltype(U), typeof(unit)}}(u=U, v=V, scale=Fill(unit, length(U)))
#     mst = apply_uv_transform.(Ref(m), st)

#     pup = update_uv(p, (U=mst.u, V=mst.v))
#     mst.scale.*visibilitymap(unmodified(m), pup)
# end

# @inline function _visibilitymap(m::AbstractModifier, p)
#     (;U, V) = p
#     st = StructArray{TransformState{eltype(U), Complex{eltype(U)}}}(u=U, v=V, scale=fill(one(Complex{eltype(U)}), length(U)))
#     mst = apply_uv_transform.(Ref(m), st)
#     pup = update_uv(p, (U=mst.u, V=mst.v))
#     mst.scale.*visibilitymap(unmodified(m), pup)
# end

@inline function visibility_point(m::M, p) where {M<:ModifiedModel}
    mbase = m.model
    transform = m.transform
    ispol = ispolarized(M)
    pt, scale = modify_uv(ispol, transform, p, unitscale(Complex{eltype(p.U)}, ispol))
    return scale * visibility_point(mbase, pt)
end

@inline function intensity_point(m::M, p) where {M<:ModifiedModel}
    mbase = m.model
    transform = m.transform
    ispol = ispolarized(M)
    pt, scale = modify_image(ispol, transform, p, unitscale(eltype(p.X), ispol))
    return scale * intensity_point(mbase, pt)
end

# function modify_uv(model, transform::Tuple, p, scale)
#     uvscale = modify_uv.(Ref(model), Ref(transform), StructArray(p), scale)
#     return first.(uvscale), last.(uvscale)
# end

function visibilitymap_analytic(m::ModifiedModel, p::AbstractFourierDualDomain)
    return visibilitymap_analytic(m, visdomain(p))
end

function visibilitymap_analytic(m::ModifiedModel, p::AbstractSingleDomain)
    vis = allocate_vismap(m, p)
    visibilitymap_analytic!(vis, m)
    return vis
end

# function __extract_tangent(dm::ModifiedModel)
#     tm = __extract_tangent(dm.model)
#     dtm = dm.transform
#     ttm = map(x -> Tangent{typeof(x)}(; ntfromstruct(x)...), dtm)
#     return tm = Tangent{typeof(dm)}(; model=tm, transform=ttm)
# end

# function __extract_tangent(dm::GeometricModel)
#     ntm = ntfromstruct(dm)
#     if ntm isa NamedTuple{(),Tuple{}}
#         tbm = ZeroTangent()
#     else
#         tbm = Tangent{typeof(dm)}(; ntm...)
#     end
#     return tbm
# end

"""
    $(TYPEDEF)

Shifts the model by `Δx` units in the x-direction and `Δy` units
in the y-direction.

# Example
```julia-repl
julia> modify(Gaussian(), Shift(2.0, 1.0)) == shifted(Gaussian(), 2.0, 1.0)
true
```
"""
struct Shift{T} <: ModelModifier{T}
    Δx::T
    Δy::T
end

#function ShiftedModel(model::AbstractModel, Δx::Number, Δy::Number)
#    T =
#    return ShiftedModel{(model, promote(Δx, Δy)...)
#end

"""
    $(SIGNATURES)

Shifts the model `m` in the image domain by an amount `Δx,Δy`
in the x and y directions respectively.
"""
shifted(model, Δx, Δy) = ModifiedModel(model, Shift(Δx, Δy))

doesnot_uv_modify(::Shift) = true

# This is a simple overload to simplify the type system
@inline radialextent_modified(r::Real, t::Shift) = r + max(abs(t.Δx), abs(t.Δy))

@inline function transform_image(model, transform::Shift, p)
    @unpack_params Δx, Δy = transform(p)
    X = p.X - Δx
    Y = p.Y - Δy
    return update_xy(p, (; X, Y))
end

@inline transform_uv(model, ::Shift, p) = p

@inline scale_image(m, ::Shift, p) = unitscale(p.X, m)
# Curently we use exp here because enzyme has an issue with cispi that will be fixed soon.
@inline function scale_uv(m, transform::Shift, p)
    @unpack_params Δx, Δy = transform(p)
    (; U, V) = p
    T = typeof(Δx)
    return exp(2im * T(π) *
               (U * Δx + V * Δy)) *
           unitscale(T, m)
end

"""
    $(TYPEDEF)

Renormalizes the flux of the model to the new value `scale*flux(model)`.
We have also overloaded the Base.:* operator as syntactic sugar
although I may get rid of this.


# Example
```julia-repl
julia> modify(Gaussian(), Renormalize(2.0)) == 2.0*Gaussian()
true
```
"""
struct Renormalize{T} <: ModelModifier{T}
    scale::T
end

"""
    $(SIGNATURES)

Renormalizes the model `m` to have total flux `f*flux(m)`.
This can also be done directly by calling `Base.:*` i.e.,

```julia-repl
julia> renormed(m, f) == f*M
true
```
"""
renormed(model::M, f) where {M<:AbstractModel} = ModifiedModel(model, Renormalize(f))
@inline doesnot_uv_modify(::Renormalize) = true

const ModNum = Union{Number,ComradeBase.DomainParams}

Base.:*(model::AbstractModel, f::ModNum) = renormed(model, f)
Base.:*(f::ModNum, model::AbstractModel) = renormed(model, f)
Base.:/(f::ModNum, model::AbstractModel) = renormed(model, inv(f))
Base.:/(model::AbstractModel, f::ModNum) = renormed(model, inv(f))
# Dispatch on RenormalizedModel so that I just make a new RenormalizedModel with a different f
# This will make it easier on the compiler.
# Base.:*(model::ModifiedModel, f::Number) = renormed(model.model, model.scale*f)
# Overload the unary negation operator to be the same model with negative flux
Base.:-(model::AbstractModel) = renormed(model, -1)
flux(t::Renormalize) = t.scale

@inline transform_image(m, ::Renormalize, p) = p
@inline transform_uv(m, ::Renormalize, p) = p

@inline function scale_image(m, transform::Renormalize, p)
    @unpack_params scale = transform(p)
    return scale * unitscale(typeof(scale), m)
end

@inline function scale_uv(m, transform::Renormalize, p)
    @unpack_params scale = transform(p)
    return scale * unitscale(typeof(scale), m)
end

@inline radialextent_modified(r::Real, ::Renormalize) = r

"""
    Stretch(α, β)
    Stretch(r)

Stretched the model in the x and y directions, i.e. the new intensity is
    Iₛ(x,y) = 1/(αβ) I(x/α, y/β),
where were renormalize the intensity to preserve the models flux.

# Example
```julia-repl
julia> modify(Gaussian(), Stretch(2.0)) == stretched(Gaussian(), 2.0, 1.0)
true
```

If only a single argument is given it assumes the same stretch is applied in both direction.

```julia-repl
julia> Stretch(2.0) == Stretch(2.0, 2.0)
true
```
"""
struct Stretch{T} <: ModelModifier{T}
    α::T
    β::T
end

Stretch(r) = Stretch(r, r)

"""
    $(SIGNATURES)

Stretches the model `m` according to the formula
    Iₛ(x,y) = 1/(αβ) I(x/α, y/β),
where were renormalize the intensity to preserve the models flux.
"""
stretched(model, α, β) = ModifiedModel(model, Stretch(α, β))
stretched(model, α) = stretched(model, α, α)

@inline doesnot_uv_modify(::Stretch) = false
@inline function transform_image(m, transform::Stretch, p)
    (; X, Y) = p
    @unpack_params α, β = transform(p)
    # @show p
    pt = update_xy(p, (; X=X / α, Y=Y / β))
    # @show pt
    return pt
end

@inline function transform_uv(m, transform::Stretch, p)
    (; U, V) = p
    @unpack_params α, β = transform(p)
    return update_uv(p, (; U=U * α, V=V * β))
end

@inline function scale_image(m, transform::Stretch, p)
    @unpack_params α, β = transform(p)
    T = typeof(α)
    return inv(α * β) * unitscale(T, m)
end

@inline scale_uv(m, tr::Stretch, p) = unitscale(typeof(getparam(tr, :α, p)), m)

@inline radialextent_modified(r::Real, t::Stretch) = r * max(t.α, t.β)

"""
    Rotate(ξ)

Type for the rotated model. This is more fine grained constrol of
rotated model.

# Example
```julia-repl
julia> modify(Gaussian(), Rotate(2.0)) == rotated(Gaussian(), 2.0)
true
```
"""
struct Rotate{T} <: ModelModifier{T}
    s::T
    c::T
    function Rotate(ξ::F) where {F<:Real}
        s, c = sincos(ξ)
        return new{F}(s, c)
    end
    function Rotate(ξ::ComradeBase.DomainParams)
        return new{eltype(ξ)}(ξ, ξ)
    end
end

function ComradeBase.getparam(m::Rotate{T},
                              s::Symbol,
                              p) where {T<:ComradeBase.DomainParams}
    m = getproperty(m, s)
    mr = Rotate(ComradeBase.build_param(m, p))
    return getproperty(mr, s)
end

"""
    $(SIGNATURES)

Rotates the model by an amount `ξ` in radians in the clockwise direction.
"""
rotated(model, ξ) = ModifiedModel(model, Rotate(ξ))

"""
    $(SIGNATURES)

Returns the rotation angle of the rotated `model`
"""
posangle(model::Rotate) = atan(model.s, model.c)

@inline doesnot_uv_modify(::Rotate) = false

@inline function transform_image(m, transform::Rotate, p)
    @unpack_params s, c = transform(p)
    (; X, Y) = p
    Xr = c * X - s * Y
    Yr = s * X + c * Y
    pt = update_xy(p, (; X=Xr, Y=Yr))
    return pt
end

@inline function transform_uv(m, transform::Rotate, p)
    @unpack_params s, c = transform(p)
    (; U, V) = p
    Ur = c * U - s * V
    Vr = s * U + c * V
    return update_uv(p, (; U=Ur, V=Vr))
end

@inline scale_image(::NotPolarized, model::Rotate, p) = one(typeof(getparam(model, :s, p)))

@inline function spinor2_rotate(c, s)
    u = oneunit(c)
    z = zero(s)
    c2 = c^2 - s^2
    s2 = 2 * c * s
    return SMatrix{4,4}(u, z, z, z,
                        z, c2, s2, z,
                        z, -s2, c2, z,
                        z, z, z, u)
end

@inline function scale_image(::IsPolarized, model::Rotate, p)
    @unpack_params s, c = model(p)
    return spinor2_rotate(c, s)
end

@inline scale_uv(::NotPolarized, model::Rotate, p) = one(typeof(getparam(model, :s, p)))

@inline function scale_uv(::IsPolarized, model::Rotate, p)
    @unpack_params s, c = model(p)
    return spinor2_rotate(c, s)
end
@inline radialextent_modified(r::Real, ::Rotate) = r
