import ComradeBase: AbstractFourierDualDomain, FourierTransform,
       forward_plan, reverse_plan, imgdomain, visdomain,
       algorithm, getplan, getphases, create_plans


struct FourierDualDomain{
        ID <: AbstractSingleDomain, VD <: AbstractSingleDomain,
        A <: FourierTransform, PI <: AbstractPlan,
        PD <: AbstractPlan,
    } <: AbstractFourierDualDomain
    imgdomain::ID
    visdomain::VD
    algorithm::A
    plan_forward::PI
    plan_reverse::PD
end

function Base.show(io::IO, mime::MIME"text/plain", g::FourierDualDomain)
    printstyled(io, "FourierDualDomain("; bold = true)
    println(io)
    printstyled(io, "Algorithm: "; bold = true)
    show(io, mime, algorithm(g))
    printstyled(io, "\nImage Domain: "; bold = true)
    show(io, mime, imgdomain(g))
    printstyled(io, "\nVisibility Domain: "; bold = true)
    return show(io, mime, visdomain(g))
end

function Serialization.serialize(
        s::Serialization.AbstractSerializer,
        cache::FourierDualDomain
    )
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(cache))
    Serialization.serialize(s, cache.algorithm)
    Serialization.serialize(s, cache.imgdomain)
    return Serialization.serialize(s, cache.visdomain)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:FourierDualDomain})
    alg = Serialization.deserialize(s)
    imd = Serialization.deserialize(s)
    vid = Serialization.deserialize(s)
    return FourierDualDomain(imd, vid, alg)
end

"""
    uviterator(nx, dx, ny dy)

Construct the u,v iterators for the Fourier transform of the image
with pixel sizes `dx, dy` and number of pixels `nx, ny`

If you are extending Fourier transform stuff please use these functions
to ensure that the centroid is being properly computed.
"""
function uviterator(nx, dx, ny, dy)
    u = fftshift(fftfreq(nx, inv(dx)))
    v = fftshift(fftfreq(ny, inv(dy)))
    return U(u), V(v)
end

"""
    uvgrid(grid::AbstractRectiGrid)

Converts from a image domain recti-grid with spatial dimension (X,Y)
to a Fourier or visibility domain grid with spatial dimensions (U,V).
Note the other dimensions are not changed.

For the inverse see [`xygrid`](@ref)
"""
function uvgrid(grid::AbstractRectiGrid)
    (; X, Y) = grid
    uvg = uviterator(length(X), step(X), length(Y), step(Y))
    pft = dims(grid)[3:end]
    puv = (uvg..., pft...)
    g = rebuild(grid; dims = puv)
    return g
end

function ifftfreq(n::Int, fs::Real)
    if iseven(n)
        x = LinRange(-n ÷ 2, n ÷ 2 - 1, n) * fs / n
        return x .+ step(x) / 2
    else
        n = n - 1
        x = LinRange(-n ÷ 2, n ÷ 2, n + 1) * fs / (n + 1)
        return x
    end
end

function xyiterator(nu, du, nv, dv)
    X = ifftfreq(nu, inv(du))
    Y = ifftfreq(nv, inv(dv))
    return (; X, Y)
end

"""
    xygrid(grid::AbstractRectiGrid)

Converts from a visi1bility domain recti-grid with spatial dimensions
(U,V) to a image domain grid with spatial dimensions (X,Y). Note the
other dimensions are not changed.

For the inverse see [`uvgrid`](@ref)
"""
function xygrid(grid::AbstractRectiGrid)
    (; U, V) = grid
    x, y = xyiterator(length(U), step(U), length(V), step(V))
    pxy = merge((X = X(x), Y = Y(y)), delete(named_dims(grid), (:U, :V)))
    g = rebuild(grid; dims = pxy)
    return g
end

"""
    FourierDualDomain(imgdomain::AbstractSingleDomain, visdomain::AbstractSingleDomain, algorithm)

Constructs a set of grids that live in the image and visibility domains. The transformation between the grids
is specified by the `algorithm` which is a subtype of `VLBISkyModels.FourierTransform`.

# Arguments

  - `imgdomain`: The image domain grid
  - `visdomain`: The visibility domain grid
  - `algorithm`: The Fourier transform algorithm to use see `subtypes(VLBISkyModels.FourierTransform)` for a list
"""
function FourierDualDomain(
        imgdomain::AbstractSingleDomain, visdomain::AbstractSingleDomain,
        algorithm
    )
    plan_forward, plan_reverse = create_plans(algorithm, imgdomain, visdomain)
    return FourierDualDomain(imgdomain, visdomain, algorithm, plan_forward, plan_reverse)
end


include("fft_alg.jl")
include("nuft/nuft.jl")
