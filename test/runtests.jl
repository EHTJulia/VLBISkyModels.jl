using VLBISkyModels
using ChainRulesTestUtils
using ChainRulesCore
using FiniteDifferences
using Zygote
using FFTW
using Plots
using Statistics
using Test
using Serialization
using StructArrays
import DimensionalData as DD
import CairoMakie as CM
using ForwardDiff
using Enzyme
using LinearAlgebra
using Downloads
using BenchmarkTools

function FiniteDifferences.to_vec(k::IntensityMap)
    v, b = to_vec(DD.data(k))
    back(x) = DD.rebuild(k, b(x))
    return v, back
end

function FiniteDifferences.to_vec(k::UnstructuredMap)
    v, b = to_vec(parent(k))
    back(x) = UnstructuredMap(b(x), axisdims(k))
    return v, back
end




@testset "VLBISkyModels.jl" begin
    include("models.jl")
    include("templates.jl")
    include("polarized.jl")
    include("utility.jl")
    include("viz.jl")
    include("stokesintensitymap.jl")
    include("multidomain.jl")
end
