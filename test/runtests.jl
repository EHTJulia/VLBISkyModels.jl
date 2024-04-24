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
import DimensionalData as DD
import CairoMakie as CM

function FiniteDifferences.to_vec(k::IntensityMap)
    v, b = to_vec(DD.data(k))
    back(x) = DD.rebuild(k, b(x))
    return v, back
end



@testset "VLBISkyModels.jl" begin
    include("models.jl")
    include("templates.jl")
    include("polarized.jl")
    include("utility.jl")
    include("viz.jl")
end
