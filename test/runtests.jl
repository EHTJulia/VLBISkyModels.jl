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
import CairoMakie as CM


@testset "VLBISkyModels.jl" begin
    include("models.jl")
    include("polarized.jl")
    include("utility.jl")
end
