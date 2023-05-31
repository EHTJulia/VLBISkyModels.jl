using VLBISkyModels
using ChainRulesTestUtils
using ChainRulesCore
using FiniteDifferences
using Zygote
using FFTW
using Plots
using Statistics
using Test

@testset "VLBISkyModels.jl" begin
    include("models.jl")
    include("polarized.jl")
end
