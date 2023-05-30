using SkyModels
using ChainRulesTestUtils
using ChainRulesCore
using FiniteDifferences
using Zygote
using FFTW
using Plots
using Statistics
using Test

@testset "SkyModels.jl" begin
    include("models.jl")
end
