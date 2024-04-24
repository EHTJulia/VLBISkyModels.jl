module VLBISkyModels

using Accessors: @set
using ArgCheck
using AbstractFFTs
using BasicInterpolators
using ChainRulesCore
using ForwardDiff
using DocStringExtensions
using DelimitedFiles
using Enzyme, EnzymeCore
using FFTW
using FillArrays
using NFFT
using NamedTupleTools
using PaddedViews
using Reexport
using RecipesBase
using SpecialFunctions
import Static
using StaticArrays
using StructArrays
using LinearAlgebra
using Printf
using Serialization

@reexport using ComradeBase
@reexport using PolarizedTypes
@reexport using DimensionalData
const DD = DimensionalData

"""
    rad2μas(x)
Converts a number from radians to micro-arcseconds (μas)
"""
@inline rad2μas(x) = 180*3600*1_000_000*x/π

"""
    μas2rad(x)
Converts a number from micro-arcseconds (μas) to rad
"""
@inline μas2rad(x) = x/(180*3600*1_000_000)*π

export linearpol, mbreve, evpa, rad2μas, μas2rad, create_cache

using ComradeBase: AbstractDomain, AbstractSingleDomain, AbstractRectiGrid,
                   AbstractModel, AbstractPolarizedModel,
                   UnstructuredDomain, RectiGrid

import ComradeBase: flux, radialextent, intensitymap, intensitymap!,
                    intensitymap_analytic, intensitymap_analytic!,
                    intensitymap_numeric, intensitymap_numeric!,
                    visibilitymap, visibilitymap!,
                    _visibilitymap, _visibilitymap!,
                    visibilitymap_analytic, visibilitymap_analytic!,
                    visibilitymap_numeric, visibilitymap_numeric!,
                    closure_phase, closure_phasemap,
                    logclosure_amplitude, logclosure_amplitudemap,
                    bispectrummap, bispectrum, allocate_imgmap, allocate_vismap,
                    create_vismap, create_imgmap


# Write your package code here.
include("stokes_image.jl")
const IntensityMapTypes{T,N} = Union{IntensityMap{T,N}, StokesIntensityMap{T,N}}
include(joinpath("fourierdomain", "fourierdomain.jl"))
include(joinpath("models", "models.jl"))
include("utility.jl")
include("rules.jl")
include(joinpath("visualizations", "vis.jl"))

if !isdefined(Base, :get_extension)
    using Requires
end

@static if !isdefined(Base, :get_extension)
    function __init__()
        @require Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include(joinpath(@__DIR__, "..", "ext", "VLBISkyModelsMakieExt.jl"))
    end
end


end
