import ComradeBase: AbstractModel, IsAnalytic, NotAnalytic,
                    IsPolarized, NotPolarized,
                    visanalytic, imanalytic, ispolarized,
                    AbstractDomain
import ComradeBase: visibility_point,
                    intensitymap, intensitymap!, intensity_point,
                    flux

export visibility, amplitude, closure_phase, logclosure_amplitude, bispectrum,
       visibilitymap, amplitudemap, closure_phasemap, logclosure_amplitudemap, bispectummap,
       flux, intensitymap, intensitymap!, PolarizedModel, convolve!

export unpack

"""
    unpack(m::AbstractModel)

Unpacks all elements of a struct into a named tuple. Note that this may
include elements that aren't direcltly model parameters.
"""
function unpack(m::ComradeBase.AbstractModel)
    return ntfromstruct(m)
end

include(joinpath(@__DIR__, "multidomain", "multidomain.jl"))
include(joinpath(@__DIR__, "pulse.jl"))
include(joinpath(@__DIR__, "geometric_models.jl"))
include(joinpath(@__DIR__, "modifiers.jl"))
include(joinpath(@__DIR__, "combinators.jl"))
include(joinpath(@__DIR__, "polarized.jl"))
include(joinpath(@__DIR__, "interpolated.jl"))
include(joinpath(@__DIR__, "continuous_image.jl"))
include(joinpath(@__DIR__, "test.jl"))
include(joinpath(@__DIR__, "misc.jl"))
include(joinpath(@__DIR__, "clean.jl"))
include(joinpath(@__DIR__, "numerical.jl"))
include(joinpath(@__DIR__, "imagetemplates", "imagetemplates.jl"))
