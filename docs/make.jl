using Pkg
script_dir = @__DIR__
Pkg.activate(script_dir)
parent_dir = dirname(script_dir)
Pkg.develop(PackageSpec(path=parent_dir))
Pkg.instantiate()

using CairoMakie
using VLBISkyModels
using Documenter
using Literate
using Pkg
using Glob

Pkg.develop(PackageSpec(url="https://github.com/ptiede/ComradeBase.jl"))

GENERATED = joinpath(@__DIR__, "../", "examples")
OUTDIR = joinpath(@__DIR__, "src", "examples")

SOURCE_FILES = Glob.glob("*.jl", GENERATED)
foreach(fn -> Literate.markdown(fn, OUTDIR, documenter=true), SOURCE_FILES)

Pkg.activate(@__DIR__)
makedocs(;
    modules=[VLBISkyModels, ComradeBase, PolarizedTypes],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    # repo="https://github.com/EHTJulia/VLBISkyModels.jl/blob/{commit}{path}#{line}",
    sitename="VLBISkyModels.jl",
    format=Documenter.HTML(;
        prettyurls=true, # always on, avoids confusion when building locally. If needed, serve the "build" folder locally with LiveServer. #get(ENV, "CI", "false") == "true",
        canonical="https://ehtjulia.github.io/VLBISkyModels.jl",
        edit_link="main",
        assets=String[],
        ),
    draft=false,
    pages=[
        "Home" => "index.md",
        "interface.md",
        "api.md",
        "base_api.md",
        "Non-Analytic Modeling"=>joinpath("examples", "nonanalytic.md"),
    ],

)

deploydocs(;
    repo="github.com/EHTJulia/VLBISkyModels.jl",
    devbranch="main",
)
