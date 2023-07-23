using VLBISkyModels
using Documenter
using Plots
using Literate
using Pkg
using Glob

Pkg.develop(PackageSpec(url="https://github.com/ptiede/ComradeBase.jl"))

GENERATED = joinpath(@__DIR__, "../", "examples")
OUTDIR = joinpath(@__DIR__, "src", "examples")

SOURCE_FILES = Glob.glob("*.jl", GENERATED)
foreach(fn -> Literate.markdown(fn, OUTDIR, documenter=true), SOURCE_FILES)


makedocs(;
    modules=[VLBISkyModels],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/EHTJulia/VLBISkyModels.jl/blob/{commit}{path}#{line}",
    sitename="VLBISkyModels.jl",
    format=Documenter.HTML(),
        # prettyurls=get(ENV, "CI", "false") == "true",
        # edit_link="main",
        # assets=String[],
     draft=false,
    pages=[
        "Home" => "index.md",
        "interface.md",
        "api.md",
        joinpath("examples", "nonanalytic.md"),
    ],
)

deploydocs(;
    repo="github.com/EHTJulia/VLBISkyModels.jl",
    devbranch="main",
    push_preview=false
)
