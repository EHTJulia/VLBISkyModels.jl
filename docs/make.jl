using VLBISkyModels
using Documenter
using Plots
using Literate
using Pkg

Pkg.develop(PackageSpec(url="https://github.com/ptiede/ComradeBase.jl"))

GENERATED = joinpath(@__DIR__, "../", "examples")
OUTDIR = joinpath(@__DIR__, "src", "examples")

SOURCE_FILES = Glob.glob("*.jl", GENERATED)
foreach(fn -> Literate.markdown(fn, OUTDIR, documenter=true), SOURCE_FILES)

MD_FILES = [
            joinpath("examples", "nonanalytic.md"),
            ]



makedocs(;
    modules=[VLBISkyModels],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/EHTJulia/VLBISkyModels.jl/blob/{commit}{path}#{line}",
    sitename="VLBISkyModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ehtjulia.github.io/VLBISkyModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "interface.md",
        "api.md"
    ],
)

deploydocs(;
    repo="https://github.com/EHTJulia/VLBISkyModels.jl",
    devbranch="main",
)
