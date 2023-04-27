using SkyModels
using Documenter

DocMeta.setdocmeta!(SkyModels, :DocTestSetup, :(using SkyModels); recursive=true)

makedocs(;
    modules=[SkyModels],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/SkyModels.jl/blob/{commit}{path}#{line}",
    sitename="SkyModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/SkyModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/SkyModels.jl",
    devbranch="main",
)
