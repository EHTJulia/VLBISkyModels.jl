using VLBISkyModels
using Documenter

DocMeta.setdocmeta!(VLBISkyModels, :DocTestSetup, :(using VLBISkyModels); recursive=true)

makedocs(;
    modules=[VLBISkyModels],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/VLBISkyModels.jl/blob/{commit}{path}#{line}",
    sitename="VLBISkyModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/VLBISkyModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/VLBISkyModels.jl",
    devbranch="main",
)
