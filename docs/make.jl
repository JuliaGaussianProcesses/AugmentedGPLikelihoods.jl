using AugmentedGPLikelihoods
using Documenter
using DocumenterCitations

DocMeta.setdocmeta!(AugmentedGPLikelihoods, :DocTestSetup, :(using AugmentedGPLikelihoods); recursive=true)

makedocs(;
    modules=[AugmentedGPLikelihoods],
    authors="Théo Galy-Fajou <theo.galyfajou@gmail.com> and contributors",
    repo="https://github.com/JuliaGaussianProcesses/AugmentedGPLikelihoods.jl/blob/{commit}{path}#{line}",
    sitename="AugmentedGPLikelihoods.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGaussianProcesses.github.io/AugmentedGPLikelihoods.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/AugmentedGPLikelihoods.jl",
    devbranch="main",
    push_preview=true,
)
