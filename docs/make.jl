using Pkg
Pkg.add(Pkg.PackageSpec(; url="https://github.com/JuliaGaussianProcesses/JuliaGPsDocs.jl"))

## Build the docs
using AugmentedGPLikelihoods
using JuliaGPsDocs

JuliaGPsDocs.generate_examples(AugmentedGPLikelihoods)

using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(
    AugmentedGPLikelihoods, :DocTestSetup, :(using AugmentedGPLikelihoods); recursive=true
)

bib = CitationBibliography(joinpath(pkgdir(AugmentedGPLikelihoods), "CITATIONS.bib"))

makedocs(;
    modules=[AugmentedGPLikelihoods],
    plugins=[bib],
    authors="Théo Galy-Fajou <theo.galyfajou@gmail.com> and contributors",
    repo="https://github.com/JuliaGaussianProcesses/AugmentedGPLikelihoods.jl/blob/{commit}{path}#{line}",
    sitename="AugmentedGPLikelihoods.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGaussianProcesses.github.io/AugmentedGPLikelihoods.jl",
        assets=String[],
        size_threshold=nothing,
    ),
    pages=[
        "Home" => "index.md",
        "Likelihoods" => map(readdir(joinpath(@__DIR__, "src", "likelihoods"))) do x
            joinpath("likelihoods", x)
        end,
        "Examples" => JuliaGPsDocs.find_generated_examples(AugmentedGPLikelihoods),
        "Misc" => "misc.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/AugmentedGPLikelihoods.jl",
    devbranch="main",
    push_preview=true,
)
