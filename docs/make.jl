using AugmentedGPLikelihoods
using Documenter
using DocumenterCitations
using Literate
using Pkg

DocMeta.setdocmeta!(
    AugmentedGPLikelihoods, :DocTestSetup, :(using AugmentedGPLikelihoods); recursive=true
)

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

# TODO use the general approach of ApproximateGPs.jl
for example in ["bernoulli", "poisson"]
    folder = joinpath(pkgdir(AugmentedGPLikelihoods), "examples", example)
    Pkg.activate(folder)
    Pkg.instantiate()
    Literate.markdown(
        joinpath(folder, example * ".jl"),
        joinpath(@__DIR__, "src/examples");
        execute=true,
        # flavor=Literate.DocumenterFlavor(),
    )
end
Pkg.activate(@__DIR__)

makedocs(
    bib;
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
        "Likelihoods" => [
            "Bernoulli" => "likelihoods/bernoulli.md",
            "Poisson" => "likelihoods/poisson.md",
        ],
        "Examples" =>
            ["Bernoulli" => "examples/bernoulli.md", "Poisson" => "examples/poisson.md"],
        "Additional Distributions" => "specialdistributions.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/AugmentedGPLikelihoods.jl",
    devbranch="main",
    push_preview=true,
)
