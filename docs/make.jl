## First process examples
const EXAMPLES_OUT = joinpath(@__DIR__, "src", "examples")
ispath(EXAMPLES_OUT) && rm(EXAMPLES_OUT; recursive=true)
mkpath(EXAMPLES_OUT)

# TODO use the general approach of ApproximateGPs.jl
examples = filter!(isdir, readdir(joinpath(@__DIR__, "..", "examples"); join=true))
let script = "using Pkg; Pkg.activate(ARGS[1]); Pkg.instantiate()"
    for example in examples
        if !success(`$(Base.julia_cmd()) -e $script $example`)
            error(
                "project environment of example ",
                basename(example),
                " could not be instantiated",
            )
        end
    end
end
# Run examples asynchronously
processes = let literatejl = joinpath(@__DIR__, "literate.jl")
    map(examples) do example
        return run(
            pipeline(
                `$(Base.julia_cmd()) $literatejl $(basename(example)) $EXAMPLES_OUT`;
                stdin=devnull,
                stdout=devnull,
                stderr=stderr,
            );
            wait=false,
        )::Base.Process
    end
end
# Check that all examples were run successfully
isempty(processes) || success(processes) || error("some examples were not run successfully")

## Build the docs
using AugmentedGPLikelihoods
using Documenter
using DocumenterCitations
using Literate
using Pkg

DocMeta.setdocmeta!(
    AugmentedGPLikelihoods, :DocTestSetup, :(using AugmentedGPLikelihoods); recursive=true
)

bib = CitationBibliography(joinpath(@__DIR__, "references.bib"))

likelihoods = filter!(filename -> endswith(filename, ".md"), readdir(EXAMPLES_OUT))

@info "Available likelihoods: $(likelihoods)"

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
        "Likelihoods" => joinpath.(Ref("likelihoods"), likelihoods),
        "Examples" => joinpath.(Ref("examples"), likelihoods),
        "Misc" => "misc.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGaussianProcesses/AugmentedGPLikelihoods.jl",
    devbranch="main",
    push_preview=true,
)
