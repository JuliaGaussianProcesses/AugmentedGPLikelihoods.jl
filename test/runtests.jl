using AugmentedGPLikelihoods
using AugmentedGPLikelihoods.SpecialDistributions
using AugmentedGPLikelihoods.TestUtils: test_auglik
const AGPL = AugmentedGPLikelihoods
using Distributions
using GPLikelihoods
using GPLikelihoods.TestInterface: test_interface
using LogExpFunctions
using MeasureBase
using Random
using SpecialFunctions
using Test
using TupleVectors

@testset "AugmentedGPLikelihoods.jl" begin
    @info "Testing likelihoods"
    @testset "Likelihoods" begin
        include("likelihoods/bernoulli.jl")
        include("likelihoods/categorical.jl")
        include("likelihoods/laplace.jl")
        include("likelihoods/negativebinomial.jl")
        include("likelihoods/poisson.jl")
        include("likelihoods/studentt.jl")
    end

    @info "Testing SpecialDistributions"
    @testset "SpecialDistributions" begin
        include("SpecialDistributions/negativemultinomial.jl")
        include("SpecialDistributions/ntdist.jl")
        include("SpecialDistributions/polyagamma.jl")
        include("SpecialDistributions/polyagammanegativemultinomial.jl")
        include("SpecialDistributions/polyagammapoisson.jl")
    end

    include("utils.jl")
end
