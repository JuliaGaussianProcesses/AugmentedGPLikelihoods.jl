using AugmentedGPLikelihoods
using AugmentedGPLikelihoods.SpecialDistributions
using AugmentedGPLikelihoods.TestUtils: test_auglik
const AGPL = AugmentedGPLikelihoods
using Distributions
using GPLikelihoods
using Random
using LogExpFunctions
using Test

@testset "AugmentedGPLikelihoods.jl" begin
    @info "Testing likelihoods"
    @testset "Likelihoods" begin
        include("likelihoods/bernoulli.jl")
        include("likelihoods/poisson.jl")
    end

    @info "Testing SpecialDistributions"
    @testset "SpecialDistributions" begin
        include("SpecialDistributions/polyagamma.jl")
        include("SpecialDistributions/polyagammapoisson.jl")
    end

    include("utils.jl")
end
