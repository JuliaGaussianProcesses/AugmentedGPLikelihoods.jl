using AugmentedGPLikelihoods
using AugmentedGPLikelihoods.SpecialDistributions
using AugmentedGPLikelihoods.TestUtils: test_auglik
const AGPL = AugmentedGPLikelihoods

using Aqua
using Distributions
using GPLikelihoods
using GPLikelihoods.TestInterface: test_interface
using JET
using MeasureBase
using LogExpFunctions
using Random
using SpecialFunctions
using Test
using TupleVectors

@testset "AugmentedGPLikelihoods.jl" begin
    @info "Running Aqua tests"
    @testset "Aqua" begin
        Aqua.test_all(AugmentedGPLikelihoods)
    end
    @info "Testing likelihoods"
    @testset "Likelihoods" begin
        include("likelihoods/bernoulli.jl")
        include("likelihoods/laplace.jl")
        include("likelihoods/negativebinomial.jl")
        include("likelihoods/poisson.jl")
        include("likelihoods/studentt.jl")
    end

    @info "Testing SpecialDistributions"
    @testset "SpecialDistributions" begin
        include("SpecialDistributions/ntdist.jl")
        include("SpecialDistributions/polyagamma.jl")
        include("SpecialDistributions/polyagammapoisson.jl")
    end

    include("utils.jl")
end
