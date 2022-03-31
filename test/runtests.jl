using ApproximateGPs
using AugmentedGPLikelihoods
using AugmentedGPLikelihoods.SpecialDistributions
using AugmentedGPLikelihoods.TestUtils: test_auglik
const AGPL = AugmentedGPLikelihoods
using Distributions
using ForwardDiff
using GPLikelihoods
using GPLikelihoods.TestInterface: test_interface
using LinearAlgebra: I
using MeasureBase
using LogExpFunctions
using Random
using ReverseDiff
using SpecialFunctions
using Test
using TupleVectors

function aug_elbo(lik, u_post, x, y)
    qf = ApproximateGPs.marginals(u_post(x))
    qΩ = aux_posterior(lik, y, qf)
    return expected_logtilt(lik, qΩ, y, qf) - aux_kldivergence(lik, qΩ, y) -
           kldivergence(u_post.approx.q, u_post.approx.fz)     # approx.fz is the prior and approx.q is the posterior 
end

function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))
end

@testset "AugmentedGPLikelihoods.jl" begin
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
