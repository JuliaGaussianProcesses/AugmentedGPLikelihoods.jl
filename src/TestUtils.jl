module TestUtils
using AugmentedGPLikelihoods
using Distributions
using GPLikelihoods: AbstractLikelihood
using Random
using Test
# Test API for augmented likelihood
function test_auglik(
    lik::AbstractLikelihood;
    n = 10,
    f = randn(n),
    qf = Normal.(randn(n), 1.0),
    rng::AbstractRNG = Random.GLOBAL_RNG,
)
    y = rand.(rng, lik.(f))
    nf = nlatent(lik)
    # Testing sampling
    @testset "Testing sampling" begin
        Ω = init_aux_variables(lik, n)
        @test Ω isa NamedTuple
        @test length(first(Ω)) == n
        Ω = aux_sample!(rng, Ω, lik, y, f)
        @test Ω isa NamedTuple
        new_Ω = aux_sample(rng, lik, y, f)
        @test new_Ω isa NamedTuple
        @test length(first(Ω)) == n

        βs = auglik_potential(lik, Ω, y)
        γs = auglik_precision(lik, Ω, y)
        @test all(x -> all(>(0), x), γs) # Check that the variance is positive
        @test length(γs) == length(βs) == nf # Check that there are n latent vectors

        pΩ = aux_prior(lik, y)
        @test keys(pΩ) == keys(Ω)
        @test logtilt(lik, Ω, y, f) isa Real
        @test aug_loglik(lik, Ω, y, f) isa Real
    end

    #Testing variational inference
    @testset "Testing Variational Inference" begin
        qΩ = init_aux_posterior(lik, n)
        @test qΩ isa NamedTuple
        @test length(first(qΩ)) == n
        qΩ = aux_posterior!(qΩ, lik, y, qf)
        @test qΩ isa NamedTuple
        new_qΩ = aux_posterior(lik, y, qf)
        @test new_qΩ isa NamedTuple
        @test length(first(new_qΩ)) == n
        @test keys(qΩ) == keys(new_qΩ)

        βs = expected_auglik_potential(lik, qΩ, y)
        γs = expected_auglik_precision(lik, qΩ, y)
        @test all(x -> all(>(0), x), γs) # Check that the variance is positive
        @test length(γs) == length(βs) == nf # Check that there are n latent vectors
        # TODO test that aux_posterior parameters return the minimizing
        # parameters for expected_aug_loglik

        pΩ = aux_prior(lik, y)
        @test keys(pΩ) == keys(qΩ)
        for s in keys(pΩ)
            @test kldivergence(first(getfield(qΩ, s)), first(getfield(pΩ, s))) isa Real
        end
        @test expected_logtilt(lik, qΩ, y, qf) isa Real
        @test aux_kldivergence(lik, qΩ, pΩ) isa Real
    end
end
end
