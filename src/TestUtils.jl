module TestUtils
using AugmentedGPLikelihoods
using Distributions
using GPLikelihoods: AbstractLikelihood
using MeasureBase
using Random
using Test
using TupleVectors
# Test API for augmented likelihood
function test_auglik(
    lik::AbstractLikelihood;
    n=10,
    f=randn(n),
    qf=Normal.(randn(n), 1.0),
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    y = rand.(rng, lik.(f))
    nf = nlatent(lik)
    # Testing sampling
    @testset "Sampling" begin
        Ω = init_aux_variables(lik, n)
        @test Ω isa TupleVector
        @test first(Ω) isa NamedTuple
        @test length(Ω) == n
        Ω = aux_sample!(rng, Ω, lik, y, f)
        @test Ω isa TupleVector
        new_Ω = aux_sample(rng, lik, y, f)
        @test new_Ω isa TupleVector
        @test length(Ω) == n

        βs = auglik_potential(lik, Ω, y)
        γs = auglik_precision(lik, Ω, y)
        β2, γ2 = auglik_potential_and_precision(lik, Ω, y)
        @test length(γs) == length(βs) == nf # Check that there are n latent vectors
        @test first(βs) isa AbstractVector
        @test first(γs) isa AbstractVector
        @test all(map(≈, βs, β2))
        @test all(map(≈, γs, γ2))
        @test all(x -> all(>=(0), x), γs) # Check that the variance is positive

        @test logtilt(lik, Ω, y, f) isa Real
        @test aug_loglik(lik, Ω, y, f) isa Real

        pΩ = aux_prior(lik, y)
        @test logdensity(pΩ, Ω) isa Real

        Ω₁ = init_aux_variables(rng, lik, n)
        Ω₂ = init_aux_variables(rng, lik, n)
        logC₁ =
            logdensity(aux_full_conditional(lik, y, f), Ω₁) - logtilt(lik, Ω₁, y, f) -
            logdensity(pΩ, Ω₁)
        logC₂ =
            logdensity(aux_full_conditional(lik, y, f), Ω₂) - logtilt(lik, Ω₂, y, f) -
            logdensity(pΩ, Ω₂)
        @test logC₁ / n ≈ logC₂ / n atol=1.0
    end

    #Testing variational inference
    @testset "Variational Inference" begin
        qΩ = init_aux_posterior(lik, n)
        @test qΩ isa ProductMeasure
        @test_broken length(qΩ) == n
        qΩ = aux_posterior!(qΩ, lik, y, qf)
        @test qΩ isa ProductMeasure
        new_qΩ = aux_posterior(lik, y, qf)
        @test new_qΩ isa ProductMeasure
        @test_broken length(new_qΩ) == n

        βs = expected_auglik_potential(lik, qΩ, y)
        γs = expected_auglik_precision(lik, qΩ, y)
        β2, γ2 = expected_auglik_potential_and_precision(lik, qΩ, y)
        @test length(γs) == length(βs) == nf # Check that there are n latent vectors
        @test first(βs) isa AbstractVector
        @test first(γs) isa AbstractVector
        @test all(map(≈, βs, β2))
        @test all(map(≈, γs, γ2))

        @test all(x -> all(>=(0), x), γs) # Check that the variance is positive

        # TODO test that aux_posterior parameters return the minimizing
        # values of the ELBO
        pΩ = aux_prior(lik, y)
        @test pΩ isa ProductMeasure
        @test kldivergence(first(marginals(qΩ)), first(marginals(pΩ))) isa Real
        @test expected_logtilt(lik, qΩ, y, qf) isa Real
        @test aux_kldivergence(lik, qΩ, pΩ) isa Real
    end
end
end
