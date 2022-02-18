module TestUtils
using AugmentedGPLikelihoods
using AugmentedGPLikelihoods.SpecialDistributions
using Distributions
using GPLikelihoods: AbstractLikelihood
using LinearAlgebra
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

        # Test that the full conditional is correct
        @testset "Full conditional Ω" begin
            pcondΩ = aux_full_conditional(lik, y, f) # Compute the full conditional of Ω
            Ω₁ = tvrand(rng, pcondΩ) # Sample a set of aux. variables
            Ω₂ = tvrand(rng, pcondΩ) # Sample another set of aux. variables
            # We compute p(f, y) by doing C = p(f,y) = p(y|Ω,f)p(Ω)/p(Ω|y,f)
            # This should be the same no matter what Ω is
            logC₁ = logtilt(lik, Ω₁, y, f) + logdensity(pΩ, Ω₁) - logdensity(pcondΩ, Ω₁)
            logC₂ = logtilt(lik, Ω₂, y, f) + logdensity(pΩ, Ω₂) - logdensity(pcondΩ, Ω₂)
            @test logC₁ ≈ logC₂ atol = 1e-5
        end

        @testset "Full conditional f" begin
            pcondΩ = aux_full_conditional(lik, y, f) # Compute the full conditional of Ω
            Ω = tvrand(rng, pcondΩ) # Sample a set of aux. variables
            K = (x -> x * x')(rand(n, n)) # Prior Covariance matrix
            S = inv(Symmetric(inv(K) + Diagonal(only(auglik_precision(lik, Ω, y)))))
            m = S * (only(auglik_potential(lik, Ω, y)))
            qF = MvNormal(m, S)
            pF = MvNormal(K)
            f₁ = rand(rng, qF)
            f₂ = rand(rng, qF)
            logC₁ = logtilt(lik, Ω, y, f₁) + logpdf(pF, f₁) - logpdf(qF, f₁)
            logC₂ = logtilt(lik, Ω, y, f₂) + logpdf(pF, f₂) - logpdf(qF, f₂)
            @test logC₁ ≈ logC₂ atol = 1e-5
        end
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
        φ = TupleVectors.unwrap(aux_posterior(lik, y, qf).pars) # TupleVector
        φ_opt = vcat(values(φ)...)
        s = keys(φ)
        n_var = length(s)
        function loss(φ)
            q = ProductMeasure(
                qΩ.f,
                TupleVector(
                    NamedTuple{s}(
                        collect(φ[((j - 1) * n_var + 1):(j * n_var)] for j in 1:n_var)
                    ),
                ),
            )
            return -expected_logtilt(lik, q, y, qf) + aux_kldivergence(lik, q, y)
        end
        ϵ = 1e-2
        # Test that by perturbing the value in random directions, the loss does not decrease
        for i in n_var * n
            (lik isa PoissonLikelihood && i <= n) && continue # We do not want to vary y
            Δ = zeros(n_var * n)
            Δ[i] = ϵ # We try one element at a time
            @test loss(φ_opt) <= loss(φ_opt + Δ)
            @test loss(φ_opt) <= loss(φ_opt - Δ)
        end
        # Optim.optimize(loss, φ_opt)
        # values of the ELBO
        pΩ = aux_prior(lik, y)
        @test pΩ isa ProductMeasure
        @test kldivergence(first(marginals(qΩ)), first(marginals(pΩ))) isa Real
        @test expected_logtilt(lik, qΩ, y, qf) isa Real
        @test aux_kldivergence(lik, qΩ, pΩ) isa Real
    end
end
end
