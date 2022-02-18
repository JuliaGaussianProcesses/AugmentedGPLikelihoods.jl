@testset "Laplace" begin
    test_interface(LaplaceLikelihood(3.0), Laplace)
    @test LaplaceLikelihood().β == 1
    test_auglik(LaplaceLikelihood(1.0); rng=MersenneTwister(42))
    # Test the custom kl divergence
    λ = rand()
    μ = rand()
    @test kldivergence(InverseGaussian(μ, 2λ), InverseGamma(1//2, λ)) ≈
        (log(2λ) / 2 - log(2π) / 2 - log(λ) / 2 + loggamma(1//2) + λ / μ)
end
