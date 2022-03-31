@testset "Laplace" begin
    test_interface(LaplaceLikelihood(3.0), Laplace)
    @test LaplaceLikelihood().β == 1
    test_auglik(LaplaceLikelihood(1.0); rng=MersenneTwister(42))
    # Test the custom kl divergence
    λ = rand()
    μ = rand()
    @test kldivergence(InverseGaussian(μ, 2λ), InverseGamma(1//2, λ)) ≈
        (log(2λ) / 2 - log(2π) / 2 - log(λ) / 2 + loggamma(1//2) + λ / μ)
    @testset "Augmented ELBO /w AD" begin
        N = 10
        x = 1:N
        y = 1:N
        function loss(θ)
            k = ScaledKernel(
                RBFKernel() ∘ ScaleTransform(inv(θ[1])), 
                θ[2]
            )
            gp = GP(k)
            lik = LaplaceLikelihood(θ[3])
            fz = gp(x, 1e-8);
            u_post = u_posterior(fz, fill(θ[4], N), Matrix{Float64}(I(N)))
            return aug_elbo(lik, u_post, x, y)
        end
        θ0 = [1., 2., 3., 4.]
        @test loss(θ0) ≈ -51.95433668026266
        @test ForwardDiff.gradient(loss, θ0) ≈ ReverseDiff.gradient(loss, θ0)
    end
end
