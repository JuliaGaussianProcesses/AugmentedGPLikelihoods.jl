@testset "StudentT" begin
    test_interface(StudentTLikelihood(3.0, 1.5), Distributions.AffineDistribution{Float64,Continuous,<:TDist})
    test_auglik(StudentTLikelihood(3.0, 1.5))
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
            lik = StudentTLikelihood(θ[3], θ[4])
            fz = gp(x, 1e-8);
            u_post = u_posterior(fz, fill(θ[5], N), Matrix{Float64}(I(N)))
            return aug_elbo(lik, u_post, x, y)
        end
        θ0 = [1., 2., 3., 4., 5.]
        @test loss(θ0) ≈ -51.95433668026266
        @test ForwardDiff.gradient(loss, θ0) ≈ ReverseDiff.gradient(loss, θ0)
    end
end
