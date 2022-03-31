@testset "Poisson{<:ScaledLogisticLink}" begin
    test_auglik(PoissonLikelihood(ScaledLogistic(10.0)))
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
            lik = PoissonLikelihood(ScaledLogistic(θ[3]))
            fz = gp(x, 1e-8);
            u_post = u_posterior(fz, fill(θ[4], N), Matrix{Float64}(I(N)))
            return aug_elbo(lik, u_post, x, y)
        end
        θ0 = [1., 2., 3., 4.]
        @test loss(θ0) ≈ -61.4626187223969
        @test ForwardDiff.gradient(loss, θ0) ≈ ReverseDiff.gradient(loss, θ0)
    end
end
