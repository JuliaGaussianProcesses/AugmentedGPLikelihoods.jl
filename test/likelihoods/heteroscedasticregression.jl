@testset "Heteroscedastic Regression" begin
    TestUtils.can_split(::AGPL.AugHeteroGaussian) = false
    n = 10
    lik = HeteroscedasticGaussianLikelihood(InvScaledLogistic(5.0))
    @test lik isa AGPL.AugHeteroGaussian
    @test test_auglik(
        lik;
        n,
        f=[randn(n) for _ in 1:AGPL.nlatent(lik)],
        qf=[Normal.(randn(n), 1.0) for _ in 1:AGPL.nlatent(lik)],
    )
end
