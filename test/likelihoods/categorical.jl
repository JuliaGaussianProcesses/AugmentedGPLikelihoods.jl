@testset "Categorical" begin
    Nclass = 3
    n = 10
    bij_lik = CategoricalLikelihood(
        BijectiveSimplexLink(LogisticSoftMaxLink(zeros(Nclass)))
    )
    @test_skip test_auglik(
        bij_lik;
        n,
        f=[randn(n) for _ in 1:AGPL.nlatent(bij_lik)],
        qf=[Normal.(randn(n), 1.0) for _ in 1:AGPL.nlatent(bij_lik)],
    )
    nonbij_lik = CategoricalLikelihood(LogisticSoftMaxLink(zeros(Nclass)))
    @test_skip test_auglik(
        nonbij_lik,
        n,
        f=[randn(n) for _ in 1:AGPL.nlatent(nonbij_lik)],
        qf=[Normal.(randn(n), 1.0) for _ in 1:AGPL.nlatent(nonbij_lik)],
    )
end
