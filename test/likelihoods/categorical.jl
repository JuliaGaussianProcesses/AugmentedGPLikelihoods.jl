@testset "Categorical" begin
    Nclass = 3
    n = 10
    bij_lik = CategoricalLikelihood(
        BijectiveSimplexLink(LogisticSoftMaxLink(zeros(Nclass)))
    )
    test_auglik(
        bij_lik;
        n,
        f=[randn(n) for _ in 1:AGPL.nlatent(bij_lik)],
        qf=[Normal.(randn(n), 1.0) for _ in 1:AGPL.nlatent(bij_lik)],
    )
    # test_auglik(CategoricalLikelihood(LogisticSoftMaxLink(zeros(Nclass))), Categorical)
end
