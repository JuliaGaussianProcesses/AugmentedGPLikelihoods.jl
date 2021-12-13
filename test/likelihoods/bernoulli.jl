@testset "Bernoulli{<:LogisticLink}" begin
    test_auglik(BernoulliLikelihood(LogisticLink()))
end
