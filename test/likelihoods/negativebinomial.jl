@testset "NegativeBinomialLikelihood{<:LogisticLink}" begin
    test_interface(NegativeBinomialLikelihood(10.0), NegativeBinomial)
    test_auglik(NegativeBinomialLikelihood(LogisticLink(), 10))
    test_auglik(NegativeBinomialLikelihood(LogisticLink(), 5.5))
end
