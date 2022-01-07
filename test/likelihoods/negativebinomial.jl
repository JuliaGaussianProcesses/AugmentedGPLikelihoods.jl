@testset "NegBinomialLikelihood{<:LogisticLink}" begin
    test_interface(NegBinomialLikelihood(10.0), NegativeBinomial)
    test_auglik(NegBinomialLikelihood(LogisticLink(), 10))
    test_auglik(NegBinomialLikelihood(LogisticLink(), 5.5))
end
