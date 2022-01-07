@testset "NegBinomialLikelihood{<:LogisticLink}" begin
    test_interface(NegBinomialLikelihood(), NegativeBinomial)
    test_auglik(NegBinomialLikelihood(LogisticLink(), 10))
end
