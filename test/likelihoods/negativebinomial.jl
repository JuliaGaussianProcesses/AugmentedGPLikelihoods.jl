@testset "NegativeBinomialLikelihood{<:NBParamFailure,<:LogisticLink}" begin
    test_auglik(NegativeBinomialLikelihood(NBParamFailure(10), LogisticLink()))
    test_auglik(NegativeBinomialLikelihood(NBParamFailure(5.5), LogisticLink()))
end
