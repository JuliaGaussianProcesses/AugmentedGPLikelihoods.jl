@testset "Categorical" begin
    Nclass = 3
    test_interface(CategoricalLikelihood(BijectiveSimplex(LogisticSoftMaxLink(zeros(Nclass)))), Categorical)
    test_interface(CategoricalLikelihood(LogisticSoftMaxLink(zeros(Nclass))), Categorical)
end
