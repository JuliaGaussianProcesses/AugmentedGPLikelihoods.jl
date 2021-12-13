@testset "Poisson{<:ScaledLogisticLink}" begin
    test_auglik(PoissonLikelihood(ScaledLogistic(10.0)))
end
