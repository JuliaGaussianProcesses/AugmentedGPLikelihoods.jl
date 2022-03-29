@testset "StudentT" begin
    test_interface(StudentTLikelihood(3.0, 1.5), AffineDistribution{Float64,Continuous,<:TDist})
    test_auglik(StudentTLikelihood(3.0, 1.5))
end
