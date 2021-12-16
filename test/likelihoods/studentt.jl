@testset "StudentT" begin
    test_auglik(StudentTLikelihood(3.0, 2.0))
end
