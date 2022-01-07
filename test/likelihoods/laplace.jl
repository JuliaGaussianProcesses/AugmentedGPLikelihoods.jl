@testset "Laplace" begin
    test_interface(LaplaceLikelihood(3.0), Laplace)
    test_auglik(LaplaceLikelihood(3.0))
end
