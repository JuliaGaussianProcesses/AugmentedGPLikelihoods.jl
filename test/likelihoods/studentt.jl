@testset "StudentT" begin
    # TODO test StudentTLikelihood GPLikelihoods interface when 
    # https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl/pull/57 is merged
    test_interface(StudentTLikelihood(3.0, 1.5), LocationScale{<:TDist})
    test_auglik(StudentTLikelihood(3.0, 1.5))
end
