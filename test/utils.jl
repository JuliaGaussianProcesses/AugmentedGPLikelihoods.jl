@testset "utils.jl" begin
    rng = MersenneTwister(42)
    m = randn(rng)
    σ = rand(rng)
    d = Normal(m, σ)
    @test second_moment(d) ≈ abs2(m) + abs2(σ)
    c = second_moment(d)
    @test approx_expected_logistic(m, c) ≈ exp(m / 2) / cosh(m / 2) / 2
    c = second_moment(Normal(10e2, σ))
    @test approx_expected_logistic(m, c) ≈ logistic(m)
end