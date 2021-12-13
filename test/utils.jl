@testset "utils.jl" begin
    rng = MersenneTwister(42)
    m = randn(rng)
    σ = rand(rng)
    d = Normal(m, σ)
    @test AGPL.second_moment(d) ≈ abs2(m) + abs2(σ)
    c = AGPL.second_moment(d)
    @test AGPL.approx_expected_logistic(m, c) ≈ exp(m / 2) / cosh(c / 2) / 2 atol=1e-5
    for T in [Float64, Float32]
        bigm = T(1000)
        c = bigm + abs(randn(rng, T))
        @test AGPL.approx_expected_logistic(bigm, c) ≈ logistic(bigm)
    end
end