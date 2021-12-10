@testset "polyagamma" begin
    p1 = PolyaGamma(1, 0)
    @test mean(p1) == 1 / 4
    s = rand(p1, 10000)
    @test mean(s) ≈ mean(p1) atol = 1e-2

    p2 = PolyaGamma(1, 2.0)
    @test mean(p2) == tanh(1.0) / 4
    s = rand(p2, 10000)
    @test mean(s) ≈ mean(p2) atol = 1e-2

    p3 = PolyaGamma(1.2, 3.2)
    s = rand(p3, 10000)
    @test_broken mean(s) ≈ mean(p3) atol = 1e-2

    @test insupport(p1, 0)
    @test !insupport(p1, -1)
    @test minimum(p1) === 0
    @test minimum(p2) === 0.0
    @test maximum(p1) == Inf
    @test params(p1) == (1, 0)
end
