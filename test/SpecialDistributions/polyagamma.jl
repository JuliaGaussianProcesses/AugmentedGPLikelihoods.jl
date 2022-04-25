@testset "polyagamma" begin
    @test mean(PolyaGamma(1, 0)) == 1 / 4
    @test mean(PolyaGamma(1, 2.0)) == tanh(1.0) / 4

    for (b, c) in ((1, 0), (1, 2.0), (3, 0), (3, 2.5), (3, 3.2), (1.2, 3.2))
        p = PolyaGamma(b, c)
        @test logpdf(p, rand(p)) isa Real
        @test mean(rand(p, 10000)) ≈ mean(p) atol = 1e-2
    end

    p = PolyaGamma(1, 0)
    @test Distributions.insupport(p, 0)
    @test !Distributions.insupport(p, -1)
    @test minimum(p) === 0
    @test maximum(p) == Inf
    @test Distributions.params(p) == (1, 0)
end
