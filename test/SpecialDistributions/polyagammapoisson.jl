@testset "polyagammapoisson.jl" begin
    # TODO add more basic test
    d = PolyaGammaPoisson(2, 3.0, 3.0)
    @test length(d) == 2
end