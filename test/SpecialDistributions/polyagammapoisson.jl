@testset "polyagammapoisson" begin
    # TODO add more basic tests
    d = PolyaGammaPoisson(2, 3.0, 3.0)
    @test length(d) == 2
end
