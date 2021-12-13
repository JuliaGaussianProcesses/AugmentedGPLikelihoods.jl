@testset "polyagammapoisson.jl" begin
    # TODO add more basic test
    d = PolyaGamaPoisson(2, 3.0, 3.0)
    @test length(d) == 2
end