@testset "ntdist.jl" begin
    rng = MersenneTwister(42)
    q = Normal()
    d = NTDist(q)
    n = 10
    ds = [NTDist(q) for _ in 1:n]
    @test d isa SpecialDistributions.AbstractNTDist
    @test rand(rng, Float64, d) isa NamedTuple
    @test rand(rng, d) isa NamedTuple
    @test rand(rng, d, n) isa TupleVector
    @test length(rand(rng, d, n)) == 10
    @test dist(d) == q
    @test mean(d) isa NamedTuple
    @test mean(ds) isa TupleVector

    x = ntrand(d)
    @test keys(x) == (:ω,)
    @test logdensity(d, x) isa Real
    @test ntrand(q) isa NamedTuple

    @test tvrand(rng, ds) isa TupleVector
    @test tvmean(ds) isa TupleVector

    @test kldivergence(d, d) == 0
end
