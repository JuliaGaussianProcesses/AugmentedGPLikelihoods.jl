#=
    Module contains a few extra distributions needed for the 
    different augmentations
=#

module SpecialDistributions
using Distributions
using LogExpFunctions
using MeasureBase
using Random
using TupleVectors
using Statistics
using SpecialFunctions
using IrrationalConstants: logtwo, twoπ, halfπ, inv2π, fourinvπ, invπ, log2π

export PolyaGamma
export NegativeMultinomial
export PolyaGammaPoisson
export PolyaGammaNegativeMultinomial

export NTDist, dist
export ntrand, ntmean
export tvrand, tvmean

include("ntdist.jl")

@doc raw"""
    ntrand([rng::AbstractRNG,] d) -> NamedTuple

Return a sample as a `NamedTuple`.
"""
ntrand

ntrand(d) = ntrand(Random.GLOBAL_RNG, d)

@doc raw"""
    tvrand([rng::AbstractRNG,] d::ProductMeasure) -> TupleVector
    tvrand([rng::AbstractRNG,] d::AbstractVector{<:AbstractNTDist}) -> TupleVector

Return a collection of samples as a TupleVector
"""
tvrand

tvrand(d) = tvrand(Random.GLOBAL_RNG, d)

tvrand(rng::AbstractRNG, d::ProductMeasure) = TupleVector(rand(rng, d))

@doc raw"""
    ntmean(d::Distribution) -> NamedTuple

Return the mean as a `NamedTuple`.
"""
ntmean

@doc raw"""
    tvmean(d::AbstractVector{<:AbstractNTDist}) -> TupleVector
    tvmean(d::ProductMeasure)

Return a collection of mean as a `TupleVector`.
"""
tvmean

tvmean(qΩ::ProductMeasure) = tvmean(marginals(qΩ))
tvmeaninv(qΩ::ProductMeasure) = tvmeaninv(marginals(qΩ))

include("negativemultinomial.jl")
include("polyagamma.jl")
include("polyagammanegativemultinomial.jl")
include("polyagammapoisson.jl")
end
