#=
    Module contains a few extra distributions needed for the 
    different augmentations
=#

module SpecialDistributions
using ArraysOfArrays
using Distributions
using LogExpFunctions
using MeasureBase
using MeasureTheory: For
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
    tvrand([rng::AbstractRNG,] d::For) -> TupleVector
    tvrand([rng::AbstractRNG,] d::AbstractVector{<:AbstractNTDist}) -> TupleVector

Return a collection of samples as a TupleVector
"""
tvrand

tvrand(d) = tvrand(Random.GLOBAL_RNG, d)

tvrand(rng::AbstractRNG, d::For) = TupleVector(rand(rng, d))

@doc raw"""
    ntmean(d::Distribution) -> NamedTuple

Return the mean as a `NamedTuple`.
"""
ntmean

@doc raw"""
    tvmean(d::AbstractVector{<:AbstractNTDist}) -> TupleVector
    tvmean(d::For)

Return a collection of mean as a `TupleVector`.
"""
tvmean

tvmean(qΩ::For) = tvmean(marginals(qΩ))
tvmeaninv(qΩ::For) = tvmeaninv(marginals(qΩ))

include("negativemultinomial.jl")
include("polyagamma.jl")
include("polyagammanegativemultinomial.jl")
include("polyagammapoisson.jl")
end
