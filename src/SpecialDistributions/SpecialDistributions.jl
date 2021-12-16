#=
    Module contains a few extra distributions needed for the 
    different augmentations
=#

module SpecialDistributions
using Distributions
using MeasureBase
using Random
using TupleVectors
using Statistics
using SpecialFunctions
using IrrationalConstants: twoπ, halfπ, inv2π, fourinvπ

export PolyaGamma
export PolyaGammaPoisson

export ntrand, ntmean, ntmeaninv
export tvrand, tvmean, tvmeaninv

@doc raw"""
    ntrand(rng::AbstractRNG, d::Distribution) -> NamedTuple

Return a sample as a `NamedTuple`.
"""
ntrand

@doc raw"""
    tvrand([rng::AbstractRNG], d::ProductMeasure) -> TupleVector

"""
tvrand(rng::AbstractRNG, d::ProductMeasure) = TupleVector(rand(rng, d))
tvrand(d::ProductMeasure) = tvrand(GLOBAL_RNG, d)

@doc raw"""
    ntmean(d::Distribution) -> NamedTuple

Return the mean as a `NamedTuple`.
"""
ntmean

@doc raw"""
    tvmean(d::AbstractVector{<:Distribution}) -> TupleVector
    tvmean(d::ProductMeasure)

Return a collection of mean as a `TupleVector`.
"""
tvmean

tvmean(qΩ::ProductMeasure) = tvmean(marginals(qΩ))
tvmeaninv(qΩ::ProductMeasure) = tvmeaninv(marginals(qΩ))

include("wrappers.jl")
include("polyagamma.jl")
include("polyagammapoisson.jl")
end
