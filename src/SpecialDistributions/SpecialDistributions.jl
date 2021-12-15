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

export PolyaGamma, PolyaGammaMT
export PolyaGammaPoisson
export ntrand, ntmean, tvmean

posℝ = @half Lebesgue(ℝ)

@doc raw"""
    ntrand(rng::AbstractRNG, d::Distribution) -> NamedTuple

Return a sample as a `NamedTuple`
"""
ntrand

@doc raw"""
    ntmean(d::Distribution) -> NamedTuple

Return the mean as a `NamedTuple`
"""
ntmean

@doc raw"""
    vtmean(d::AbstractVector{<:Distribution})
    vtmean(d::ProductMeasure)

Return a collection of mean as a `TupleVector`
"""
tvmean

tvmean(qΩ::ProductMeasure) = tvmean(marginals(qΩ))

include("polyagamma.jl")
include("polyagammapoisson.jl")
end
