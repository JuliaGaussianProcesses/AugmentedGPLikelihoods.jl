#=
    Module contains a few extra distributions needed for the 
    different augmentations
=#

module SpecialDistributions
using Distributions
using Random
using Statistics
using SpecialFunctions
using IrrationalConstants: twoπ, halfπ, inv2π, fourinvπ

export PolyaGamma
export PolyaGammaPoisson

include("polyagamma.jl")
include("polyagammapoisson.jl")
end
