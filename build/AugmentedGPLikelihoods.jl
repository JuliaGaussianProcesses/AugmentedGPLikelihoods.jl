module AugmentedGPLikelihoods

using Reexport

using Distributions
@reexport using GPLikelihoods
using GPLikelihoods: AbstractLikelihood, AbstractLink
using IrrationalConstants
using LogExpFunctions
using MeasureBase
using Random: AbstractRNG, GLOBAL_RNG
using SpecialFunctions
using TupleVectors

export nlatent

export init_aux_variables, init_aux_posterior
export aux_sample, aux_sample!, aux_full_conditional
export aux_posterior, aux_posterior!
export auglik_potential,
    auglik_precision, auglik_potential_and_precision, auglik_potential_and_precision
export expected_auglik_potential,
    expected_auglik_precision, expected_auglik_potential_and_precision

export logtilt, expected_logtilt
export aux_prior
export aug_loglik, aux_kldivergence

export ScaledLogistic

include("api.jl")
include("generic.jl")
include("SpecialDistributions/SpecialDistributions.jl")
using .SpecialDistributions

include("likelihoods/bernoulli.jl")
include("likelihoods/poisson.jl")
include("likelihoods/studentt.jl")

include("utils.jl")
include("TestUtils.jl")

end