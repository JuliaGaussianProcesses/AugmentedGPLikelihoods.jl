module AugmentedGPLikelihoods

using Reexport

using Distributions
@reexport using GPLikelihoods
using GPLikelihoods: AbstractLikelihood
using Random: AbstractRNG, GLOBAL_RNG

export nlatent

export init_aux_variables, init_aux_posterior
export aux_sample, aux_sample!
export aux_posterior, aux_posterior!
export auglik_potential, auglik_precision
export expected_auglik_potential, expected_auglik_precision

export logtilt, expected_logtilt
export aux_prior
export aug_loglik, expected_aug_loglik

include("api.jl")
include("generic.jl")
include("SpecialDistributions/SpecialDistributions.jl")
using .SpecialDistributions

include("likelihoods/bernoulli.jl")

include("TestUtils.jl")

end
