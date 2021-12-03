module AugmentedGPLikelihoods


using Reexport

using Distributions
@reexport using GPLikelihoods
using Random: AbstractRNG, GLOBAL_RNG

export init_aux_variables, init_aux_posterior
export aux_sample, aux_sample!
export aux_posterior, aux_posterior!
export vi_shift, vi_rate
export sample_shift, sample_rate

include("api.jl")
include("SpecialDistributions/SpecialDistributions.jl")
using .SpecialDistributions

include("likelihoods/bernoulli.jl")


end
