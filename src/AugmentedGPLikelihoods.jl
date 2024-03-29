module AugmentedGPLikelihoods

using Reexport

using ArraysOfArrays
using ChainRulesCore: @ignore_derivatives
using Distributions
@reexport using GPLikelihoods
using GPLikelihoods: AbstractLikelihood, AbstractLink, BijectiveSimplexLink
using IrrationalConstants
using LogExpFunctions
using MeasureBase: MeasureBase, logdensity_def, marginals
using MeasureTheory: For
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
export aux_prior, aux_kldivergence
export aug_loglik, expected_aug_loglik

export ScaledLogistic, InvScaledLogistic
export logisticsoftmax
export LogisticSoftMaxLink
export BijectiveSimplexLink

export LaplaceLikelihood, StudentTLikelihood

include("api.jl")
include("generic.jl")
include("SpecialDistributions/SpecialDistributions.jl")
using .SpecialDistributions

include("likelihoods/bernoulli.jl")
include("likelihoods/heteroscedasticgaussian.jl")
include("likelihoods/laplace.jl")
include("likelihoods/negativebinomial.jl")
include("likelihoods/categorical.jl")
include("likelihoods/poisson.jl")
include("likelihoods/studentt.jl")

include("utils.jl")
include("TestUtils.jl")

end
