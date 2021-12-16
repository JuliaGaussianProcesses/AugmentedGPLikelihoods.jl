function aux_sample!(Ω, lik::AbstractLikelihood, y, f)
    return aux_sample!(GLOBAL_RNG, Ω, lik, y, f)
end

function aux_sample(lik::AbstractLikelihood, y, f)
    return aux_sample(GLOBAL_RNG, lik, y, f)
end

function aux_sample(rng::AbstractRNG, lik::AbstractLikelihood, y, f)
    return aux_sample!(rng, init_aux_variables(lik, length(y)), lik, y, f)
end

function aux_posterior(lik::AbstractLikelihood, y, f)
    aux_posterior!(init_aux_posterior(lik, length(y)), lik, y, f)
end

function aux_full_conditional(lik::AbstractLikelihood, y, f)
    return For(1:length(y)) do i
        aux_full_conditional(lik, y[i], f[i])
    end
end

function init_aux_variables(lik::AbstractLikelihood, n::Int)
    return init_aux_variables(GLOBAL_RNG, lik, n)
end

function aug_loglik(lik::AbstractLikelihood, Ω, y, f)
    return logtilt(lik, Ω, y, f) + logdensity(aux_prior(lik, y), Ω)
end

function aux_kldivergence(lik::AbstractLikelihood, qΩ::ProductMeasure, y)
    return aux_kldivergence(lik, qΩ, aux_prior(lik, y))
end

function aux_kldivergence(::AbstractLikelihood, qΩ::ProductMeasure, pΩ::ProductMeasure)
    return mapreduce(Distributions.kldivergence, +, marginals(qΩ), marginals(pΩ))
end

function auglik_potential_and_precision(lik::AbstractLikelihood, Ω, y)
    return (auglik_potential(lik, Ω, y), auglik_precision(lik, Ω, y))
end

function expected_auglik_potential_and_precision(lik::AbstractLikelihood, qΩ, y)
    return (expected_auglik_potential(lik, qΩ, y), expected_auglik_precision(lik, qΩ, y))
end

nlatent(::AbstractLikelihood) = 1 # Default number of latent for each likelihood
# This should potentially move to GPLikelihoods.jl
