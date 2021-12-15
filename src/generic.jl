function aux_sample!(Ω, lik::AbstractLikelihood, y, f)
    return aux_sample!(GLOBAL_RNG, Ω, lik, y, f)
end

function aux_sample(lik::AbstractLikelihood, y, f)
    return aux_sample(GLOBAL_RNG, lik, y, f)
end

function aux_sample(rng::AbstractRNG, lik::AbstractLikelihood, y, f)
    return aux_sample!(rng, init_aux_variables(lik, length(y)), lik, y, f)
end

function aux_full_conditional(lik::AbstractLikelihood, y, f)
    return Product(aux_full_conditional.(Ref(lik), y, f))
end

function init_aux_variables(lik::AbstractLikelihood, n::Int)
    return init_aux_variables(GLOBAL_RNG, lik, n)
end

function aux_posterior(lik::AbstractLikelihood, y, qf)
    return aux_posterior!(init_aux_posterior(lik, length(y)), lik, y, qf)
end

function aug_loglik(lik::AbstractLikelihood, Ω, y, f)
    return logtilt(lik, Ω, y, f) + mapreduce(+, Ω, aux_prior(lik, y)) do ω, pω
        logpdf(pω, ω)
    end
end

function aux_kldivergence(lik::AbstractLikelihood, qΩ::NamedTuple, y)
    return aux_kldivergence(lik, qΩ, aux_prior(lik, y))
end

function aux_kldivergence(::AbstractLikelihood, qΩ::NamedTuple, pΩ::NamedTuple)
    return mapreduce(+, qΩ, pΩ) do qω, pω
        mapreduce(Distributions.kldivergence, +, qω, pω)
    end
end

function auglik_potential_and_precision(lik::AbstractLikelihood, Ω, y)
    return (auglik_potential(lik, Ω, y), auglik_precision(lik, Ω, y))
end

function expected_auglik_potential_and_precision(lik::AbstractLikelihood, qΩ, y)
    return (expected_auglik_potential(lik, qΩ, y), expected_auglik_precision(lik, qΩ, y))
end

nlatent(::AbstractLikelihood) = 1
