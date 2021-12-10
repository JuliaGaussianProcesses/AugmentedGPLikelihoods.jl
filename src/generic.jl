function aux_sample!(Ω, lik::AbstractLikelihood, y, f)
    return aux_sample!(GLOBAL_RNG, Ω, lik, y, f)
end

function aux_sample(lik::AbstractLikelihood, y, f)
    aux_sample(GLOBAL_RNG, lik, y, f)
end

function aux_sample(rng::AbstractRNG, lik::AbstractLikelihood, y, f)
    aux_sample!(rng, init_aux_variables(lik, length(y)), lik, y, f)
end

function init_aux_variables(lik::AbstractLikelihood, n::Int)
    init_aux_variables(GLOBAL_RNG, lik, n)
end

function aux_posterior(lik::AbstractLikelihood, y, qf)
    aux_posterior!(init_aux_posterior(lik, length(y)), lik, y, qf)
end

function aug_loglik(lik::AbstractLikelihood, Ω, y, f)
    return logtilt(lik, Ω, y, f) + mapreduce(+, Ω, aux_prior(lik, y)) do ω, pω
        mapreduce(logpdf, +, pω, ω)
    end
end

aux_kldivergence(lik::AbstractLikelihood, qΩ::NamedTuple, y) = aux_kldivergence(lik, qΩ, aux_prior(lik, y))

function aux_kldivergence(::AbstractLikelihood, qΩ::NamedTuple, pΩ::NamedTuple)
    return mapreduce(+, qΩ, pΩ) do qω, pω
        mapreduce(Distributions.kldivergence, +, qω, pω)
    end
end

nlatent(::AbstractLikelihood) = 1
