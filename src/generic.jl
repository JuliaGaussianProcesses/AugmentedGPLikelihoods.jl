function aux_sample!(Ω, lik::AbstractLikelihood, y, f)
    return aux_sample!(GLOBAL_RNG, Ω, lik, y, f)
end

function aux_sample!(
    rng::AbstractRNG, Ω, lik::AbstractLikelihood, y::AbstractVector, f::AbstractVector
)
    map!(Ω, y, f) do yᵢ, fᵢ
        ntrand(rng, aux_full_conditional(lik, yᵢ, fᵢ))
    end
    return Ω
end

function aux_sample(lik::AbstractLikelihood, y, f)
    return aux_sample(GLOBAL_RNG, lik, y, f)
end

function aux_sample(rng::AbstractRNG, lik::AbstractLikelihood, y, f)
    return aux_sample!(rng, init_aux_variables(lik, length(y)), lik, y, f)
end

function aux_posterior(lik::AbstractLikelihood, y, f)
    return aux_posterior!(init_aux_posterior(lik, length(y)), lik, y, f)
end

function aux_full_conditional(lik::AbstractLikelihood, y, f)
    return For(1:length(y)) do i
        aux_full_conditional(lik, y[i], f[i])
    end
end

function init_aux_variables(lik::AbstractLikelihood, n::Int)
    return init_aux_variables(GLOBAL_RNG, lik, n)
end

function init_aux_posterior(lik::AbstractLikelihood, n::Int)
    return init_aux_posterior(Float64, lik, n)
end

function logtilt(
    lik::AbstractLikelihood, Ω::TupleVector, y::AbstractVector, f::AbstractVector
)
    return mapreduce(+, aux_field(lik, Ω), y, f) do Ωᵢ, yᵢ, fᵢ
        logtilt(lik, Ωᵢ, yᵢ, fᵢ)
    end
end

function aug_loglik(lik::AbstractLikelihood, Ω, y, f)
    return logtilt(lik, Ω, y, f) + logdensity_def(aux_prior(lik, y), Ω)
end

function expected_aug_loglik(lik::AbstractLikelihood, Ω, y, f)
    return expected_logtilt(lik, qΩ, y, qf) + aux_kldivergence(lik, qΩ, y)
end

function aux_kldivergence(lik::AbstractLikelihood, qΩ::For, y)
    return aux_kldivergence(lik, qΩ, aux_prior(lik, y))
end

function aux_kldivergence(::AbstractLikelihood, qΩ::For, pΩ::For)
    return mapreduce(Distributions.kldivergence, +, marginals(qΩ), marginals(pΩ))
end

function auglik_potential_and_precision(lik::AbstractLikelihood, Ω, y, f=nothing)
    return (auglik_potential(lik, Ω, y, f), auglik_precision(lik, Ω, y, f))
end

function expected_auglik_potential_and_precision(lik::AbstractLikelihood, qΩ, y, f=nothing)
    return (expected_auglik_potential(lik, qΩ, y, f), expected_auglik_precision(lik, qΩ, y, f))
end

auglik_potential(lik::AbstractLikelihood, Ω, y, _=nothing) = auglik_potential(lik, Ω, y)
auglik_precision(lik::AbstractLikelihood, Ω, y, _=nothing) = auglik_precision(lik, Ω, y)

expected_auglik_potential(lik::AbstractLikelihood, qΩ, y, _) = expected_auglik_potential(lik, qΩ, y)
expected_auglik_precision(lik::AbstractLikelihood, qΩ, y, _) = expected_auglik_precision(lik, qΩ, y)

# Generic wrapper for prior not taking any argument
aux_prior(lik::AbstractLikelihood, ::Real) = aux_prior(lik)

nlatent(::AbstractLikelihood) = 1 # Default number of latent for each likelihood
# This should potentially move to GPLikelihoods.jl
