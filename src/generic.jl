function aux_sample!(Ω, lik::Likelihood, y, f)
    return aux_sample!(GLOBAL_RNG, Ω, lik, y, f)
end

function aux_sample(lik::Likelihood, y, f)
    aux_sample(GLOBAL_RNG, lik, y, f)
end

function aux_sample(rng::AbstractRNG, lik::Likelihood, y, f)
    aux_sample!(rng, init_aux_variables(lik, length(y)), y, f)
end