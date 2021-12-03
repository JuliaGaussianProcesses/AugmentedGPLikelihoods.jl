function aux_sample!(Ω, lik::AbstractLikelihood, y, f)
    return aux_sample!(GLOBAL_RNG, Ω, lik, y, f)
end

function aux_sample(lik::AbstractLikelihood, y, f)
    aux_sample(GLOBAL_RNG, lik, y, f)
end

function aux_sample(rng::AbstractRNG, lik::AbstractLikelihood, y, f)
    aux_sample!(rng, init_aux_variables(lik, length(y)), y, f)
end

function init_aux_variables(lik::AbstractLikelihood, n::Int)
    init_aux_variables(GLOBAL_RNG, lik, n)
end