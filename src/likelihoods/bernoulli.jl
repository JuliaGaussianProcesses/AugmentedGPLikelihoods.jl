function init_aux_variables(rng::AbstractRNG, ::BernoulliLikelihood{<:LogisticLink}, n::Int)
    return (;ω=rand(rng, PolyaGamma(1, 0.0), n))
end

function init_aux_posterior(::BernoulliLikelihood{<:LogisticLink}, n::Int)
    return (;ω=[PolyaGamma(1, 0.0) for _ in 1:n])
end

function aux_sample!(rng::AbstractRNG, Ω, ::BernoulliLikelihood{<:LogisticLink}, ::AbstractVector, f::AbstractVector)
    map!(Ω.ω, f) do f
        rand(rng, PolyaGamma(1, abs(f)))
    end
    return Ω
end

function aux_posterior!(Ω, ::BernoulliLikelihood{<:LogisticLink}, ::AbstractVector, q_f::AbstractVector{<:Normal})
    map!(Ω.ω, q_f) do q
        PolyaGamma(1, sqrt(abs2(mean(q)) + var(q)))
    end
    return Ω
end

function vi_shift(::BernoulliLikelihood{<:LogisticLink}, ::Any, y::AbstractVector)
    return (sign.(y .- 0.5),)
end

function vi_rate(::BernoulliLikelihood{<:LogisticLink}, Ω, ::AbstractVector)
    return (mean.(Ω.ω),)
end

function sample_shift(lik::BernoulliLikelihood{<:LogisticLink}, Ω, y::AbstractVector)
    vi_shift(lik, Ω, y)
end

function sample_rate(::BernoulliLikelihood{<:LogisticLink}, Ω, ::AbstractVector)
    (Ω.ω,)
end

function aug_loglik(::BernoulliLikelihood{<:LogisticLink}, Ω, y, f)
    return mapreduce(+, y, f, Ω.ω) do y, f, ω
        -log(2) + (sign(y - 0.5) * f - abs2(f) * ω) / 2
    end
end

function aux_prior(::BernoulliLikelihood{<:LogisticLink}, y)
    return (;ω=[PolyaGamma(1, 0.0) for _ in 1:length(y)])
end

function aug_expected_loglik(::BernoulliLikelihood{<:LogisticLink}, Ω, y, qf)
    return mapreduce(+, y, qf, Ω.ω) do y, f, ω
        m = mean(f)
        -log(2) + (sign(y - 0.5) * m - (abs2(m) + var(f)) * mean(ω)) / 2
    end
end

function kl_term(::BernoulliLikelihood{<:LogisticLink}, Ω, y)
    return sum(kl_term, Ω.ω)
end

# Shortcut for computating KL(PG(ω|b, c)||PG(b, 0))
function kl_term(q::PolyaGamma)
    return q.b * logcosh(q.c / 2) - abs2(q.c) * mean(q) / 2
end