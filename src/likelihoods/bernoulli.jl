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

function aux_posterior!(Ω, ::BernoulliLikelihood{<:LogisticLink}, ::AbstractVector, qf::AbstractVector{<:Normal})
    map!(Ω.ω, qf) do q
        PolyaGamma(1, sqrt(abs2(mean(q)) + var(q)))
    end
    return Ω
end

function auglik_potential(::BernoulliLikelihood{<:LogisticLink}, ::Any, y::AbstractVector)
    (sign.(y .- 0.5),)
end

function auglik_precision(::BernoulliLikelihood{<:LogisticLink}, Ω, ::AbstractVector)
    (Ω.ω,)
end

function expected_auglik_potential(lik::BernoulliLikelihood{<:LogisticLink}, Ω, y::AbstractVector)
    return auglik_potential(lik, Ω, y)
end

function expected_auglik_precision(::BernoulliLikelihood{<:LogisticLink}, Ω, ::AbstractVector)
    return (mean.(Ω.ω),)
end


function logtilt(::BernoulliLikelihood{<:LogisticLink}, Ω, y, f)
    return mapreduce(+, y, f, Ω.ω) do y, f, ω
        -log(2) + (sign(y - 0.5) * f - abs2(f) * ω) / 2
    end
end

function aux_prior(::BernoulliLikelihood{<:LogisticLink}, y)
    return (;ω=[PolyaGamma(1, 0.0) for _ in 1:length(y)])
end

function expected_logtilt(::BernoulliLikelihood{<:LogisticLink}, Ω, y, qf)
    return mapreduce(+, y, qf, Ω.ω) do y, f, ω
        m = mean(f)
        -log(2) + (sign(y - 0.5) * m - (abs2(m) + var(f)) * mean(ω)) / 2
    end
end

function kl_term(lik::BernoulliLikelihood{<:LogisticLink}, Ω, y)
    return mapreduce(kldivergence, +, Ω.ω, aux_prior(lik, y).ω)
end