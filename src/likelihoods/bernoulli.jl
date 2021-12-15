function init_aux_variables(rng::AbstractRNG, ::BernoulliLikelihood{<:LogisticLink}, n::Int)
    return TupleVector((; ω=rand(rng, PolyaGamma(1, 0.0), n)))
end

function init_aux_posterior(::BernoulliLikelihood{<:LogisticLink}, n::Int)
    return [PolyaGamma(1, 0.0) for _ in 1:n]
end

function aux_sample!(
    rng::AbstractRNG,
    Ω,
    ::BernoulliLikelihood{<:LogisticLink},
    ::AbstractVector,
    f::AbstractVector,
)
    return map!(Ω.ω, f) do f
        rand(rng, aux_full_conditional(lik, nothing, f))
    end
end

function aux_full_conditional(::BernoulliLikelihood{<:LogisticLink}, ::Any, f::Real)
    return PolyaGamma(1, abs(f))
end

function aux_posterior!(
    qΩ, ::BernoulliLikelihood{<:LogisticLink}, ::AbstractVector, qf::AbstractVector{<:Normal}
)
    return map!(qΩ, qf) do q
        PolyaGamma(1, sqrt(abs2(mean(q)) + var(q)))
    end
end

function auglik_potential(::BernoulliLikelihood{<:LogisticLink}, ::Any, y::AbstractVector)
    return (sign.(y .- 0.5) / 2,)
end

function auglik_precision(::BernoulliLikelihood{<:LogisticLink}, Ω, ::AbstractVector)
    return (Ω.ω,)
end

function expected_auglik_potential(
    lik::BernoulliLikelihood{<:LogisticLink}, qΩ, y::AbstractVector
)
    return auglik_potential(lik, qΩ, y)
end

function expected_auglik_precision(
    ::BernoulliLikelihood{<:LogisticLink}, qΩ, ::AbstractVector
)
    return (mean.(qΩ),)
end

function logtilt(::BernoulliLikelihood{<:LogisticLink}, Ω, y, f)
    return mapreduce(+, y, f, Ω.ω) do y, f, ω
        -log(2) + (sign(y - 0.5) * f - abs2(f) * ω) / 2
    end
end

function aux_prior(::BernoulliLikelihood{<:LogisticLink}, y)
    return Fill(PolyaGamma(1, 0.0), length(y))
end

function expected_logtilt(::BernoulliLikelihood{<:LogisticLink}, qΩ, y, qf)
    return mapreduce(+, y, qf, qΩ) do y, f, qω
        m = mean(f)
        -log(2) + (sign(y - 0.5) * m - (abs2(m) + var(f)) * mean(qω)) / 2
    end
end
