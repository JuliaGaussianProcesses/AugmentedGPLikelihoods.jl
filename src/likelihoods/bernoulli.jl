aux_field(::BernoulliLikelihood{<:LogisticLink}, Ω) = getproperty(Ω, :ω)

function init_aux_variables(rng::AbstractRNG, ::BernoulliLikelihood{<:LogisticLink}, n::Int)
    return TupleVector(; ω=rand(rng, PolyaGamma(1, 0.0), n))
end

function init_aux_posterior(T::DataType, ::BernoulliLikelihood{<:LogisticLink}, n::Int)
    return For(TupleVector(; c=zeros(T, n))) do φ
        NTDist(PolyaGamma(1, φ.c))
    end
end

function aux_full_conditional(::BernoulliLikelihood{<:LogisticLink}, ::Any, f::Real)
    return NTDist(PolyaGamma(1, abs(f)))
end

function aux_posterior!(
    qΩ,
    ::BernoulliLikelihood{<:LogisticLink},
    ::AbstractVector,
    qf::AbstractVector{<:Normal},
)
    map!(sqrt ∘ second_moment, only(qΩ.inds).c, qf)
    return qΩ
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
    return (tvmean(qΩ).ω,)
end

function logtilt(::BernoulliLikelihood{<:LogisticLink}, ω::Real, y::Real, f::Real)
    return -log(2) + (sign(y - 0.5) * f - abs2(f) * ω) / 2
end

function aux_prior(lik::BernoulliLikelihood{<:LogisticLink}, y::AbstractVector)
    return For(length(y)) do _
        NTDist(aux_prior(lik))
    end
end

aux_prior(::BernoulliLikelihood{<:LogisticLink}) = PolyaGamma(1, 0.0)

function expected_logtilt(
    ::BernoulliLikelihood{<:LogisticLink}, qωᵢ::NTDist{<:PolyaGamma}, yᵢ::Real, qfᵢ::Normal
)
    m = mean(qfᵢ)
    θ = ntmean(qωᵢ)
    return -log(2) + (sign(yᵢ - 0.5) * m - (abs2(m) + var(qfᵢ)) * θ.ω) / 2
end
