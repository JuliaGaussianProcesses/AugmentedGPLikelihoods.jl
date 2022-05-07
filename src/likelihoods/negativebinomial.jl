const NegBinomialLikelihood = NegativeBinomialLikelihood{<:NBParamFailure}

@deprecate NegBinomialLikelihood(r::Real) NegativeBinomialLikelihood(NBParamFailure(r))
@deprecate NegBinomialLikelihood(link::AbstractLink, r::Real) NegativeBinomialLikelihood(NBParamFailure(r), link)

aux_field(::NegBinomialLikelihood, Ω) = getproperty(Ω, :ω)

function init_aux_variables(rng::AbstractRNG, ::NegBinomialLikelihood, n::Int)
    return TupleVector((; ω=rand(rng, PolyaGamma(1, 0), n)))
end

function init_aux_posterior(T::DataType, lik::NegBinomialLikelihood, n::Int)
    return For(TupleVector(; y=zeros(Int, n), c=zeros(T, n))) do φ
        NTDist(PolyaGamma(φ.y + lik.params.failures, φ.c)) # Distributions uses a different parametrization
    end
end

function aux_full_conditional(lik::NegBinomialLikelihood, y::Real, f::Real)
    return NTDist(PolyaGamma(y + lik.params.failures, abs(f)))
end

function aux_posterior!(
    qΩ, ::NegBinomialLikelihood, y::AbstractVector, qf::AbstractVector{<:Normal}
)
    φ = only(qΩ.inds)
    φ.y .= y
    map!(φ.c, qf) do fᵢ
        sqrt(second_moment(fᵢ))
    end
    return qΩ
end

function auglik_potential(lik::NegBinomialLikelihood, ::Any, y::AbstractVector)
    return ((y .- lik.params.failures) / 2,)
end

function auglik_precision(::NegBinomialLikelihood, Ω, ::AbstractVector)
    return (Ω.ω,)
end

function expected_auglik_potential(lik::NegBinomialLikelihood, qΩ, y::AbstractVector)
    return auglik_potential(lik, qΩ, y)
end

function expected_auglik_precision(::NegBinomialLikelihood, qΩ, ::AbstractVector)
    return (tvmean(qΩ).ω,)
end

negbin_logconst(y, r::Real) = loggamma(y + r) - loggamma(y + 1) - loggamma(r)
negbin_logconst(y, r::Int) = first(logabsbinomial(y + r - 1, y))

function logtilt(lik::NegBinomialLikelihood, ω::Real, y::Integer, f::Real)
    return negbin_logconst(y, lik.params.failures) - (y + lik.params.failures) * logtwo +
           (f * (y - lik.params.failures) - abs2(f) * ω) / 2
end

function expected_logtilt(
    lik::NegBinomialLikelihood, qωᵢ::NTDist{<:PolyaGamma}, yᵢ::Integer, qfᵢ::Normal
)
    θ = ntmean(qωᵢ)
    return negbin_logconst(yᵢ, lik.params.failures) - (yᵢ + lik.params.failures) * logtwo +
           (mean(qfᵢ) * (yᵢ - lik.params.failures) - second_moment(qfᵢ) * θ.ω) / 2
end

function aux_prior(lik::NegBinomialLikelihood, y::AbstractVector{<:Integer})
    return For(y) do yᵢ
        NTDist(aux_prior(lik, yᵢ))
    end
end

aux_prior(lik::NegBinomialLikelihood, y::Int) = PolyaGamma(lik.params.failures + y, 0)
