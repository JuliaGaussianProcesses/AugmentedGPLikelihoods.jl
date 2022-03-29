aux_field(::NegativeBinomialLikelihood, Ω) = getproperty(Ω, :ω)

function init_aux_variables(rng::AbstractRNG, ::NegativeBinomialLikelihood, n::Int)
    return TupleVector((; ω=rand(rng, PolyaGamma(1, 0), n)))
end

function init_aux_posterior(T::DataType, lik::NegativeBinomialLikelihood, n::Int)
    return For(TupleVector(; y=zeros(Int, n), c=zeros(T, n))) do φ
        NTDist(PolyaGamma(φ.y + lik.r, φ.c)) # Distributions uses a different parametrization
    end
end

function aux_full_conditional(lik::NegativeBinomialLikelihood, y::Real, f::Real)
    return NTDist(PolyaGamma(y + lik.r, abs(f)))
end

function aux_posterior!(
    qΩ, ::NegativeBinomialLikelihood, y::AbstractVector, qf::AbstractVector{<:Normal}
)
    φ = qΩ.pars
    φ.y .= y
    map!(φ.c, qf) do fᵢ
        sqrt(second_moment(fᵢ))
    end
    return qΩ
end

function auglik_potential(lik::NegativeBinomialLikelihood, ::Any, y::AbstractVector)
    return ((y .- lik.r) / 2,)
end

function auglik_precision(::NegativeBinomialLikelihood, Ω, ::AbstractVector)
    return (Ω.ω,)
end

function expected_auglik_potential(lik::NegativeBinomialLikelihood, qΩ, y::AbstractVector)
    return auglik_potential(lik, qΩ, y)
end

function expected_auglik_precision(::NegativeBinomialLikelihood, qΩ, ::AbstractVector)
    return (tvmean(qΩ).ω,)
end

negbin_logconst(y, r::Real) = loggamma(y + r) - loggamma(y + 1) - loggamma(r)
negbin_logconst(y, r::Int) = first(logabsbinomial(y + r - 1, y))

function logtilt(lik::NegativeBinomialLikelihood, ω::Real, y::Integer, f::Real)
    negbin_logconst(y, lik.r) - (y + lik.r) * logtwo +
        (f * (y - lik.r) - abs2(f) * ω) / 2
end

function expected_logtilt(
    lik::NegativeBinomialLikelihood, qωᵢ::NTDist{<:PolyaGamma}, yᵢ::Integer, qfᵢ::Normal
)
    θ = ntmean(qωᵢ)
    return negbin_logconst(yᵢ, lik.r) - (yᵢ + lik.r) * logtwo +
           (mean(qfᵢ) * (yᵢ - lik.r) - second_moment(qfᵢ) * θ.ω) / 2
end

function aux_prior(lik::NegativeBinomialLikelihood, y::AbstractVector{<:Integer})
    return For(y) do yᵢ
        NTDist(aux_prior(lik, yᵢ))
    end
end

aux_prior(lik::NegativeBinomialLikelihood, y::Int) = PolyaGamma(lik.r + y, 0)
