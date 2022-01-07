# TODO Move this to GPLikelihoods.jl
@doc raw"""
    NegBinomialLikelihood(r::Real)

Likelihood with a Negative Binomial distribution:
```math
    p(y|f, r) = {y + r - 1 \chose y}(1-\sigma(f))^r \sigma^y(f).
```

## Arguments
- `r::Real`, number of failures until the experiment is stopped
"""
struct NegBinomialLikelihood{Tl<:AbstractLink,Tr<:Real} <: AbstractLikelihood
    invlink::Tl
    r::Tr
end

NegBinomialLikelihood(r::Real) = NegBinomialLikelihood(LogisticLink(), r)

function (lik::NegBinomialLikelihood)(f::AbstractVector{<:Real})
    return Product(lik.(f))
end

function init_aux_variables(rng::AbstractRNG, ::NegBinomialLikelihood, n::Int)
    return TupleVector((; ω=rand(rng, PolyaGamma(1, 0), n)))
end

function init_aux_posterior(T::DataType, lik::NegBinomialLikelihood, n::Int)
    return For(TupleVector(; y=zeros(Int, n), c=zeros(T, n))) do φ
        NTDist(PolyaGamma(φ.y + lik.r, φ.c)) # Distributions uses a different parametrization
    end
end

function aux_full_conditional(lik::NegBinomialLikelihood, y::Real, f::Real)
    return NTDist(PolyaGamma(y + lik.r, abs(f)))
end

function aux_posterior!(
    qΩ, ::NegBinomialLikelihood, y::AbstractVector, qf::AbstractVector{<:Normal}
)
    φ = qΩ.pars
    φ.y = y
    map!(φ.c, qf) do fᵢ
        sqrt(second_moment(fᵢ))
    end
    return qΩ
end

function auglik_potential(lik::NegBinomialLikelihood, ::Any, y::AbstractVector)
    return ((y .- lik.r) / 2,)
end

function auglik_precision(::NegBinomialLikelihood, Ω, ::AbstractVector)
    return (Ω.ω,)
end

function expected_auglik_potential(::NegBinomialLikelihood, qΩ, y::AbstractVector)
    return auglik_potential(lik, qΩ, y)
end

function expected_auglik_precision(::NegBinomialLikelihood, qΩ, ::AbstractVector)
    return (tvmean(qΩ).ω,)
end

function logtilt(lik::NegBinomialLikelihood, Ω, y, f)
    return mapreduce(+, y, f, Ω.ω) do yᵢ, fᵢ, ωᵢ
        logfactorial(yᵢ + lik.r - 1, yᵢ) - (yᵢ + lik.r) * logtwo + (fᵢ * (yᵢ - lik.r) - abs2(fᵢ) * ωᵢ) / 2
    end
end

function expected_logtilt(::NegBinomialLikelihood, qΩ, y, qf)
    return mapreduce(+, y, qf, marginals(qΩ)) do yᵢ, qfᵢ, qωᵢ
        θ = ntmean(qωᵢ)
        logfactorial(yᵢ + lik.r - 1, yᵢ) - (yᵢ + lik.r) * logtwo + (mean(qfᵢ) * (yᵢ - lik.r) - second_moment(qfᵢ) * θ.ωᵢ) / 2
    end
end

function aux_prior(lik::NegBinomialLikelihood, y)
    return For(y) do yᵢ
        NTDist(PolyaGamma(lik.r + yᵢ, 0))
    end
end
