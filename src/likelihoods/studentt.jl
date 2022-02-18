# TODO Move this to GPLikelihoods.jl
@doc raw"""
    StudentTLikelihood(ν::Real, σ::Real)

Likelihood with a Student-T likelihood:
```math
    p(y|f,\sigma, \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\pi\nu}\sigma}\left(1 + \frac{1}{\nu}\left(\frac{x-\nu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}}.
```

## Arguments
- `ν::Real`, number of degrees of freedom, should be positive and larger than 0.5 to be able to compute moments
- `σ::Real`, scaling of the inputs.
"""
struct StudentTLikelihood{Tν<:Real,Tσ<:Real,Thalfν<:Real} <: AbstractLikelihood
    ν::Tν
    σ::Tσ
    σ²::Tσ
    halfν::Thalfν
end

StudentTLikelihood(ν::Real, σ::Real) = StudentTLikelihood(ν, σ, abs2(σ), ν / 2)

(lik::StudentTLikelihood)(f::Real) = LocationScale(f, lik.σ, TDist(lik.ν))

_α(lik::StudentTLikelihood) = (lik.ν + 1) / 2

function (lik::StudentTLikelihood)(f::AbstractVector{<:Real})
    return Product(lik.(f))
end

aux_field(::StudentTLikelihood, Ω) = getproperty(Ω, :ω) 

function init_aux_variables(rng::AbstractRNG, ::StudentTLikelihood, n::Int)
    return TupleVector((; ω=rand(rng, Gamma(), n)))
end

function init_aux_posterior(T::DataType, lik::StudentTLikelihood, n::Int)
    α = _α(lik)
    return For(TupleVector(; β=zeros(T, n))) do φ
        NTDist(Gamma(α, inv(φ.β))) # Distributions uses a different parametrization
    end
end

function aux_full_conditional(lik::StudentTLikelihood, y::Real, f::Real)
    return NTDist(Gamma(_α(lik), 2 / (lik.ν / abs2(lik.σ) + abs2(y - f))))
end

function aux_posterior!(
    qΩ, lik::StudentTLikelihood, y::AbstractVector, qf::AbstractVector{<:Normal}
)
    φ = qΩ.pars
    map!(φ.β, y, qf) do yᵢ, fᵢ
        (lik.ν / abs2(lik.σ) + second_moment(fᵢ, yᵢ)) / 2
    end
    return qΩ
end

function auglik_potential(::StudentTLikelihood, Ω, y::AbstractVector)
    return (y .* Ω.ω,)
end

function auglik_precision(::StudentTLikelihood, Ω, ::AbstractVector)
    return (Ω.ω,)
end

function expected_auglik_potential(::StudentTLikelihood, qΩ, y::AbstractVector)
    return (tvmean(qΩ).ω .* y,)
end

function expected_auglik_precision(::StudentTLikelihood, qΩ, ::AbstractVector)
    return (tvmean(qΩ).ω,)
end

function logtilt(::StudentTLikelihood, ω::Real, y::Real, f::Real)
    return logpdf(Normal(f, sqrt(inv(ω))), y)
end

function expected_logtilt(::StudentTLikelihood, qωᵢ::NTDist{<:Gamma}, yᵢ::Real, qfᵢ::Normal)
    θ = ntmean(qωᵢ)
    return logpdf(Normal(yᵢ, sqrt(inv(θ.ω))), mean(qfᵢ)) - var(qfᵢ) * θ.ω / 2
end

function aux_prior(lik::StudentTLikelihood, y::AbstractVector{<:Real})
    return For(length(y)) do _
        NTDist(aux_prior(lik))
    end
end

aux_prior(lik::StudentTLikelihood) = Gamma(lik.halfν, lik.σ² / lik.halfν)
