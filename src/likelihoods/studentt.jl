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
struct StudentTLikelihood{Tν<:Real,Tσ<:Real} <: AbstractLikelihood
    ν::Tν
    σ::Tσ
end

(lik::StudentTLikelihood)(f::Real) = LocationScale(f, lik.σ, TDist(lik.ν))

_α(lik::StudentTLikelihood) = (lik.ν + 1) / 2

function (lik::StudentTLikelihood)(f::AbstractVector{<:Real})
    return Product(lik.(f))
end

function init_aux_variables(rng::AbstractRNG, ::StudentTLikelihood, n::Int)
    return TupleVector((; ω=rand(rng, InverseGamma(), n)))
end

function init_aux_posterior(T::DataType, lik::StudentTLikelihood, n::Int)
    α = _α(lik)
    return For(TupleVector(; β=zeros(T, n))) do φ
        NTDist(InverseGamma(α, φ.β))
    end
end

function aux_full_conditional(lik::StudentTLikelihood, y::Real, f::Real)
    return NTDist(InverseGamma(_α(lik), (abs2(lik.σ) * lik.ν + abs2(y - f)) / 2))
end

function aux_posterior!(
    qΩ, lik::StudentTLikelihood, y::AbstractVector, qf::AbstractVector{<:Normal}
)
    φ = qΩ.pars
    map!(φ.β, y, qf) do yᵢ, qfᵢ
        (abs2(lik.σ) * lik.ν + second_moment(qfᵢ, yᵢ)) / 2
    end
    return qΩ
end

# TODO use a different parametrization to avoid all these inverses
function auglik_potential(::StudentTLikelihood, Ω, y::AbstractVector)
    return (y ./ Ω.ω,)
end

function auglik_precision(::StudentTLikelihood, Ω, ::AbstractVector)
    return (inv.(Ω.ω),)
end

function expected_auglik_potential(::StudentTLikelihood, qΩ, y::AbstractVector)
    return (tvmeaninv(qΩ).ω .* y,)
end

function expected_auglik_precision(::StudentTLikelihood, qΩ, ::AbstractVector)
    return (tvmeaninv(qΩ).ω,)
end

function logtilt(::StudentTLikelihood, Ω, y, f)
    return mapreduce(+, y, f, Ω.ω) do yᵢ, fᵢ, ω
        logpdf(Normal(yᵢ, ω), fᵢ)
    end
end

function aux_prior(lik::StudentTLikelihood, y)
    halfν = lik.ν / 2
    return For(length(y)) do _
        NTDist(InverseGamma(halfν, halfν * abs2(lik.σ)))
    end
end

function expected_logtilt(::StudentTLikelihood, qΩ, y, qf)
    return mapreduce(+, y, qf, marginals(qΩ)) do yᵢ, fᵢ, qω
        θ = ntmeaninv(qω)
        logpdf(Normal(yᵢ, θ.ω), mean(fᵢ)) - var(fᵢ) * θ.ω / 2
    end
end
