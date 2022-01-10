# TODO Move this to GPLikelihoods.jl
@doc raw"""
    LaplaceLikelihood(β::Real)

Likelihood with a [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution):
```math
    p(y|f,\beta) = \frac{1}{2\beta}\exp\left(-\frac{|y-f|}{\beta}\right)
```

## Arguments
- `β::Real`, scale parameter
"""
struct LaplaceLikelihood{Tβ<:Real} <: AbstractLikelihood
    β::Tβ
end

LaplaceLikelihood() = LaplaceLikelihood(1.0)

(lik::LaplaceLikelihood)(f::Real) = Laplace(f, lik.β)

function (lik::LaplaceLikelihood)(f::AbstractVector{<:Real})
    return Product(lik.(f))
end

@inline laplace_λ(lik::LaplaceLikelihood) = inv(2 * abs2(lik.β))

function init_aux_variables(rng::AbstractRNG, ::LaplaceLikelihood, n::Int)
    return TupleVector((; ω=rand(rng, InverseGamma(), n)))
end

function init_aux_posterior(T::DataType, lik::LaplaceLikelihood, n::Int)
    λ = laplace_λ(lik)
    return For(TupleVector(; μ=zeros(T, n))) do φ
        NTDist(InverseGaussian(φ.μ, λ))
    end
end

function aux_full_conditional(lik::LaplaceLikelihood, y::Real, f::Real)
    return NTDist(InverseGaussian(inv(2 * lik.β * abs(y - f)), 2 * laplace_λ(lik)))
end

function aux_posterior!(
    qΩ, lik::LaplaceLikelihood, y::AbstractVector, qf::AbstractVector{<:Normal}
)
    φ = qΩ.pars
    map!(φ.μ, y, qf) do yᵢ, qfᵢ
        inv(2 * lik.β * sqrt(second_moment(qfᵢ, yᵢ)))
    end
    return qΩ
end

function auglik_potential(::LaplaceLikelihood, Ω, y::AbstractVector)
    return (y .* Ω.ω,)
end

function auglik_precision(::LaplaceLikelihood, Ω, ::AbstractVector)
    return (Ω.ω,)
end

function expected_auglik_potential(::LaplaceLikelihood, qΩ, y::AbstractVector)
    return (tvmean(qΩ).ω .* y,)
end

function expected_auglik_precision(::LaplaceLikelihood, qΩ, ::AbstractVector)
    return (tvmean(qΩ).ω,)
end

function logtilt(lik::LaplaceLikelihood, Ω, y, f)
    return mapreduce(+, y, f, Ω.ω) do yᵢ, fᵢ, ωᵢ
        -log(sqrtπ) + loggamma(1//2)  - abs2(yᵢ - fᵢ) * ωᵢ - log(2 * lik.β)
    end
end

function expected_logtilt(lik::LaplaceLikelihood, qΩ, y, qf)
    return mapreduce(+, y, qf, marginals(qΩ)) do yᵢ, qfᵢ, qωᵢ
        θ = ntmean(qωᵢ)
        -log(sqrtπ) + loggamma(1//2) - log(2 * lik.β) - second_moment(qfᵢ, yᵢ) * θ.ω
    end
end

function aux_prior(lik::LaplaceLikelihood, y)
    return For(length(y)) do _
        NTDist(InverseGamma(1//2, laplace_λ(lik)))
    end
end
