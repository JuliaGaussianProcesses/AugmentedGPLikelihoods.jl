@doc raw"""
    NegativeMultinomial(x₀::Real, p::AbstractVector)

Negative Multinomial distribution defined as
```math
    p(\boldsymbol{x}|x_0, \boldsymbol{p}) = \Gamma\left(\sum_{i=0}^M x_i \right)\frac{p_0^{x_0}}{\Gamma(x_0)}\prod_{i=1}^M \frac{p_i^{x_i}}{x_i!}
```
where $p_0= 1-\sum_{i=1}^M p_i$.

For a detailed understanding of this distribution, see "Negative multinomial distribution" - Sibuya et al. - 1964
"""
struct NegativeMultinomial{Tx₀<:Real,Tp<:AbstractVector} <:
       Distributions.DiscreteMultivariateDistribution
    x₀::Tx₀
    p::Tp
    function NegativeMultinomial(x₀::Real, p::AbstractVector)
        x₀ > 0 || throw(ArgumentError("x₀ has to be positive"))
        (all(>=(0), p) && sum(p) < 1) || throw(
            ArgumentError(
                "All p should be positive and their sum should be strictly smaller than 1",
            ),
        )
        return new{typeof(x₀),typeof(p)}(x₀, p)
    end
end

_p₀(d::NegativeMultinomial) = 1 - sum(d.p)

Distributions.params(d::NegativeMultinomial) = (d.x₀, d.p)

Base.eltype(::NegativeMultinomial) = Int

Base.length(d::NegativeMultinomial) = length(d.p)

function Distributions._rand!(
    rng::AbstractRNG, d::NegativeMultinomial, x::AbstractVector{<:Real}
)
    p₀ = _p₀(d)
    θ = rand(rng, Gamma(d.x₀, inv(p₀) - 1))
    λ = d.p * θ / (1 - p₀) # convert parameters to the scaled Poisson ones
    for i in eachindex(x)
        x[i] = rand(rng, Poisson(λ[i]))
    end
    return x
end

function Distributions._logpdf(d::NegativeMultinomial, x::AbstractVector)
    return loggamma(d.x₀ + sum(x)) + d.x₀ * log(_p₀(d)) - loggamma(d.x₀) +
           mapreduce(+, d.p, x) do pᵢ, xᵢ
        xᵢ * log(pᵢ) - logfactorial(xᵢ)
    end
end

Distributions.mean(d::NegativeMultinomial) = d.x₀ / _p₀(d) * d.p

function Distributions.var(d::NegativeMultinomial)
    p₀ = _p₀(d)
    x₀ = d.x₀
    return x₀ / p₀^2 * abs2.(d.p) + x₀ / p₀ * p
end

function Distributions.cov(d::NegativeMultinomial)
    p₀ = _p₀(d)
    x₀ = d.x₀
    return x₀ / p₀^2 * d.p * d.p' + x₀ / p₀ * Diagonal(d.p)
end

function Distributions.mgf(d::NegativeMultinomial, t::AbstractVector)
    return (_p₀(d) / (1 - dot(d.p, exp.(t))))^d.x₀
end

function Distributions.kldivergence(p::NegativeMultinomial, q::NegativeMultinomial)
    p₀ = _p₀(p)
    x₀ = p.x₀
    x₀ == q.x₀ || error(
        "KL divergence cannot be computed with different x₍", "p: $(p.x₀), q: $(q.x₀)"
    )
    return x₀ * log(p₀) - x₀ * log(_p₀(q)) +
           x₀ / p₀ * sum(1:length(p)) do i
        p.p[i] * (log(p.p[i]) - log(q.p[i]))
    end
end
