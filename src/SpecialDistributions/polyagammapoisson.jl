@doc raw"""
    PolyaGammaPoisson(y::Real, c::Real, λ::Real)

A bivariate distribution, used as hierachical prior as:
```math
    p(\omega, n) = \operatorname{PG}(\omega|y + n, c)\operatorname{Po}(n|\lambda).
```

Random samples as well as statistics from the distribution will returned as `NamedTuple` : `(;ω, n)`.

This structured distributions is needed for example for the [`PoissonLikelihood`](@ref).
"""
struct PolyaGammaPoisson{Ty,Tc,Tλ}
    y::Ty # Intermediate first parameter for PG(y + n, c)
    c::Tc # Second parameter for PG
    λ::Tλ # Poisson Parameter
end

Distributions.Poisson(d::PolyaGammaPoisson) = Poisson(d.λ)

Distributions.length(::PolyaGammaPoisson) = 2

function Distributions.rand(rng::AbstractRNG, d::PolyaGammaPoisson)
    n = rand(rng, Poisson(d))
    ω = rand(rng, PolyaGamma.(n + d.y, d.c))
    return (; ω, n)
end

function Distributions.logpdf(d::PolyaGammaPoisson, x::NamedTuple)
    logpdf_n = logpdf(Poisson(d), x.n)
    logpdf_ω = logpdf(PolyaGamma(d.y + x.n, d.c), x.ω)
    return logpdf_ω + logpdf_n
end

function Distributions.mean(ds::AbstractVector{<:PolyaGammaPoisson})
    n = getfield.(ds, :λ)
    ω = map(ds, n) do d, n
        mean(PolyaGamma(d.y + n, d.c))
    end
    return TupleVector((; ω, n))
end

function Distributions.mean(d::PolyaGammaPoisson)
    return (; ω=mean(PolyaGamma(d.y + d.λ, d.c)), n=d.λ)
end

function Distributions.kldivergence(q::PolyaGammaPoisson, p::PolyaGammaPoisson)
    (p.c == 0 && p.y == q.y) || error("No solution for this prior")
    return kldivergence(PolyaGamma(q.y + q.λ, q.c), PolyaGamma(q.y + q.λ, 0)) +
           kldivergence(Poisson(q), Poisson(p))
end
