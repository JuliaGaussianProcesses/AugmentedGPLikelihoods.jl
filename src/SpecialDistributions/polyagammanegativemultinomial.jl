@doc raw"""
    PolyaGammaNegativeMultinomial(y::BitVector, c::AbstractVector{<:Real}, p::AbstractVector{<:Real})

A multivariate distribution, used as hierachical prior as:
```math
    p(\boldsymbol{\omega}, \boldsymbol{n}) = \operatorname{NM}(\boldsymbol{n}|1, \boldsymbol{p})\prod_{i=1}^K\operatorname{PG}(\omega|y_i + n_i, c).
```

Random samples as well as statistics from the distribution will returned as a `NamedTuple` : `(;ω, n)`.

This structured distributions is needed for the [`CategoricalLikelihood`](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/stable/api#GPLikelihoods.CategoricalLikelihood) with a `LogisticSoftMaxLink`. 
"""
struct PolyaGammaNegativeMultinomial{Ty,Tc,Tp} <: AbstractNTDist
    y::Ty # Intermediate first parameter for PG(y + n, c)
    c::Tc # Second parameter for PG
    p::Tp # Negative Multinomial parameters
end

function Base.:(==)(p::PolyaGammaNegativeMultinomial, q::PolyaGammaNegativeMultinomial)
    return (p.y == q.y) && (p.c == q.c) && (p.p == q.p) 
end

NegativeMultinomial(d::PolyaGammaNegativeMultinomial) = NegativeMultinomial(1, d.p)

Distributions.length(::PolyaGammaNegativeMultinomial) = 2 * length(d.p)

function ntrand(rng::AbstractRNG, d::PolyaGammaNegativeMultinomial)
    n = rand(rng, NegativeMultinomial(d))
    ω = rand.(rng, PolyaGamma.(n + d.y, d.c))
    return (; ω, n)
end

function MeasureBase.logdensity(d::PolyaGammaNegativeMultinomial, x::NamedTuple)
    logpdf_n = logpdf(NegativeMultinomial(d), x.n)
    logpdf_ω = sum(1:length(x)) do i
        return logpdf(PolyaGamma(d.y[i] + x.n[i], d.c[i]), x.ω[i])
    end
    return logpdf_ω + logpdf_n
end

function tvmean(ds::AbstractVector{<:PolyaGammaNegativeMultinomial})
    n = nestedview(mapreduce(d -> mean(NegativeMultinomial(d)), hcat, ds))
    ω = nestedview(
        mapreduce(hcat, ds, n) do d, n
            mean.(PolyaGamma.(d.y + n, d.c))
        end,
    )
    return TupleVector(; ω, n)
end

function ntmean(d::PolyaGammaNegativeMultinomial)
    n = mean(NegativeMultinomial(d))
    return (; ω=mean.(PolyaGamma.(d.y + n, d.c)), n)
end

function Distributions.kldivergence(
    q::PolyaGammaNegativeMultinomial, p::PolyaGammaNegativeMultinomial
)
    # TODO: Optimize this
    (all(==(0), p.c) && all(p.y .== q.y)) ||
        error("No solution for this prior. qΩ = $q, pΩ = $p")
    n = mean(NegativeMultinomial(q))
    return sum(kldivergence.(PolyaGamma.(q.y + n, q.c), PolyaGamma.(q.y + n, 0))) +
           kldivergence(NegativeMultinomial(q), NegativeMultinomial(p))
end
