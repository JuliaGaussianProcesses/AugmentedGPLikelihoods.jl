const pg_t = 0.64
const pg_inv_t = inv(pg_t)

@doc raw"""
    PolyaGamma(b::Real, c::Real) <: ContinuousUnivariateDistribution

## Arguments
- `b::Real` 
- `c::Real` exponential tilting

Create a PolyaGamma sampler with parameters `b` and `c`.
Note that sampling will differ if `b` is a `Int` or a `Real`.
"""
struct PolyaGamma{Tb,Tc} <: Distributions.ContinuousUnivariateDistribution
    b::Tb
    c::Tc
end

Base.eltype(::PolyaGamma{T,Tc}) where {T,Tc} = Tc

Distributions.params(d::PolyaGamma) = (d.b, d.c)

function Statistics.mean(d::PolyaGamma)
    if iszero(d.c)
        return d.b / 4
    else
        return d.b / (2 * d.c) * tanh(d.c / 2)
    end
end


Base.minimum(d::PolyaGamma) = zero(eltype(d))
Base.maximum(::PolyaGamma) = Inf
Distributions.insupport(::PolyaGamma, x::Real) = zero(x) <= x < Inf

function Distributions.logpdf(d::PolyaGamma, x::Real)
    b, c = Distributions.params(d)
    if iszero(b)
        return iszero(x) ? zero(x) : -Inf # if b is zero, then the distribution
    # simplified as a delta dirac.
    else
        iszero(x) && -Inf # The limit to p(x) for x-> 0 is 0.
        return logtilt(x, b, c) + (b - 1) * log(2) - loggamma(b) + log(
            sum(0:200) do n
                ifelse(iseven(n), 1, -1) * exp(
                    loggamma(n + b) - loggamma(n + 1) + log(2n + b) - log(twoπ * x^3) / 2 -
                    abs2(2n + b) / (8x),
                )
            end,
        )
    end
end

Distributions.logpdf(d::PolyaGamma, x::NamedTuple{(:ω,),<:Tuple{<:Real}}) = logpdf(d, x.ω)

# Shortcut for computating KL(PG(ω|b, c)||PG(b, 0))
function Distributions.kldivergence(q::PolyaGamma, p::PolyaGamma)
    (q.b == p.b && iszero(p.c)) || error(
        "cannot compute the KL divergence for Polya-Gamma distributions",
        "with different parameter b, (q.b = $(q.b), p.b = $(p.b))",
        " or if p.c ≠ 0 (p.c = $(p.c))",
    )
    return logtilt(mean(q), q.b, q.c)
end

function logtilt(ω, b, c)
    return b * log(cosh(c / 2)) - abs2(c) * ω / 2
end

function ntrand(rng::AbstractRNG, d::PolyaGamma)
    return (; ω=rand(rng, d))
end

# Dispatch for MeasureTheory
function Base.rand(rng::AbstractRNG, ::Type{T}, d::PolyaGamma) where {T}
    return ntrand(rng, d)
end

function Distributions.rand(rng::AbstractRNG, d::PolyaGamma)
    if iszero(d.b)
        return zero(eltype(d))
    end
    return draw_sum(rng, d)
end

## Sampling when `b` is an integer
function draw_sum(rng::AbstractRNG, d::PolyaGamma{<:Int})
    return sum(Base.Fix1(sample_pg1, rng), d.c * ones(d.b))
end

function draw_sum(rng::AbstractRNG, d::PolyaGamma{<:Real})
    if d.b < 1
        return rand_gamma_sum(rng, d, d.b)
    end
    trunc_b = floor(Int, d.b)
    res_b = d.b - trunc_b
    trunc_term = sum(Base.Fix1(sample_pg1, rng), d.c * ones(trunc_b))
    res_term = rand_gamma_sum(rng, d, res_b)
    return trunc_term + res_term
end

## Utility functions
function a(n::Int, x::Real)
    k = (n + 0.5) * π
    if x > pg_t
        return k * exp(-k^2 * x / 2)
    elseif x > 0
        expnt = -3 / 2 * (log(halfπ) + log(x)) + log(k) - 2 * (n + 1//2)^2 / x
        return exp(expnt)
    else
        error("x should be a positive real")
    end
end

function mass_texpon(z::Real)
    t = pg_t

    K = π^2 / 8 + z^2 / 2
    b = sqrt(inv(t)) * (t * z - 1)
    a = -sqrt(inv(t)) * (t * z + 1)

    x0 = log(K) + K * t
    xb = x0 - z + logcdf(Distributions.Normal(), b)
    xa = x0 + z + logcdf(Distributions.Normal(), a)

    qdivp = fourinvπ * (exp(xb) + exp(xa))

    return 1 / (1 + qdivp)
end

# Sample from a truncated inverse gaussian
function rand_truncated_inverse_gaussian(rng::AbstractRNG, z::Real)
    μ = inv(z)
    x = one(z) + pg_t
    if μ > pg_t
        d_exp = Exponential()
        while true
            E = rand(rng, d_exp)
            E′ = rand(rng, d_exp)
            while E^2 > 2E′ / pg_t
                E = rand(rng, d_exp)
                E′ = rand(rng, d_exp)
            end
            x = pg_t / (1 + E * pg_t)^2
            α = exp(-z^2 * x / 2)
            α >= rand(rng) && break
        end
    else
        while (x > pg_t)
            Y = randn(rng)^2
            μY = μ * Y
            x = μ + μ * μY / 2 - μ / 2 * sqrt(4 * μY + μY^2)
            if rand(rng) > μ / (μ + x)
                x = μ^2 / x
            end
            x > pg_t && break
        end
    end
    return x
end

# Sample from PG(1, z)
# Algorithm 1 from "Bayesian Inference for logistic models..." p. 26
function sample_pg1(rng::AbstractRNG, z::Real)
    # Change the parameter.
    z = abs(z) / 2

    # Now sample 0.25 * J^*(1, Z := Z/2).
    K = π^2 / 8 + z^2 / 2
    t = pg_t

    r = mass_texpon(z)

    while true
        if r > rand(rng) # sample from truncated exponential
            x = t + rand(rng, Exponential()) / K
        else # sample from truncated inverse Gaussian
            x = rand_truncated_inverse_gaussian(rng, z)
        end
        s = a(0, x)
        y = rand(rng) * s
        n = 0
        while true
            n = n + 1
            if isodd(n)
                s = s - a(n, x)
                y <= s && return x / 4
            else
                s = s + a(n, x)
                y > s && break
            end
        end
    end
end # Sample PG(1, c)

# Sample ω as the series of Gamma variables (truncated at 200)
function rand_gamma_sum(rng::AbstractRNG, d::PolyaGamma, e::Real)
    C = inv2π / π
    c = d.c
    w = (c * inv2π)^2
    d = Gamma(e, 1)
    return C * sum(1:200) do k
        rand(rng, d) / ((k - 0.5)^2 + w)
    end
end
