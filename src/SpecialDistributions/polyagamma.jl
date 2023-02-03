using Statistics

const PG_T = 0.64
const π²_8 = π^2 / 8

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
        return iszero(x) ? zero(x) : -Inf # The limit of PG when b->0  
    # is the delta dirac at 0.
    else
        iszero(x) && -Inf # The limit to p(x) for x-> 0 is 0.
        ext = logtilt(x, b, c) + (b - 1) * logtwo - (log2π + 3 * log(x)) / 2

        if x < 1e-2
            return ext + calc_log_series(x, b, 100)
        else
            sum_series = calc_series(x, b, 100)
            return ext + log(max(sum_series, floatmin(sum_series)))
        end
    end
end

function calc_log_series(x::Real, b::Real, max_half_n::Integer)
    max_n = 2*max_half_n

    log_series_m_prods = cumsum(log.(1 .+ ((b - 1) ./ (1:max_n))))

    log_out = map(0:2:max_n) do n
        Rn = 2n + b
        log_exp_out = Rn^2 / -8x

        log_c_nb = log(n + b) - log(n + 1) + log(2 / Rn + 1)
        log_inner = log1mexp(log_c_nb + ((Rn + 1) / -2x))

        log_series_prod = if n == 0 0.0 else log_series_m_prods[n] end

        log_series_prod + log(Rn) + log_exp_out + log_inner
    end

    logsumexp(log_out)
end

function calc_series(x::Real, b::Real, max_half_n::Integer)
    max_n = 2*max_half_n

    series_m_prods = cumprod(1 .+ ((b - 1) ./ (1:max_n)))

    sum(0:2:max_n) do n
        Rn = 2n + b
        exp_out = exp(Rn^2 / -8x)

        c_nb = ((n + b) / (n + 1)) * (2 / Rn + 1)
        inner = 1 - c_nb * exp((Rn + 1) / -2x)

        series_prod = if n == 0 1.0 else series_m_prods[n] end

        series_prod * Rn * exp_out * inner
    end
end


# calc_series(x, b, 10) * gamma(b) ≈ _series_term_naive.(x, b, 0:10)

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
    return b * logcosh(c / 2) - abs2(c) * ω / 2
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
function draw_sum(rng::AbstractRNG, d::PolyaGamma{<:Integer})
    b, c = Distributions.params(d)
    return sum(1:b) do _
        sample_pg1(rng, c)
    end
end

## Sampling when `b` is a Real (and might need to be truncated)
function draw_sum(rng::AbstractRNG, d::PolyaGamma{<:Real})
    b, c = Distributions.params(d)
    if b < 1
        return rand_gamma_sum(rng, d, b)
    end
    trunc_b = floor(Int, b)
    trunc_term = sum(1:trunc_b) do _
        sample_pg1(rng, c)
    end

    res_b = b - trunc_b
    if iszero(res_b)
        return trunc_term
    else
        res_term = rand_gamma_sum(rng, d, res_b)
        return trunc_term + res_term
    end
end

# Sample ω as the series of Gamma variables (truncated at 200)
function rand_gamma_sum(rng::AbstractRNG, d::PolyaGamma, e::Real)
    inv2π² = inv2π * invπ
    w = (d.c * inv2π)^2
    ga = Gamma(e, 1)
    return inv2π² * sum(1:200) do k
        rand(rng, ga) / ((k - 0.5)^2 + w)
    end
end

## Utility functions
function a(n::Int, x::Real)
    k = (n + 1//2) * π
    if x > PG_T
        return k * exp(-k^2 * x / 2)
    elseif x > 0
        expnt = -3 / 2 * (log(halfπ) + log(x)) - 2 * (n + 1//2)^2 / x
        return k * exp(expnt)
    else
        throw(DomainError(x, "x should be a positive real"))
    end
end

function mass_texpon(z::Real, K::Real)
    t = PG_T

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
    x = one(z) + PG_T
    if μ > PG_T
        α = zero(μ)
        while α < rand(rng)
            E = randexp(rng)
            E′ = randexp(rng)
            while E^2 > (2E′ / PG_T)
                E = randexp(rng)
                E′ = randexp(rng)
            end
            x = PG_T / abs2(1 + E * PG_T)
            α = exp(-z^2 * x / 2)
        end
    else
        while (x > PG_T)
            y = randn(rng)^2
            μy = μ * y
            x = μ + μ * μy / 2 - μ * sqrt(4 * μy + abs2(μy)) / 2
            if μ / (μ + x) < rand(rng)
                x = μ^2 / x
            end
        end
    end
    return x
end

# Sample from PG(1, z)
# Algorithm 1 from "Bayesian Inference for logistic models..." p. 26
function sample_pg1(rng::AbstractRNG, c::Real)
    # Change the parameter.
    z = abs(c) / 2

    # Now sample 0.25 * J^*(1, Z := Z/2).
    if iszero(z) # We specialize on c = 0
        r = 0.5776972428360435
        K = π²_8
    else
        K = π²_8 + z^2 / 2
        r = mass_texpon(z, K)
    end
    while true
        if r > rand(rng) # sample from truncated exponential
            x = PG_T + randexp(rng) / K
        else # sample from truncated inverse Gaussian
            x = rand_truncated_inverse_gaussian(rng, z)
        end
        s = a(0, x)
        y = rand(rng) * s
        n = 0
        while true
            n += 1
            if isodd(n)
                s -= a(n, x)
                y <= s && return x / 4
            else
                s += a(n, x)
                y > s && break
            end
        end
    end
end # Sample PG(1, c)
