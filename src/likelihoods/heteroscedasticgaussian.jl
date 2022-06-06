struct InvScaledLogistic{T} <: AbstractLink
    λ::T
end

(l::InvScaledLogistic)(f::Real) = inv(l.λ * logistic(f))
_λ(l::InvScaledLogistic) = l.invlink.λ

const AugHeteroGaussian = HeteroscedasticGaussianLikelihood{<:InvScaledLogistic}

nlatent(::HeteroscedasticGaussianLikelihood) = 2

aux_field(::AugHeteroGaussian, Ω::NamedTuple) = values(Ω)
aux_field(::AugHeteroGaussian, Ω::TupleVector) = zip(Ω.ω, Ω.n)

function init_aux_variables(rng::AbstractRNG, ::AugHeteroGaussian, ndata::Int)
    return TupleVector((;
        ω=rand(rng, PolyaGamma(1, 0), ndata), n=rand(rng, Poisson(), ndata)
    ))
end

function init_aux_posterior(T::DataType, ::AugHeteroGaussian, n::Int)
    return For(TupleVector(; c=zeros(T, n), λ=zeros(T, n), ψ=zeros(T, n))) do q
        PolyaGammaPoisson(1//2, q.c, q.λ)
    end
end

function aux_full_conditional(
    lik::AugHeteroGaussian, y::Real, (f, g)::AbstractVector{<:Real}
)
    return PolyaGammaPoisson(1//2, abs(g), inv(lik.invlink(-g)) * abs2(f - y) / 2)
end

function aux_posterior!(
    qΩ,
    lik::AugHeteroGaussian,
    y::AbstractVector{<:Real},
    qfg::AbstractVector{<:AbstractVector{<:Normal}},
)
    λ = _λ(lik)
    φ = only(qΩ.inds)
    @. φ.ψ = second_moment(first(qfg) - y) / 2
    @. φ.c = sqrt(second_moment(last(qfg)))
    @. φ.λ = λ * approx_expected_logistic(-mean(last(qfg)), φ.c) * φ.ψ
    return qΩ
end

function auglik_potential(
    lik::AugHeteroGaussian, Ω, y::AbstractVector, g::AbstractVector{<:Real}
)
    return (y .* inv.(lik.invlink.(g)), (1//2 .- Ω.n) / 2)
end

function auglik_precision(
    lik::AugHeteroGaussian, Ω, ::AbstractVector, g::AbstractVector{<:Real}
)
    return (inv.(lik.invlink.(g)), Ω.ω)
end

function expected_auglik_potential(
    lik::AugHeteroGaussian, qΩ, y::AbstractVector, qg::AbstractVector{<:Normal}
)
    λ = _λ(lik)
    c = only(qΩ.inds).c
    return (
        y .* λ .* (1 .- approx_expected_logistic.(-mean.(qg), c)),
        (1//2 .- tvmean(qΩ).n) / 2,
    )
end

function expected_auglik_precision(
    ::AugHeteroGaussian, qΩ, ::AbstractVector, qg::AbstractVector{<:Normal}
)
    λ = _λ(lik)
    c = only(qΩ.pars).c
    return (λ * (1 .- approx_expected_logistic.(-mean.(qg), c)), tvmean(qΩ).ω)
end

function expected_auglik_potential_and_precision(
    lik::AugHeteroGaussian, qΩ, y::AbstractVector, qg::AbstractVector{<:Normal}
)
    λ = _λ(lik)
    c = only(qΩ.pars).c
    θ = tvmean(qΩ)
    λσg = λ * approx_expected_logistic.(-mean.(qg), c)
    return ((y .* λσg / 2, (1//2 .- θ.n) / 2), (λσg, θ.ω))
end

function aug_loglik(
    lik::AugHeteroGaussian,
    (ω, n)::Tuple{<:Real,<:Integer},
    y::Real,
    (f, g)::AbstractVector{<:Real},
)
    return -(1//2 + n) * logtwo +
           ((1//2 - n) * g - abs2(g) * ω) / 2 +
           logpdf(PolyaGamma(1//2 + n, 0), ω) +
           logpdf(Poisson(_λ(lik) / 2 * abs2(y - f)), n)
end

function expected_aug_loglik(
    lik::AugHeteroGaussian, qΩ, y, qfg::AbstractVector{<:AbstractVector{<:Normal}}
)
    λ = _λ(lik)
    return mapreduce(+, y, qfg, @ignore_derivatives marginals(qΩ)) do yᵢ, qfgᵢ, qωᵢ
        θ = ntmean(qωᵢ)
        qg = last(qfgᵢ)
        qf = first(qfgᵢ)
        g = mean(qg)
        return -(1//2 + θ.n) * logtwo +
               ((1//2 - θ.n) * g - (abs2(g) + var(first(qg))) * θ.ω) / 2 +
               kldivergence(qωᵢ, PolyaGammaPoisson(1//2, 0, λ / 2 * (abs2(yᵢ - mean(qf)) + var(qf))))
    end
end
