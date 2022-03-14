struct InvScaledLogistic{T} <: AbstractLink
    λ::T
end

(l::InvScaledLogistic)(f::Real) = inv(l.λ * logistic(f))

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
    return For(TupleVector(; c=zeros(T, n), λ=zeros(T, n))) do q
        PolyaGammaPoisson(1//2, q.c, q.λ)
    end
end

function aux_full_conditional(lik::AugHeteroGaussian, y::Real, (f,g)::AbstractVector{<:Real})
    return PolyaGammaPoisson(1//2, abs(g), inv(lik.invlink(g)) * abs2(f - y) / 2)
end

function aux_posterior!(
    qΩ, lik::AugHeteroGaussian, y::AbstractVector{<:Real}, qfg::AbstractVector{<:AbstractVector{<:Normal}}
)
    λ = lik.invlink.λ
    φ = qΩ.pars
    ψ = second_moment.(first.(qfg) .- y) / 2
    @. φ.c = sqrt(second_moment(first(qfg)))
    @. φ.λ = λ * approx_expected_logistic(-mean(last(qfg)), φ.c) * ψ
    return qΩ
end

function auglik_potential(lik::AugHeteroGaussian, Ω, y::AbstractVector, g::AbstractVector)
    return (y .* inv.(lik.invlink(g)) / 2, (1//2 - Ω.n) / 2)
end

function auglik_precision(::AugHeteroGaussian, Ω, ::AbstractVector)
    return (Ω.λσg, Ω.ω)
end

function expected_auglik_potential(lik::AugHeteroGaussian, qΩ, y::AbstractVector, qg::AbstractVector{<:Normal})
    λ = lik.invlink.λ
    c = qΩ.pars.c
    return (y .* λ .* approx_expected_logistic.(-mean.(qg), c) / 2, (1//2 .- tvmean(qΩ).n) / 2)
end

function expected_auglik_precision(::AugHeteroGaussian, qΩ, ::AbstractVector, qg::AbstractVector{<:Normal})
    λ = lik.invlink.λ
    c = qΩ.pars.c
    return (λ * approx_expected_logistic.(-mean.(qg), c), tvmean(qΩ).ω)
end

function expected_auglik_potential_and_precision(::AugHeteroGaussian, qΩ, y::AbstractVector)
    λ = lik.invlink.λ
    c = qΩ.pars.c
    θ = tvmean(qΩ)
    λσg = λ * approx_expected_logistic.(-mean.(qg), c)
    return ((y .* λσg / 2, (1 // 2 - θ.n) / 2), (λσg, θ.ω))
end

function logtilt(lik::AugHeteroGaussian, (ω, n)::Tuple{<:Real,<:Integer}, y::Real, (f,g)::AbstractVector{<:Real})
    return y * logλ(lik) - (y + n) * logtwo - logfactorial(y) +
           ((y - n) * f - abs2(f) * ω) / 2
end

function aux_prior(lik::AugHeteroGaussian, y::AbstractVector{<:Real})
    λ = lik.invlink.λ
    return For(y) do yᵢ
        PolyaGammaPoisson(yᵢ, 0, λ)
    end
end

aux_prior(lik::AugHeteroGaussian, y::Integer) = PolyaGammaPoisson(1 // 2, 0, lik.invlink.λ)

function expected_logtilt(lik::AugHeteroGaussian, qΩ, y, qfg::AbstractVector{<:AbstractVector{<:Normal}})
    logλ = log(lik.invlink.λ)
    return mapreduce(+, y, qf, @ignore_derivatives marginals(qΩ)) do yᵢ, qfᵢ, qω
        θ = ntmean(qω)
        m = mean(qfᵢ)
        return -(yᵢ + θ.n) * logtwo +
               ((yᵢ - θ.n) * m - (abs2(m) + var(qfᵢ)) * θ.ω) / 2 +
               yᵢ * logλ - logfactorial(yᵢ)
    end
end
