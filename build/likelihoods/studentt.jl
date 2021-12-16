# TODO Move this to GPLikelihoods.jl
struct StudentTLikelihood{Tν<:Real,Tσ<:Real} <: AbstractLikelihood
    ν::Tν
    σ::Tσ
end

(lik::StudentTLikelihood)(f::Real) = LocationScale(f, lik.σ, TDist(lik.ν))

_α(lik::StudentTLikelihood) = (lik.ν + 1) / 2

function (lik::StudentTLikelihood)(f::AbstractVector{<:Real})
    return Produce(lik.(f))
end

function init_aux_variables(rng::AbstractRNG, ::StudentTLikelihood, n::Int)
    return TupleVector((; ω=rand(rng, InverseGamma(), n)))
end

function init_aux_posterior(T::DataType, lik::StudentTLikelihood, n::Int)
    α = _α(lik)
    return For(TupleVector(; β=zeros(T, n))) do φ
        InverseGamma(α, φ.β)
    end
end

function aux_sample!(
    rng::AbstractRNG, Ω, lik::StudentTLikelihood, ::AbstractVector, f::AbstractVector
)
    map!(Ω.ω, f) do f
        rand(rng, aux_full_conditional(lik, nothing, f))
    end
    return Ω
end

function aux_full_conditional(lik::StudentTLikelihood, y::Real, f::Real)
    return InverseGamma(_α(lik), (abs2(lik.σ) * lik.ν + abs(y - f)) / 2)
end

function aux_posterior!(
    qΩ, lik::StudentTLikelihood, y::AbstractVector, qf::AbstractVector{<:Normal}
)
    φ = qΩ.pars.β
    map!(φ.β, y, qf) do y, qf
        (abs2(lik.σ) * lik.ν + second_moment(qf - y)) / 2
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
    return (y ./ tvmean(qΩ).ω)
end

function expected_auglik_precision(::StudentTLikelihood, qΩ, ::AbstractVector)
    return (inv.(tvmean(qΩ).ω),)
end

function logtilt(::StudentTLikelihood, Ω, y, f)
    return mapreduce(+, y, f, Ω.ω) do y, f, ω
        -log(2) + (sign(y - 0.5) * f - abs2(f) * ω) / 2
    end
end

function aux_prior(lik::StudentTLikelihood, y)
    return For(length(y)) do _
        PolyaGamma(1, 0.0)
    end
end

function expected_logtilt(::StudentTLikelihood, qΩ, y, qf)
    return mapreduce(+, y, qf, marginals(qΩ)) do y, f, qω
        m = mean(f)
        -log(2) + (sign(y - 0.5) * m - (abs2(m) + var(f)) * mean(qω)) / 2
    end
end
