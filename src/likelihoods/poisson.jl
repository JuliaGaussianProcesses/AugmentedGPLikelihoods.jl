struct ScaledLogistic{T} <: AbstractLink
    λ::T
end

(l::ScaledLogistic)(f::Real) = l.λ * logistic(f)

const AugPoisson = PoissonLikelihood{<:ScaledLogistic}

function init_aux_variables(rng::AbstractRNG, ::AugPoisson, ndata::Int)
    return TupleVector((;ω=rand(rng, PolyaGamma(1, 0), ndata), n=rand(rng, Poisson(), ndata)))
end

function init_aux_posterior(::AugPoisson, n::Int)
    return [PolyaGammaPoisson(1, 0.0, 1.0) for _ in 1:n]
end

function aux_sample!(
    rng::AbstractRNG, Ω, lik::AugPoisson, y::AbstractVector{<:Int}, f::AbstractVector
)
    return map!(Ω, y, f) do y, f
        rand(rng, aux_full_conditional(lik, y, f))
    end
end

function aux_full_conditional(lik::AugPoisson, y::Int, f::Real)
    return PolyaGammaPoisson(y, abs(f), lik.invlink(f))
end

function aux_posterior!(
    qΩ, lik::AugPoisson, y::AbstractVector{<:Int}, qf::AbstractVector{<:Normal}
)
    λ = lik.invlink.λ
    map!(qΩ, qf, y) do f, y
        c = sqrt(second_moment(f))
        PolyaGammaPoisson(y, c, λ * approx_expected_logistic(-mean(f), c))
    end
    return qΩ
end

function auglik_potential(::AugPoisson, Ω, y::AbstractVector)
    return ((y .- Ω.n) / 2,) # use
end

function auglik_precision(::AugPoisson, Ω, ::AbstractVector)
    return (Ω.ω,)
end

function expected_auglik_potential(::AugPoisson, qΩ, y::AbstractVector)
    return ((y .- getfield.(qΩ, :λ)) / 2,) # Short cut to get the mean n
end

function expected_auglik_precision(::AugPoisson, qΩ, ::AbstractVector)
    return (mean(qΩ).ω,) # It is not possible to shorten this since n is needed to compute ω
end

function expected_auglik_potential_and_precision(::AugPoisson, qΩ, y::AbstractVector)
    θ = mean(qΩ)
    return ((y .- θ.n) / 2,), (θ.ω,)
end

function logtilt(lik::AugPoisson, Ω, y, f)
    logλ = log(lik.invlink.λ)
    return mapreduce(+, y, f, Ω) do y, f, (ω, n)
        return -(y + n) * logtwo +
               ((y - n) * f - abs2(f) * ω) / 2 +
               y * logλ +
               logfactorial(y)
    end
end

function aux_prior(lik::AugPoisson, y)
    return PolyaGammaPoisson.(y, 0, lik.invlink.λ)
end

function expected_logtilt(lik::AugPoisson, qΩ, y, qf)
    logλ = log(lik.invlink.λ)
    return mapreduce(+, y, qf, qΩ) do y, f, qΩ
        θ = mean(qΩ)
        m = mean(f)
        return -(y + n) * logtwo +
               (sign(y + θ.n) * m - (abs2(m) + var(f)) * θ.ω) / 2 +
               y * logλ +
               logfactorial(y)
    end
end
