struct ScaledLogistic{T} <: AbstractLink
    λ::T
end

(l::ScaledLogistic)(f::Real) = l.λ * logistic(f)

const AugPoisson = PoissonLikelihood{<:ScaledLogistic}

function init_aux_variables(rng::AbstractRNG, ::AugPoisson, ndata::Int)
    return TupleVector((;
        ω=rand(rng, PolyaGamma(1, 0), ndata), n=rand(rng, Poisson(), ndata)
    ))
end

function init_aux_posterior(T::DataType, ::AugPoisson, n::Int)
    return For(TupleVector(; y=zeros(Int, n), c=zeros(T, n), λ=zeros(T, n))) do q
        PolyaGammaPoisson(q.y, q.c, q.λ)
    end
end

function aux_full_conditional(lik::AugPoisson, y::Int, f::Real)
    return PolyaGammaPoisson(y, abs(f), lik.invlink(-f))
end

function aux_posterior!(
    qΩ, lik::AugPoisson, y::AbstractVector{<:Int}, qf::AbstractVector{<:Normal}
)
    λ = lik.invlink.λ
    φ = qΩ.pars
    @. φ.c = sqrt(second_moment(qf))
    @. φ.y = y
    @. φ.λ = λ * approx_expected_logistic(-mean(qf), φ.c)
    return qΩ
end

function auglik_potential(::AugPoisson, Ω, y::AbstractVector)
    return ((y .- Ω.n) / 2,) # use
end

function auglik_precision(::AugPoisson, Ω, ::AbstractVector)
    return (Ω.ω,)
end

function expected_auglik_potential(::AugPoisson, qΩ, y::AbstractVector)
    return ((y .- tvmean(qΩ).n) / 2,) # Short cut to get the mean n
end

function expected_auglik_precision(::AugPoisson, qΩ, ::AbstractVector)
    return (tvmean(qΩ).ω,) # It is not possible to shorten this since n is needed to compute ω
end

function expected_auglik_potential_and_precision(::AugPoisson, qΩ, y::AbstractVector)
    θ = tvmean(qΩ)
    return ((y .- θ.n) / 2,), (θ.ω,)
end

function logtilt(lik::AugPoisson, Ω, y, f)
    logλ = log(lik.invlink.λ)
    return mapreduce(+, y, f, Ω) do yᵢ, fᵢ, (ω, n)
        return yᵢ * logλ -
                (yᵢ + n) * logtwo -
                logfactorial(yᵢ)
               ((yᵢ - n) * fᵢ - abs2(fᵢ) * ω) / 2
    end
end

function aux_prior(lik::AugPoisson, y::AbstractVector{<:Int})
    λ = lik.invlink.λ
    return For(y) do yᵢ
        PolyaGammaPoisson(yᵢ, 0, λ)
    end
end

function expected_logtilt(lik::AugPoisson, qΩ, y, qf)
    logλ = log(lik.invlink.λ)
    return mapreduce(+, y, qf, marginals(qΩ)) do yᵢ, fᵢ, qω
        θ = ntmean(qω)
        m = mean(fᵢ)
        return -(yᵢ + θ.n) * logtwo +
               ((yᵢ - θ.n) * m - (abs2(m) + var(fᵢ)) * θ.ω) / 2 +
               yᵢ * logλ -
               logfactorial(yᵢ)
    end
end
