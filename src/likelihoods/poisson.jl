struct ScaledLogistic{T} <: AbstractLink
    λ::T
end

(l::ScaledLogistic)(f::Real) = l.λ * logistic(f)

const AugPoisson = PoissonLikelihood{<:ScaledLogistic}

function init_aux_variables(rng::AbstractRNG, ::AugPoisson, n::Int)
    return (; ωn=[(; ω=rand(rng, PolyaGamma(1, 0)), n=rand(rng, Poisson())) for _ in 1:n])
end

function init_aux_posterior(::AugPoisson, n::Int)
    return (; ωn=[PolyaGammaPoisson(1, 0.0, 1.0) for _ in 1:n])
end

function aux_sample!(
    rng::AbstractRNG, Ω, lik::AugPoisson, y::AbstractVector{<:Int}, f::AbstractVector
)
    map!(Ω.ωn, f, y) do f, y
        rand(rng, PolyaGammaPoisson(y, abs(f), lik.invlink(f)))
    end
    return Ω
end

function aux_posterior!(
    qΩ, lik::AugPoisson, y::AbstractVector{<:Int}, qf::AbstractVector{<:Normal}
)
    λ = lik.invlink.λ
    map!(qΩ.ωn, qf, y) do f, y
        c = sqrt(second_moment(f))
        PolyaGammaPoisson(y, c, λ * approx_expected_logistic(-mean(f), c))
    end
    return qΩ
end

function auglik_potential(::AugPoisson, Ω, y::AbstractVector)
    return ((y .- last.(Ω.ωn)) / 2,) # use
end

function auglik_precision(::AugPoisson, Ω, ::AbstractVector)
    return (first.(Ω.ωn),)
end

function expected_auglik_potential(::AugPoisson, Ω, y::AbstractVector)
    return ((y .- last.(mean.(Ω.ωn))) / 2,)
end

function expected_auglik_precision(::AugPoisson, Ω, ::AbstractVector)
    return (first.(mean.(Ω.ωn)),)
end

function expected_auglik_potential_and_precision(::AugPoisson, Ω, y::AbstractVector)
    θ = mean.(qΩ.ωn)
    return (first.(θ),), (last.(θ),)
end

function logtilt(lik::AugPoisson, Ω, y, f)
    logλ = log(lik.invlink.λ)
    return mapreduce(+, y, f, Ω.ωn) do y, f, (ω, n)
        return -(y + n) * logtwo +
               ((y - n) * f - abs2(f) * ω) / 2 +
               y * logλ +
               logfactorial(y)
    end
end

function aux_prior(lik::AugPoisson, y)
    return (; ωn=PolyaGammaPoisson.(y, 0, lik.invlink.λ))
end

function expected_logtilt(lik::AugPoisson, qΩ, y, qf)
    logλ = log(lik.invlink.λ)
    return mapreduce(+, y, qf, qΩ.ωn) do y, f, ωn
        (ω, n) = mean(ωn)
        m = mean(f)
        return -(y + n) * logtwo +
               (sign(y + n) * m - (abs2(m) + var(f)) * ω) / 2 +
               y * logλ +
               logfactorial(y)
    end
end
