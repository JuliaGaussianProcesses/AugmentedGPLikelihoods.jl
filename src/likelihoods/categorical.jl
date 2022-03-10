function logisticsoftmax(x::AbstractVector{<:Real})
    σs = logistic.(x)
    return σs / sum(σs)
end

const LogisticSoftMaxLink = Link{typeof(logisticsoftmax)}

# Augmentations are possible for both options
const BijectiveLogisticSoftMaxLikelihood = CategoricalLikelihood{
    <:BijectiveSimplexLink{<:Union{LogisticSoftMaxLink,typeof(logisticsoftmax)}}
}
const LogisticSoftMaxLikelihood = CategoricalLikelihood{<:Union{LogisticSoftMaxLink,typeof(logisticsoftmax)}}
const LogisticSoftMaxLikelihoods = Union{
    BijectiveLogisticSoftMaxLikelihood,LogisticSoftMaxLikelihood
}

aux_field(::LogisticSoftMaxLikelihoods, Ω::NamedTuple) = values(Ω)
aux_field(::LogisticSoftMaxLikelihoods, Ω::TupleVector) = zip(Ω.ω, Ω.n)

function init_aux_variables(rng::AbstractRNG, l::LogisticSoftMaxLikelihoods, ndata::Int)
    return TupleVector(;
        ω=[rand(rng, PolyaGamma(1, 0), nlatent(l)) for _ in 1:ndata],
        n=[rand(rng, Poisson(), nlatent(l)) for _ in 1:ndata],
    )
end

function init_aux_posterior(T::DataType, l::LogisticSoftMaxLikelihoods, n::Int)
    nclasses = l.nclasses
    return For(
        TupleVector(;
            y=[falses(nclasses) for _ in 1:n],
            c=[zeros(T, nclasses) for _ in 1:n],
            p=[zeros(T, nclasses) for _ in 1:n],
        ),
    ) do q
        PolyaGammaNegativeMultinomial(q.y, q.c, q.p)
    end
end

function aux_full_conditional(
    lik::BijectiveLogisticSoftMaxLikelihood, y::AbstractVector, f::AbstractVector{<:Real}
)
    return PolyaGammaNegativeMultinomial(y, abs(f), lik.invlink(-f))
end

function aux_full_conditional(
    lik::LogisticSoftMaxLikelihood, y::AbstractVector, f::AbstractVector{<:Real}
)
    return PolyaGammaNegativeMultinomial(y, abs(f)) # TODO
end

function aux_posterior!(
    qΩ,
    ::BijectiveLogisticSoftMaxLikelihood,
    y::AbstractVector{<:AbstractVector},
    qf::AbstractVector{<:AbstractVector{<:Normal}},
)
    φ = qΩ.pars
    for (i, φᵢ) in enumerate(φ)
        @. φᵢ.c = sqrt(second_moment(qf[i]))
        @. φᵢ.y = y[i]
        @. φᵢ.p = approx_expected_logisticsoftmax(-mean.(qf[i]), φᵢ.c)
    end
    return qΩ
end

# function aux_posterior!(
#     qΩ,
#     lik::LogisticSoftMaxLikelihood,
#     y::AbstractVector{<:AbstractVector},
#     qf::AbstractVector{<:AbstractVector{<:Normal}},
# )
#     λ = lik.invlink.λ
#     φ = qΩ.pars
#     @. φ.c = sqrt(second_moment(qf))
#     @. φ.y = y
#     @. φ.λ = λ * approx_expected_logistic(-mean(qf), φ.c)
#     return qΩ
# end

function auglik_potential(
    ::BijectiveLogisticSoftMaxLikelihood, Ω, y::AbstractVector{<:AbstractVector}
)
    return (y .- Ω.n) ./ 2 # TODO make sure that this is the right dimensionality
    # We want to have a Tuple of vector of the same size as the number of classes
end

function auglik_precision(::BijectiveLogisticSoftMaxLikelihood, Ω, ::AbstractVector)
    return Ω.ω
end

# function expected_auglik_potential(::AugPoisson, qΩ, y::AbstractVector)
#     return ((y .- tvmean(qΩ).n) / 2,) # Short cut to get the mean n
# end

# function expected_auglik_precision(::AugPoisson, qΩ, ::AbstractVector)
#     return (tvmean(qΩ).ω,) # It is not possible to shorten this since n is needed to compute ω
# end

# function expected_auglik_potential_and_precision(::AugPoisson, qΩ, y::AbstractVector)
#     θ = tvmean(qΩ)
#     return ((y .- θ.n) / 2,), (θ.ω,)
# end

# function logtilt(lik::AugPoisson, (ω, n)::Tuple{<:Real,<:Integer}, y::Integer, f::Real)
#     return y * logλ(lik) - (y + n) * logtwo - logfactorial(y) +
#            ((y - n) * f - abs2(f) * ω) / 2
# end

function aux_prior(lik::BijectiveLogisticSoftMaxLikelihood, y::AbstractVector)
    return PolyaGammaNegativeMultinomial(
        y, zeros(Int, length(y)), repeat(logistic(0) + nlatent(lik), nlatent(lik))
    )
end
function aux_prior(lik::LogisticSoftMaxLikelihood, y::AbstractVector)
    return PolyaGammaNegativeMultinomial(y, 0) # TODO
end

# function expected_logtilt(lik::AugPoisson, qΩ, y, qf::AbstractVector{<:Normal})
#     logλ = log(lik.invlink.λ)
#     return mapreduce(+, y, qf, @ignore_derivatives marginals(qΩ)) do yᵢ, qfᵢ, qω
#         θ = ntmean(qω)
#         m = mean(qfᵢ)
#         return -(yᵢ + θ.n) * logtwo +
#                ((yᵢ - θ.n) * m - (abs2(m) + var(qfᵢ)) * θ.ω) / 2 +
#                yᵢ * logλ - logfactorial(yᵢ)
#     end
# end
