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
        ω=nestedview(rand(rng, PolyaGamma(1, 0), nlatent(l), ndata)),
        n=nestedview(rand(rng, Poisson(), nlatent(l), ndata)),
    )
end

function init_aux_posterior(T::DataType, lik::LogisticSoftMaxLikelihoods, n::Int)
    nl = nlatent(lik) # This needs # https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl/pull/68
    return For(
        TupleVector(;
            y=nestedview(falses(nl, n)),
            c=nestedview(zeros(T, nl, n)),
            p=nestedview(zeros(T, nl, n)),
        ),
    ) do q
        PolyaGammaNegativeMultinomial(q.y, q.c, q.p)
    end
end

function aux_full_conditional(
    lik::BijectiveLogisticSoftMaxLikelihood, y::AbstractVector{<:Bool}, f::AbstractVector{<:Real}
)
    return PolyaGammaNegativeMultinomial(y, abs.(f), lik.invlink(-f)[1:end-1])
end

function aux_full_conditional(
    lik::LogisticSoftMaxLikelihood, y::AbstractVector{<:Bool}, f::AbstractVector{<:Real}
)
    return PolyaGammaNegativeMultinomial(y, abs(f)) # TODO
end

function aux_posterior!(
    qΩ,
    ::BijectiveLogisticSoftMaxLikelihood,
    y::AbstractVector{<:AbstractVector{<:Bool}},
    qf::AbstractVector{<:AbstractVector{<:Normal}},
)
    φ = qΩ.pars
    for (i, φᵢ) in enumerate(φ)
        @. φᵢ.c = sqrt(second_moment(qf[i]))
        @. φᵢ.y = y[i]
        φᵢ.p .= approx_expected_logisticsoftmax(-mean.(qf[i]), φᵢ.c)
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
    ::BijectiveLogisticSoftMaxLikelihood, Ω, y::ArrayOfSimilarArrays{<:Bool}
)
    return nestedview(((flatview(y) - flatview(Ω.n)) / 2)')
    # We want to have a Tuple of vector of the same size as the number of classes
end

function auglik_precision(::BijectiveLogisticSoftMaxLikelihood, Ω, ::AbstractVector)
    return transpose_nested(Ω.ω)
end

function expected_auglik_potential(::BijectiveLogisticSoftMaxLikelihood, qΩ, y::ArrayOfSimilarArrays{<:Bool})
    return nestedview(((flatview(y) - flatview(tvmean(qΩ).n)) / 2)')
end

function expected_auglik_precision(::BijectiveLogisticSoftMaxLikelihood, qΩ, ::AbstractVector)
    return transpose_nested(tvmean(qΩ).ω)
end

function expected_auglik_potential_and_precision(::BijectiveLogisticSoftMaxLikelihood, qΩ, y::AbstractVector)
    θ = tvmean(qΩ)
    return nestedview(((flatview(y) - flatview(θ.n)) / 2)'), transpose_nested(θ.ω)
end

function logtilt(lik::BijectiveLogisticSoftMaxLikelihood, (ω, n)::Tuple{<:Real,<:Integer}, y::Integer, f::Real)
    # TODO
    # return logistic(0) - y * logλ(lik) - (y + n) * logtwo - logfactorial(y) +
        #    ((y - n) * f - abs2(f) * ω) / 2
end

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
