function logisticsoftmax(x::AbstractVector{<:Real})
    σs = logistic.(x)
    return σs / sum(σs)
end

struct LogisticSoftMaxLink{Tθ<:AbstractVector} <: AbstractLink
    logθ::Tθ
end

LogisticSoftMaxLink(nclass::Integer) = LogisticSoftMaxLink(zeros(nclass))

function _get_const(l::BijectiveSimplexLink{<:LogisticSoftMaxLink})
    return exp(last(l.link.logθ)) * logistic(0)
end

_sum_θ(l::LogisticSoftMaxLink) = exp(logsumexp(l.logθ))

function _sum_θ(l::BijectiveSimplexLink{<:LogisticSoftMaxLink})
    return _get_const(l) + exp(logsumexp(l.link.logθ[1:(end - 1)]))
end

function _scale_σf(l::LogisticSoftMaxLink, f::AbstractVector{<:Real})
    return exp.(l.logθ) .* logistic.(f)
end

function _scale_σf(
    l::BijectiveSimplexLink{<:LogisticSoftMaxLink}, f::AbstractVector{<:Real}
)
    return exp.(l.link.logθ[1:(end - 1)]) .* logistic.(f)
end

function (l::LogisticSoftMaxLink)(f::AbstractVector{<:Real})
    σs = exp.(l.logθ) .* logistic.(f)
    return σs ./ sum(σs)
end

# Augmentations are possible for both options
const BijectiveLogisticSoftMaxLikelihood = CategoricalLikelihood{
    <:BijectiveSimplexLink{<:LogisticSoftMaxLink}
}
const LogisticSoftMaxLikelihood = CategoricalLikelihood{<:LogisticSoftMaxLink}
const LogisticSoftMaxLikelihoods = Union{
    BijectiveLogisticSoftMaxLikelihood,LogisticSoftMaxLikelihood
}

nlatent(l::BijectiveLogisticSoftMaxLikelihood) = length(l.invlink.link.logθ) - 1
nlatent(l::LogisticSoftMaxLikelihood) = length(l.invlink.logθ)

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
    lik::LogisticSoftMaxLikelihoods, y::AbstractVector{<:Bool}, f::AbstractVector{<:Real}
)
    return PolyaGammaNegativeMultinomial(
        y, abs.(f), _scale_σf(lik.invlink, f) / _sum_θ(lik.invlink)
    )
end

function aux_posterior!(
    qΩ,
    lik::BijectiveLogisticSoftMaxLikelihood,
    y::AbstractVector{<:AbstractVector{<:Bool}},
    qf::AbstractVector{<:AbstractVector{<:Normal}},
)
    φ = only(qΩ.inds)
    for (i, φᵢ) in enumerate(φ)
        φᵢ.c .= sqrt.(second_moment.(qf[i]))
        φᵢ.y .= y[i]
        φᵢ.p .=
            approx_expected_logistic.(-mean.(qf[i]), φᵢ.c) /
            (_get_const(lik.invlink) + nlatent(lik))
    end
    return qΩ
end

function aux_posterior!(
    qΩ,
    lik::LogisticSoftMaxLikelihood,
    y::AbstractVector{<:AbstractVector},
    qf::AbstractVector{<:AbstractVector{<:Normal}},
)
    φ = only(qΩ.inds)
    for (i, φᵢ) in enumerate(φ)
        @. φᵢ.c = sqrt(second_moment(qf[i]))
        φᵢ.y .= y[i]
        φᵢ.p .= approx_expected_logistic.(-mean.(qf[i]), φᵢ.c) ./ nlatent(lik)
    end
    return qΩ
end

function auglik_potential(::LogisticSoftMaxLikelihoods, Ω, y::ArrayOfSimilarArrays{<:Bool})
    return nestedview(((flatview(y) - flatview(Ω.n)) / 2)')
    # We want to have a Tuple of vector of the same size as the number of classes
end

function auglik_precision(::LogisticSoftMaxLikelihoods, Ω, ::AbstractVector)
    return transpose_nested(Ω.ω)
end

function expected_auglik_potential(
    ::LogisticSoftMaxLikelihoods, qΩ, y::ArrayOfSimilarArrays{<:Bool}
)
    return nestedview(((flatview(y) - flatview(tvmean(qΩ).n)) / 2)')
end

function expected_auglik_precision(::LogisticSoftMaxLikelihoods, qΩ, ::AbstractVector)
    return transpose_nested(tvmean(qΩ).ω)
end

function expected_auglik_potential_and_precision(
    ::LogisticSoftMaxLikelihoods, qΩ, y::AbstractVector
)
    θ = tvmean(qΩ)
    return nestedview(((flatview(y) - flatview(θ.n)) / 2)'), transpose_nested(θ.ω)
end

function logtilt(
    ::LogisticSoftMaxLikelihoods,
    (ω, n)::Tuple{<:AbstractVector{<:Real},<:AbstractVector{<:Integer}},
    y::AbstractVector{<:Integer},
    f::AbstractVector{<:Real},
)
    return -sum(y + n) * logtwo + sum((y - n) .* f - abs2.(f) .* ω) / 2
end

function aux_prior(lik::LogisticSoftMaxLikelihoods, y::ArrayOfSimilarArrays{<:Bool})
    return For(y) do yᵢ
        aux_prior(lik, yᵢ)
    end
end

function aux_prior(lik::BijectiveLogisticSoftMaxLikelihood, y::AbstractVector{<:Integer})
    return PolyaGammaNegativeMultinomial(
        y, zeros(Int, length(y)), fill(inv(_sum_θ(lik.invlink)), nlatent(lik))
    )
end
function aux_prior(lik::LogisticSoftMaxLikelihood, y::AbstractVector{<:Integer})
    # There is no proper prior for this so we just pretend this will work
    return PolyaGammaNegativeMultinomial(
        y, zeros(Int, length(y)), fill(1 / nlatent(lik), nlatent(lik))
    )
end

function aux_kldivergence(::LogisticSoftMaxLikelihood, qΩ::For, pΩ::For)
    return error(
        "due to the ill-conditioning of the augmented prior of the `LogisticSoftMaxLink` (non-bijective), the kl-divergence cannot be computed, use the `BijectiveLogisticSoftMaxLink` instead.
        It might be possible to provide a more global workaround, but it is not listed as a priority for the package right now.",
    )
end

function expected_logtilt(
    ::BijectiveLogisticSoftMaxLikelihood, qω, y, qf::AbstractVector{<:Normal}
)
    θ = ntmean(qω)
    return -sum(y + θ.n) * logtwo +
           mapreduce(+, y, θ.n, θ.ω, mean.(qf), var.(qf)) do yᵢ, nᵢ, ωᵢ, mᵢ, vᵢ
        ((yᵢ - nᵢ) * mᵢ - (abs2(mᵢ) + vᵢ) * ωᵢ) / 2
    end
end
