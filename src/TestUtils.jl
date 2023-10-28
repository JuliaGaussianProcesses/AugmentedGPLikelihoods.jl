module TestUtils
using AugmentedGPLikelihoods
const AGPL = AugmentedGPLikelihoods
using AugmentedGPLikelihoods.SpecialDistributions
using ArraysOfArrays
using Distributions
using GPLikelihoods: AbstractLikelihood
using LinearAlgebra
using MeasureBase: MeasureBase, logdensity_def, marginals
using MeasureTheory: For
using SplitApplyCombine: invert
using Random
using Test
using TupleVectors

export flatten_params, unflatten_params
# Test API for augmented likelihood
remove_ntdist_wrapper(d::NTDist) = d.d
remove_ntdist_wrapper(d) = d

can_split(::AbstractLikelihood) = true


function flatten_params(φ, n_l)
    if n_l == 1
        return vcat(values(φ)...)
    else
        return vcat(flatview.(values(φ))...)
    end
end

function unflatten_params(φ, n_l, n)
    if n_l == 1
        return collect(Iterators.partition(φ, n))
    else
        parts = Iterators.partition(1:size(φ, 1), n_l)
        return collect(nestedview(φ[part, :]) for part in parts)
    end
end

invert_if_array_of_arrays(f::AbstractVector) = f
invert_if_array_of_arrays(f::AbstractVector{<:AbstractVector}) = invert(f)

function gen_y(rng::AbstractRNG, lik, f)
    if nlatent(lik) == 1
        return rand(rng, lik(f))
    else
        return rand(rng, lik(invert(f)))
    end
end

function gen_y(rng::AbstractRNG, lik::CategoricalLikelihood, f)
    y = rand(rng, lik(invert(f)))
    return nestedview((sort(unique(y)) .== y')[1:nlatent(lik), :])
end

function test_auglik(
    lik::AbstractLikelihood;
    n=10,
    f=randn(n),
    qf=Normal.(randn(n), 1.0),
    rng::AbstractRNG=Random.GLOBAL_RNG,
)
    y = gen_y(rng, lik, f)
    ft = invert_if_array_of_arrays(f)
    qft = invert_if_array_of_arrays(qf)
    nf = nlatent(lik)
    # Testing sampling
    @testset "Sampling" begin
        Ω = init_aux_variables(lik, n)
        @test Ω isa TupleVector
        @test first(Ω) isa NamedTuple
        @test length(Ω) == n
        Ω = aux_sample!(rng, Ω, lik, y, ft)
        @test Ω isa TupleVector
        new_Ω = aux_sample(rng, lik, y, ft)
        @test new_Ω isa TupleVector
        @test length(Ω) == n

        βs = auglik_potential(lik, Ω, y, ft)
        γs = auglik_precision(lik, Ω, y, ft)
        β2, γ2 = auglik_potential_and_precision(lik, Ω, y, ft)
        @test length(γs) == length(βs) == nf # Check that there are n latent vectors
        @test first(βs) isa AbstractVector
        @test first(γs) isa AbstractVector
        @test all(map(≈, βs, β2))
        @test all(map(≈, γs, γ2))
        @test all(x -> all(>=(0), x), γs) # Check that the variance is positive

        # When we have an explicit version of p(Ω) and of the tilt
        if can_split(lik)
            sumlogtilt = logtilt(lik, Ω, y, ft)
            @test sumlogtilt isa Real
            @test logtilt(lik, AGPL.aux_field(lik, first(Ω)), first(y), first(ft)) isa Real
            sumlogtilt_alt = mapreduce(+, AGPL.aux_field(lik, Ω), y, ft) do ωᵢ, yᵢ, fᵢ
                logtilt(lik, ωᵢ, yᵢ, fᵢ)
            end
            @test sumlogtilt ≈ sumlogtilt_alt
            pΩ = aux_prior(lik, y)
            @test logdensity_def(pΩ, Ω) isa Real
            pω = aux_prior(lik, first(y)) # Scalar version
            @test pω == remove_ntdist_wrapper(first(marginals(pΩ)))
        end

        @test aug_loglik(lik, Ω, y, ft) isa Real
        # Test that the full conditional is correct
        @testset "Full conditional Ω" begin
            pcondΩ = aux_full_conditional(lik, y, ft) # Compute the full conditional of Ω
            Ω₁ = tvrand(rng, pcondΩ) # Sample a set of aux. variables
            Ω₂ = tvrand(rng, pcondΩ) # Sample another set of aux. variables
            # We compute p(f, y) by doing C = p(f,y) = p(y|Ω,f)p(Ω)/p(Ω|y,f)
            # This should be the same no matter what Ω is
            logC₁ = aug_loglik(lik, Ω₁, y, ft) - logdensity_def(pcondΩ, Ω₁)
            logC₂ = aug_loglik(lik, Ω₂, y, ft) - logdensity_def(pcondΩ, Ω₂)
            @test logC₁ ≈ logC₂ atol = 1e-5
        end

        @testset "Full conditional f" begin
            pcondΩ = aux_full_conditional(lik, y, ft) # Compute the full conditional of Ω
            Ω = tvrand(rng, pcondΩ) # Sample a set of aux. variables
            K = (x -> x * x')(rand(n, n)) # Prior Covariance matrix
            if nlatent(lik) == 1
                S = inv(Symmetric(inv(K) + Diagonal(only(auglik_precision(lik, Ω, y)))))
                m = S * (only(auglik_potential(lik, Ω, y)))
                qF = MvNormal(m, S)
                pF = MvNormal(K)
                f₁ = rand(rng, qF)
                f₂ = rand(rng, qF)
                logC₁ = logtilt(lik, Ω, y, f₁) + logpdf(pF, f₁) - logpdf(qF, f₁)
                logC₂ = logtilt(lik, Ω, y, f₂) + logpdf(pF, f₂) - logpdf(qF, f₂)
                @test logC₁ ≈ logC₂ atol = 1e-5
            else
                S = inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(auglik_precision(lik, Ω, y, ft))))
                m = S .* auglik_potential(lik, Ω, y, ft)
                qF = MvNormal.(m, S)
                pF = MvNormal(K)
                f₁ = rand.(rng, qF)
                f₂ = rand.(rng, qF)
                @show f₁, invert(f₁)
                logC₁ =
                    aug_loglik(lik, Ω, y, invert(f₁)) + sum(logpdf.(Ref(pF), f₁)) -
                    sum(logpdf.(qF, f₁))
                logC₂ =
                    aug_loglik(lik, Ω, y, invert(f₂)) + sum(logpdf.(Ref(pF), f₂)) -
                    sum(logpdf.(qF, f₂))
                @test logC₁ ≈ logC₂ atol = 1e-5
            end
        end
    end

    #Testing variational inference
    @testset "Variational Inference" begin
        qΩ = init_aux_posterior(lik, n)
        @test qΩ isa For
        @test length(qΩ) == n
        qΩ = aux_posterior!(qΩ, lik, y, qft)
        @test qΩ isa For
        new_qΩ = aux_posterior(lik, y, qft)
        @test new_qΩ isa For
        @test length(new_qΩ) == n

        βs = expected_auglik_potential(lik, qΩ, y, qft)
        γs = expected_auglik_precision(lik, qΩ, y, qft)
        β2, γ2 = expected_auglik_potential_and_precision(lik, qΩ, y, qft)
        @test length(γs) == length(βs) == nf # Check that there are n latent vectors
        @test first(βs) isa AbstractVector
        @test first(γs) isa AbstractVector
        @test all(map(≈, βs, β2))
        @test all(map(≈, γs, γ2))

        @test all(x -> all(>=(0), x), γs) # Check that the variance is positive

        # TODO test that aux_posterior parameters return the minimizing
        φ = TupleVectors.unwrap(only(aux_posterior(lik, y, qft).inds)) # TupleVector
        φ_opt = flatten_params(φ, nlatent(lik))
        s = keys(φ)
        n_var = length(s)
        function loss(φ)
            q = For(qΩ.f, TupleVector(NamedTuple{s}(unflatten_params(φ, nlatent(lik), n))))
            return -expected_aug_loglik(lik, q, y, qft)
        end
        ϵ = 1e-2
        # Test that by perturbing the value in random directions, the loss does not decrease
        for i in CartesianIndices(φ_opt)
            Δ = if nlatent(lik) == 1
                (lik isa Union{NegativeBinomialLikelihood,PoissonLikelihood} && i[1] <= n) && continue # We do not want to vary y
                zeros(n_var * n)
            else
                (lik isa CategoricalLikelihood && i[1] <= nlatent(lik)) && continue
                zeros(n_var * nlatent(lik), n)
            end
            Δ[i] = ϵ # We try one element at a time
            # @test loss(φ_opt) <= loss(φ_opt + Δ)
            # @test loss(φ_opt) <= loss(φ_opt - Δ)
        end
        # Optim.optimize(loss, φ_opt)
        # values of the ELBO
        if can_split(lik)
            pΩ = aux_prior(lik, y)
            @test pΩ isa For
            @test kldivergence(first(marginals(qΩ)), first(marginals(pΩ))) isa Real
            @test expected_logtilt(lik, qΩ, y, qft) isa Real
            @test aux_kldivergence(lik, qΩ, pΩ) isa Real
        end
    end
end

end
