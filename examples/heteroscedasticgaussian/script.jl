# Heteroscedastic Gaussian Regression

# We load all the necessary packages
using AbstractGPs
using ApproximateGPs
using ArraysOfArrays
using AugmentedGPLikelihoods
using Distributions
using LinearAlgebra
using SplitApplyCombine
# Plotting libraries
using Plots

# We create some random data (sorted for plotting reasons)
N = 100
x = collect(range(-10, 10; length=N))
lik = HeteroscedasticGaussianLikelihood(InvScaledLogistic(3.0))
SplitApplyCombine.invert(x::ArrayOfSimilarArrays) = nestedview(flatview(x)')

kernel = 5.0 * with_lengthscale(SqExponentialKernel(), 2.0)
gp_f = GP(kernel)
μ₀g = -3.0
gp_g = GP(μ₀g, kernel)
f = rand(gp_f(x, 1e-6))
g = rand(gp_g(x, 1e-6))

py = lik(invert([f, g]))  # invert turns [f, g] into [[f_1, g_1], ...]
y = rand(py);
# We plot the sampled data
plt = plot(x, mean(py); ribbon=sqrt.(var(py)), label="y|f,g", lw=2.0)
scatter!(plt, x, y; msw=0.0, label="y")
plt2 = plot(x, [f, g]; label=["f" "g"], lw=2.0)
plot(plt, plt2)
# ## CAVI Updates
# We write our CAVI algorithmm
function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(copy(m), S)))
end;

# We can also optimize the likelihood parameter λ
function opt_lik(
    lik::HeteroscedasticGaussianLikelihood,
    (qf, qg)::AbstractVector{<:AbstractVector{<:Normal}},
    y::AbstractVector{<:Real},
)
    ψ = AugmentedGPLikelihoods.second_moment.(qf .- y) / 2
    c = sqrt.(AugmentedGPLikelihoods.second_moment.(qg))
    σ̃g = AugmentedGPLikelihoods.approx_expected_logistic.(-mean.(qg), c)
    λ = max(length(y) / (2 * dot(ψ, 1 .- σ̃g)), lik.invlink.λ)
    return HeteroscedasticGaussianLikelihood(InvScaledLogistic(λ))
end

function cavi!(fzs, x, y, ms, Ss, qΩ, lik; niter=10)
    K = ApproximateGPs._chol_cov(fzs[1])
    for _ in 1:niter
        posts_u = u_posterior.(fzs, ms, Ss)
        posts_fs = AbstractGPs.marginals.([p_u(x) for p_u in posts_u])
        lik = opt_lik(lik, posts_fs, y)
        aux_posterior!(qΩ, lik, y, invert(posts_fs))
        ηs, Λs = expected_auglik_potential_and_precision(lik, qΩ, y, invert(posts_fs))
        Ss .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(Λs)))
        ms .= Ss .* (ηs .+ Ref(K) .\ mean.(fzs))
    end
    return lik
end
# Now we just initialize the variational parameters
ms = nestedview(zeros(N, nlatent(lik)))
Ss = [Matrix{Float64}(I(N)) for _ in 1:nlatent(lik)]
qΩ = init_aux_posterior(lik, N)
fzs = [gp_f(x, 1e-8), gp_g(x, 1e-8)];
x_te = -10:0.01:10
# We run CAVI for 3-4 iterations
new_lik = cavi!(fzs, x, y, ms, Ss, qΩ, lik; niter=20);
# And visualize the obtained variational posterior
f_te = u_posterior(fzs[1], ms[1], Ss[1])(x_te)
g_te = u_posterior(fzs[2], ms[2], Ss[2])(x_te)
plot!(plt, x_te, mean(f_te); ribbon=sqrt.(new_lik.invlink.(mean(g_te))), label="y|q(f,g)")
plot!(plt2, f_te; color=1, linestye=:dash, alpha=0.5, label="q(f)")
plot!(plt2, g_te; color=2, linestyle=:dash, alpha=0.5, label="q(g)")
display(plot(plt, plt2))

##
# ## ELBO
# How can one compute the Augmented ELBO?
# Again AugmentedGPLikelihoods provides helper functions
# to not have to compute everything yourself
function aug_elbo(lik, u_post, x, y)
    qf = marginals(u_post(x))
    qΩ = aux_posterior(lik, y, qf)
    return expected_logtilt(lik, qΩ, y, qf) - aux_kldivergence(lik, qΩ, y) -
           kldivergence(u_post.approx.q, u_post.approx.fz)
end

# aug_elbo(lik, u_posterior(fz, m, S), x, y)
# ## Gibbs Sampling
# We create our Gibbs sampling algorithm (we could do something fancier with
# AbstractMCMC)
function gibbs_sample(fzs, fs, Ω; nsamples=200)
    K = ApproximateGPs._chol_cov(fzs[1])
    Σ = [zeros(N, N) for _ in 1:nlatent(lik)]
    μ = [zeros(N) for _ in 1:nlatent(lik)]
    return map(1:nsamples) do _
        aux_sample!(Ω, lik, y, invert(fs))
        Σ .=
            inv.(
                Symmetric.(
                    Ref(inv(K)) .+ Diagonal.(auglik_precision(lik, Ω, y, invert(fs)))
                )
            )
        μ .= Σ .* (auglik_potential(lik, Ω, y, invert(fs)) .+ Ref(K) .\ mean.(fzs))
        rand!.(MvNormal.(μ, Σ), fs) # this corresponds to f -> g
        return copy(fs)
    end
end;
# We initialize our random variables
fs_init = nestedview(randn(N, nlatent(lik)))
Ω = init_aux_variables(lik, N);
# Run the sampling for default number of iterations (200)
fs_samples = gibbs_sample(fzs, fs_init, Ω);
# And visualize the samples overlapped to the variational posterior
# that we found earlier.

for fs in fs_samples
    plot!(plt, x, fs[1]; color=3, alpha=0.07, label="")
    for i in 1:nlatent(lik)
        plot!(plt2, x, fs[i]; color=i, alpha=0.07, label="")
    end
end
plt2

plot(plt, plt2)
