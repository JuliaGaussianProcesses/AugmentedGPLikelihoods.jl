# # Classification with augmented variables

# We load all the necessary packages
using AbstractGPs
using ApproximateGPs
using AugmentedGPLikelihoods
using Distributions
using LinearAlgebra
# Plotting libraries
using Plots

# We create some random data (sorted for plotting reasons)
N = 100
x = range(-10, 10; length=N)
kernel = with_lengthscale(SqExponentialKernel(), 2.0)
gp = GP(kernel)
lik = BernoulliLikelihood()
lf = LatentGP(gp, lik, 1e-6)
f, y = rand(lf(x))
# We plot the sampled data
plt = scatter(x, y; label="Data")
plot!(plt, x, f; color=:red, label="Latent GP")
# ## CAVI Updates
# We write our CAVI algorithmm
function u_posterior(fz, m, S)
    posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))
end

function cavi!(fz::AbstractGPs.FiniteGP, x, y, m, S, Ω; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    for _ in 1:niter
        post_u = u_posterior(fz, m, S)
        post_fs = marginals(post_u(x))
        Ω = aux_posterior!(Ω, lik, y, post_fs)
        S .= inv(Symmetric(inv(K) + Diagonal(only(vi_rate(lik, Ω, y)))))
        m .= S * (only(vi_shift(lik, Ω, y)) - K \ mean(fz))
    end
    return m, S, Ω
end
# Now we just initialize the variational parameters
m = zeros(N)
S = Matrix{Float64}(I(N))
Ω = init_aux_posterior(lik, N)
fz = gp(x, 1e-8)
# And visualize the current posterior
x_te = -10:0.01:10
plot!(plt, x_te, u_posterior(fz, m, S); color=:blue, alpha=0.3, label="Initial VI Posterior")
# We run CAVI for 3-4 iterations
cavi!(fz, x, y, m, S, Ω; niter=4)
# And visualize the obtained variational posterior
plot!(plt, x_te, u_posterior(fz, m, S); color=:darkgreen, alpha=0.3, label="Final VI Posterior")

# ## Gibbs Sampling
# We create our Gibbs sampling algorithm (we could do something fancier with
# AbstractMCMC)
function gibbs_sample(fz, f, Ω; nsamples=200)
    K = ApproximateGPs._chol_cov(fz)
    Σ = zeros(length(f), length(f))
    μ = zeros(length(f))
    return map(1:nsamples) do _
        aux_sample!(Ω, lik, y, f)
        Σ .= inv(Symmetric(inv(K) + Diagonal(only(sample_rate(lik, Ω, y)))))
        μ .= Σ * (only(sample_shift(lik, Ω, y)) - K \ mean(fz))
        rand!(MvNormal(μ, Σ), f)
        return copy(f)
    end
end;
# We initialize our random variables
f = randn(N)
Ω = init_aux_variables(lik, N);
# Run the sampling for default number of iterations (200)
fs = gibbs_sample(fz, f, Ω);
# And visualize the samples overlapped to the variational posterior
# that we found earlier.
for f in fs
    plot!(plt, x, f; color=:blue, alpha=0.07, label="")
end
plt
# ## ELBO
# How can one compute the Augmented ELBO?
# Again AugmentedGPLikelihoods provides helper functions
# to not have to compute everything yourself
function aug_elbo(lik, u_post, Ω, x, y)
    qf = marginals(u_post(x))
    aug_expected_loglik(lik, Ω, y, qf) - kl_term(lik, Ω, y) - kl_term(u_post.q, u_post.fz)
end
