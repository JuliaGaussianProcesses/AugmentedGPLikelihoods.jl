# # Multi-Class Classification

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
Nclass = 4
x = range(-10, 10; length=N)
lik = CategoricalLikelihood(BijectiveSimplexLink(logisticsoftmax))
AugmentedGPLikelihoods.nlatent(::CategoricalLikelihood) = Nclass - 1
invert(x::ArrayOfSimilarArrays) = nestedview(flatview(x)')
X = MOInput(x, nlatent(lik))
kernel = with_lengthscale(SqExponentialKernel(), 2.0)
gp = GP(kernel)
gpm = GP(IndependentMOKernel(kernel))
fs = rand(gpm(X, 1e-6))
fs = nestedview(reduce(hcat, Iterators.partition(fs, N)))
y = rand(lik(invert(fs)))
Y = nestedview(unique(y)[1:end-1] .== permutedims(y))
# lf = LatentGP(gp, lik, 1e-6)
# f, y = rand(lf(X));
# We plot the sampled data
plt = scatter(x, y; group=y, label=[1 2 3 4], msw=0.0)
plot!(plt, x, [fs, zeros(N)]; color=[1 2 3 4], label="", lw=3.0)
# ## CAVI Updates
# We write our CAVI algorithmm
function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(m, S)))
end

function cavi!(fz::AbstractGPs.FiniteGP, x, y, ms, Ss, qΩ; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    for _ in 1:niter
        posts_u = u_posterior.(Ref(fz), m, S)
        posts_fs = marginals.([p_u(x) for p_u in posts_u])
        aux_posterior!(qΩ, lik, y, SplitApplyCombine.invert(posts_fs))
        Ss .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(expected_auglik_precision(lik, qΩ, y))))
        ms .= Ss .* (expected_auglik_potential(lik, qΩ, y) .- Ref(K \ mean(fz)))
    end
    return ms, Ss
end
# Now we just initialize the variational parameters
ms = [zeros(N) for _ in 1:nlatent(lik)]
Ss = [Matrix{Float64}(I(N)) for _ in 1:nlatent(lik)]
qΩ = init_aux_posterior(lik, N)
fz = gp(x, 1e-8);
# And visualize the current posterior
x_te = -10:0.01:10
for i in 1:nlatent(lik)
    plot!(
        plt, x_te, u_posterior(fz, ms[i], Ss[i]); color=i, alpha=0.3, label=""
    )
end  
plt
# We run CAVI for 3-4 iterations
cavi!(fz, x, Y, ms, Ss, qΩ; niter=4);
# And visualize the obtained variational posterior
for i in 1:nlatent(lik)
    plot!(
        plt,
        x_te,
        u_posterior(fz, ms[i], Ss[i]);
        color=i,
        alpha=0.3,
        label="",
    )
end
plt
# ## Classification - ELBO
# How can one compute the Augmented ELBO?
# Again AugmentedGPLikelihoods provides helper functions
# to not have to compute everything yourself
function aug_elbo(lik, u_post, x, y)
    qf = marginals(u_post(x))
    qΩ = aux_posterior(lik, y, qf)
    return expected_logtilt(lik, qΩ, y, qf) - aux_kldivergence(lik, qΩ, y) -
           ApproximateGPs._prior_kl(u_post.approx)
end

# aug_elbo(lik, u_posterior(fz, m, S), x, y)
# ## Classification - Gibbs Sampling
# We create our Gibbs sampling algorithm (we could do something fancier with
# AbstractMCMC)
function gibbs_sample(fz, f, Ω; nsamples=200)
    K = ApproximateGPs._chol_cov(fz)
    Σ = zeros(length(f), length(f))
    μ = zeros(length(f))
    return map(1:nsamples) do _
        aux_sample!(Ω, lik, y, f)
        Σ .= inv(Symmetric(inv(K) + Diagonal(only(auglik_precision(lik, Ω, y)))))
        μ .= Σ * (only(auglik_potential(lik, Ω, y)) - K \ mean(fz))
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
    plot!(plt, x, f; color=:black, alpha=0.07, label="")
end
plt
