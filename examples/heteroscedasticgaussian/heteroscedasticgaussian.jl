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
lik = HeteroscedasticGaussianLikelihood(InvScaledLogistic(10.0))
SplitApplyCombine.invert(x::ArrayOfSimilarArrays) = nestedview(flatview(x)')
X = MOInput(x, nlatent(lik))
kernel = 10.0 * with_lengthscale(SqExponentialKernel(), 2.0)
gpf = GP(kernel)
μ₀g = -5.0
gpg = GP(μ₀g, kernel)
gps = [gpf, gpg]
gpm = GP(IndependentMOKernel(kernel))
fs = rand(gpm(X, 1e-6))
fs = nestedview(reduce(hcat, Iterators.partition(fs, N)))
fs[2] .+= μ₀g
py = lik(invert(fs))
y = rand(py)
# lf = LatentGP(gp, lik, 1e-6)
# f, y = rand(lf(X));
# We plot the sampled data
plt = plot(x, fs[1], label="f", lw=2.0)
scatter!(plt, x, y; msw=0.0, label="y")
plt2 = plot(x, fs, label=["f" "g"], lw=2.0)
# ## CAVI Updates
# We write our CAVI algorithmm
function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(copy(m), S)))
end

function cavi!(fzs, x, y, ms, Ss, qΩ; niter=10)
    K = ApproximateGPs._chol_cov(fzs[1])
    for _ in 1:niter
        posts_u = u_posterior.(fzs, ms, Ss)
        posts_fs = marginals.([p_u(x) for p_u in posts_u])
        aux_posterior!(qΩ, lik, y, invert(posts_fs))
        ηs, Λs = expected_auglik_potential_and_precision(lik, qΩ, y, last.(invert(posts_fs)))
        Ss .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(Λs)))
        ms .= Ss .* (ηs .- Ref(K) .\ mean.(fzs))
    end
    return ms, Ss
end
# Now we just initialize the variational parameters
ms = nestedview(zeros(N, nlatent(lik)))
Ss = [Matrix{Float64}(I(N)) for _ in 1:nlatent(lik)]
qΩ = init_aux_posterior(lik, N)
fzs = [gpf(x, 1e-8), gpg(x, 1e-8)];
x_te = -10:0.01:10
# We run CAVI for 3-4 iterations
cavi!(fzs, x, y, ms, Ss, qΩ; niter=20);
# And visualize the obtained variational posterior
f_te = u_posterior(fzs[1], ms[1], Ss[1])(x_te)
g_te = u_posterior(fzs[2], ms[2], Ss[2])(x_te)
plot!(plt, x_te, mean(f_te), ribbon=sqrt.(lik.invlink.(mean(g_te))), label="p(y|f,g)")
plot!(plt2, f_te; color=1, linestye=:dash, alpha=0.3, label="q(f)")
plot!(plt2, g_te; color=2, linestyle=:dash, alpha=0.3, label="q(g)")
plot(plt, plt2) |> display

##
plt2 = vline(x, group=y, lw=20/length(x) * 20.0, alpha=0.3, ylims=(0,1),title="p(y=k|f)")
ps = getproperty.(lik.(invert(mean.([u_post(x_te) for u_post in u_posterior.(Ref(fz), ms, Ss)]))), :p)
for i in 1:Nclass
    plot!(plt2, x_te, invert(ps)[i], color=i, lw =2.0, label="")
end
plt2 |> display
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
function gibbs_sample(fz, fs, Ω; nsamples=200)
    K = ApproximateGPs._chol_cov(fz)
    Σ = [zeros(N, N) for _ in 1:nlatent(lik)]
    μ = [zeros(N) for _ in 1:nlatent(lik)]
    return map(1:nsamples) do _
        aux_sample!(Ω, lik, Y, invert(fs))
        Σ .= inv.(Symmetric.(Ref(inv(K)) .+ Diagonal.(auglik_precision(lik, Ω, Y))))
        μ .= Σ .* (auglik_potential(lik, Ω, Y) .- Ref(K \ mean(fz)))
        rand!.(MvNormal.(μ, Σ), fs)
        return copy(fs)
    end
end;
# We initialize our random variables
fs_init = nestedview(randn(N, nlatent(lik)))
Ω = init_aux_variables(lik, N);
# Run the sampling for default number of iterations (200)
fs_samples = gibbs_sample(fz, fs_init, Ω);
# And visualize the samples overlapped to the variational posterior
# that we found earlier.
for fs in fs_samples
    for i in 1:nlatent(lik)
        plot!(plt, x, fs[i]; color=i, alpha=0.07, label="")
    end
end