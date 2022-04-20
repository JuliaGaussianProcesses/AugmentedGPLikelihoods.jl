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
Nclass = 3
x = collect(range(-10, 10; length=N))

# We will present simultaneously the bijective and non-bijective likelihood
liks = [
    CategoricalLikelihood(BijectiveSimplexLink(LogisticSoftMaxLink(zeros(Nclass)))),
    CategoricalLikelihood(LogisticSoftMaxLink(zeros(Nclass))),
]

# This is small hack until https://github.com/JuliaGaussianProcesses/GPLikelihoods.jl/pull/68 is merged
AugmentedGPLikelihoods.nlatent(::CategoricalLikelihood{<:BijectiveSimplexLink}) = Nclass - 1
AugmentedGPLikelihoods.nlatent(::CategoricalLikelihood{<:LogisticSoftMaxLink}) = Nclass

SplitApplyCombine.invert(x::ArrayOfSimilarArrays) = nestedview(flatview(x)')
# We deifne the models
kernel = 5.0 * with_lengthscale(SqExponentialKernel(), 2.0)
gp = GP(kernel)
fz = gp(x, 1e-8);
# We use a multi-output GP to generate our (N-1) latent GPs
gpm = GP(IndependentMOKernel(kernel))
X = MOInput(x, Nclass - 1);

# We sample (N-1) latent GPs to force the setting of the bijective version
# which the non-bijective version should also be able to recover
fs = rand(gpm(X, 1e-6))
fs = nestedview(reduce(hcat, Iterators.partition(fs, N)))
lik_true = liks[1](invert(fs)) # the likelihood
y = rand(lik_true);

# We build the one-hot encoding for each likelihood (different)
Ys = map(liks) do lik
    Y = nestedview(sort(unique(y))[1:nlatent(lik)] .== permutedims(y))
    return Y
end;
# We plot the sampled data
plts = map(["Bijective Logistic-softmax", "Logistic-softmax"]) do title
    plt = plot(; title)
    scatter!(plt, x, y; group=y, label=[1 2 3 4], msw=0.0)
    plot!(plt, x, vcat(fs, [zeros(N)]); color=[1 2 3 4], label="", lw=3.0)
end
plot(plts[1]; title="")
# ## CAVI Updates
# We write our CAVI algorithmm
function u_posterior(fz, m, S)
    return posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(copy(m), S)))
end

function cavi!(fz::AbstractGPs.FiniteGP, lik, x, Y, ms, Ss, qΩ; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    for _ in 1:niter
        posts_u = u_posterior.(Ref(fz), ms, Ss)
        posts_fs = marginals.([p_u(x) for p_u in posts_u])
        aux_posterior!(qΩ, lik, Y, SplitApplyCombine.invert(posts_fs))
        Ss .=
            inv.(
                Symmetric.(Ref(inv(K)) .+ Diagonal.(expected_auglik_precision(lik, qΩ, Y)))
            )
        ms .= Ss .* (expected_auglik_potential(lik, qΩ, Y) .- Ref(K \ mean(fz)))
    end
    return ms, Ss
end
# Now we just initialize the variational parameters and run CAVI
ms_Ss = map(liks, Ys) do lik, Y
    m = nestedview(zeros(N, nlatent(lik)))
    S = [Matrix{Float64}(I(N)) for _ in 1:nlatent(lik)]
    qΩ = init_aux_posterior(lik, N)
    fz = gp(x, 1e-8)
    cavi!(fz, lik, x, Y, m, S, qΩ; niter=20)
    return (; m, S)
end
# And visualize the obtained variational posterior
x_te = -10:0.01:10
for i in 1:2
    for j in 1:nlatent(liks[i])
        plot!(
            plts[i],
            x_te,
            u_posterior(fz, ms_Ss[i].m[j], ms_Ss[i].S[j]);
            color=j,
            alpha=0.3,
            lw=3.0,
            label="",
        )
    end
end
plot(plts...)

p_plts = [plot() for _ in 1:2]
for i in 1:2
    scatter!(p_plts[i], x, y / Nclass; group=y, label=[1 2 3 4], msw=0.0)

    ## vline!(p_plts[i], x, group=fs_ys[i].y, lw=20/length(x) * 20.0, alpha=0.2, ylims=(0,1),title="p(y=k|f)", label="")
    lik_pred =
        liks[i].(
            invert(
                mean.([
                    u_post(x_te) for u_post in u_posterior.(Ref(fz), ms_Ss[i].m, ms_Ss[i].S)
                ]),
            ),
        )
    ps = getproperty.(lik_pred, :p)
    lik_pred_x =
        liks[i].(
            invert(
                mean.([
                    u_post(x) for u_post in u_posterior.(Ref(fz), ms_Ss[i].m, ms_Ss[i].S)
                ]),
            ),
        )
    ps_x = getproperty.(lik_pred_x, :p)
    ps_true = getproperty.(lik_true.v, :p)
    @show sum(norm, ps_x .- ps_true)
    @show sum(zip(lik_pred_x, y)) do (p, y)
        logpdf(p, y)
    end
    for k in 1:Nclass
        plot!(p_plts[i], x, invert(ps_true)[k]; color=k, lw=2.0, label=k)
        plot!(p_plts[i], x_te, invert(ps)[k]; color=k, lw=2.0, label="", ls=:dash)
    end
end
plot(p_plts...)
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
function gibbs_sample(fz, lik, Y, fs, Ω; nsamples=200)
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
samples = map(liks, Ys, plts) do lik, Y, plt
    fs_init = nestedview(randn(N, nlatent(lik)))
    Ω = init_aux_variables(lik, N)
    ## Run the sampling for default number of iterations (200)
    return gibbs_sample(fz, lik, Y, fs_init, Ω)
    ## And visualize the samples overlapped to the variational posterior
    ## that we found earlier.
end;

for i in 1:2
    for fs in samples[i]
        for j in 1:nlatent(liks[i])
            plot!(plts[i], x, fs[j]; color=j, alpha=0.07, label="")
        end
    end
end
plot(plts...)
