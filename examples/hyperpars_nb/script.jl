# # Hyperparameter learning with augmented GP likelihoods

# We load all the necessary packages
using ApproximateGPs
using AugmentedGPLikelihoods
using Distributions
using LinearAlgebra
using Optim
using ParameterHandling
using Random
using Statistics
using StatsBase
using Zygote

# ## Global model settings
kernel(θ) = θ.variance * SEKernel() ∘ ScaleTransform(θ.invlengthscale)
likelihood(θ) = NegativeBinomialLikelihood(NBParamFailure(θ.r))

# We subtract the logarithm of the likelihood parameter from the
# mean. This accomplishes the same as if the likelihood were para-
# metrized to yield E[y|f] = exp(f), and makes E[y] independent of
# r.
build_f(θ) = GP(θ.mean - log(θ.r), kernel(θ))
build_latent_gp(θ) = LatentGP(build_f(θ), likelihood(θ), 1e-6)

# ## ELBO
# We define abstractions for holding the data for optimization
# and methods to initialize, evaluate, and optimize the loss

abstract type AbstractLoss end

function optimize(l::AbstractLoss)
    θ0 = init(l)
    θ0_flat, unflatten = ParameterHandling.flatten(θ0)
    unpack = ParameterHandling.value ∘ unflatten

    opt = Optim.optimize(
        l ∘ unpack,
        θ -> only(Zygote.gradient(l ∘ unpack, θ)),
        θ0_flat,
        Optim.BFGS(;
            alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
            linesearch=Optim.LineSearches.BackTracking(),
        ),
        inplace=false,
    )
    display(opt)
    return unpack(opt.minimizer)
end

function init_(l::AbstractLoss)
    θ0 = init(l)
    θ0_flat, unflatten = ParameterHandling.flatten(θ0)
    unpack = ParameterHandling.value ∘ unflatten
    return unpack(θ0_flat)
end

function (l::AbstractLoss)()
    θ = init_(l)
    return l(θ)
end

# This function will construct the posterior from the data and parameters.
# Implementations below.
function build_posterior(l::AbstractLoss, θ) end

# This function estimates initial values for the mean and variance of the
# latent function f, based on the marginal mean and variance of y, assuming
# that the data came from a latent GP with Poisson likelihood.
# This will be used below in the initializations of the optimizer.
function initial_μ_σ²_Poisson(mean_y, var_y)
    var_y < mean_y && return zero(mean_y), one(var_y)
    σ² = log((var_y / mean_y - 1) / mean_y + 1)
    μ = log(mean_y) - σ²/2
    return μ, σ²
end

function correlation_length(y)
    logvar = log(autocov(y, [0])[1])
    i0 = length(y)
    for i in 0:length(y)
        ac = autocov(y, [i])[1]
        if log(ac) - logvar < -0.5
            i0 = i
            break
        end
    end
    return i0
end

# ## Diagonal SVGP model
# We define a standard diagonal sparse variational GP for comparison

struct DSVGPLoss{Tx, Ty} <: AbstractLoss
    x::Tx # input data
    y::Ty # observations
end

function init(l::DSVGPLoss)
    μ, σ² = initial_μ_σ²_Poisson(mean(l.y), var(l.y))
    xmin = minimum(l.x)
    xmax = maximum(l.x)
    Δt = (xmax - xmin) / length(l.x)
    invl0 = inv(correlation_length(l.y) * Δt)
    r0 = 100.
    M = min(30, length(l.y))
    z0 = rand(M)
    z0 .*= xmax - xmin
    z0 .+= xmin
    return (
        mean = μ,
        invlengthscale = positive(invl0),
        variance = positive(σ²),
        r = positive(r0),
        z = bounded.(z0, xmin, xmax),
        m = zeros(M),
        d = positive.(ones(M))
    )
end

build_model(::DSVGPLoss, θ) = build_latent_gp(θ)

function (l::DSVGPLoss)(θ)
    f = build_model(l, θ)
    fz = f(θ.z).fx # pseudo-points
    q = MvNormal(θ.m, Diagonal(θ.d))  # variational guide
    approx = SparseVariationalApproximation(fz, q)
    L = elbo(approx, f(l.x), l.y)
    return -L
end

function build_posterior(l::DSVGPLoss, θ)
    f = build_model(l, θ)
    fz = f(θ.z).fx # pseudo-points
    q = MvNormal(θ.m, Diagonal(θ.d))  # variational guide
    approx = SparseVariationalApproximation(fz, q)
    post = posterior(approx, f(l.x), l.y)
    posty = LatentGP(post, f.lik, f.Σy)
    residuals = l.y - mean(posty(l.x))
    return posty, residuals
end


# ## Augmented variational GP
# We try to replicate the SVGP, but using the augmented GP machinery
# In order to use the CAVI updates from the documentation of 
# GPLikelihoods.jl, we use a full covariance matrix for the variational
# approximation

struct AVGPLoss{Tx, Ty, Tm, TS} <: AbstractLoss
    x::Tx # input data
    y::Ty # observations
    m::Tm # variational mean
    S::TS # variational covariance matrix
end

function AVGPLoss(x, y)
    M = length(y)
    m = zeros(M)
    S = Symmetric(Matrix{Float64}(I, M, M))
    return AVGPLoss(x, y, m, S)
end

function init(l::AVGPLoss)
    μ, σ² = initial_μ_σ²_Poisson(mean(l.y), var(l.y))
    xmin = minimum(l.x)
    xmax = maximum(l.x)
    Δt = (xmax - xmin) / length(l.x)
    invl0 = inv(correlation_length(l.y) * Δt)
    r0 = 100.
    return (
        mean = μ,
        invlengthscale = positive(invl0),
        variance = positive(σ²),
        r = positive(r0),
    )
end

build_model(::AVGPLoss, θ) = build_latent_gp(θ)

function (l::AVGPLoss)(θ)
    f = build_model(l, θ)
    lik = f.lik
    fz = f(l.x).fx
    N = length(l.x)
    qΩ = init_aux_posterior(lik, N)
    Zygote.@ignore cavi!(fz, lik, l.x, l.y, l.m, l.S, qΩ; niter=1)
    L = aug_elbo(lik, u_posterior(fz, l.m, l.S), l.x, l.y)
    return -L
end

# In order to make this work with Zygote, we had to change from
# `Centered` to `NonCentered` here. This deviates from the usual practice.

function u_posterior(fz::AbstractGPs.FiniteGP, m::AbstractVector, S::AbstractMatrix)
    return posterior(SparseVariationalApproximation(NonCentered(), fz, MvNormal(m, S)))
end

function aug_elbo(lik::GPLikelihoods.AbstractLikelihood, u_post, x::AbstractVector, y::AbstractVector)
    qf = marginals(u_post(x))
    qΩ = Zygote.@ignore aux_posterior(lik, y, qf)
    term1 = expected_logtilt(lik, qΩ, y, qf)
    term2 = ApproximateGPs.SparseVariationalApproximationModule._prior_kl(u_post.approx)
    term3 = Zygote.@ignore begin
        aux_kldivergence(lik, qΩ, y)
    end
    return term1 - term2 - term3
end

# This function had to be modified for the `NonCentered` `SparseVariationalApproximation`
# This is an educated guess, hopefully it is correct!

function cavi!(fz::AbstractGPs.FiniteGP, lik, x, y, m, S, qΩ; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    κ = K.L
    for _ in 1:niter
        post_u = u_posterior(fz, m, S)
        post_fs = marginals(post_u(x))
        aux_posterior!(qΩ, lik, y, post_fs)
        S .= inv(Symmetric(I + κ' * Diagonal(only(expected_auglik_precision(lik, qΩ, y))) * κ))
        m .= S * κ' * only(expected_auglik_potential(lik, qΩ, y)) - κ \ mean(fz)
    end
    return m, S
end

function build_posterior(l::AVGPLoss, θ)
    f = build_model(l, θ)
    lik = f.lik
    fz = f(l.x).fx
    N = length(l.x)
    qΩ = init_aux_posterior(lik, N)
    cavi!(fz, lik, l.x, l.y, l.m, l.S, qΩ; niter=100)
    post = u_posterior(fz, l.m, l.S)
    posty = LatentGP(post, lik, 1e-6)
    residuals = l.y - mean(posty(l.x))
    return posty, residuals
end


# ## Experiments
# We generate some synthetic data

Random.seed!(1234)
N = 100
x = range(0, 1; length=N)
μ, σ² = initial_μ_σ²_Poisson(100, 5000)
k = σ² * with_lengthscale(SEKernel(), 0.1)
r = 100.
gp = GP(μ - log(r), k)
lik = NegativeBinomialLikelihood(NBParamFailure(r))
lgp = LatentGP(gp, lik, 1e-6)
f, y = rand(lgp(x))

# We plot the latent function and the observations
using Plots
p1 = scatter(x, y, label = "y", color = :black, markersize = 3, legend = :outerleft)
p2 = scatter(x, f, label = "f", legend = :outerleft, color = :black, lw = 1)
plot(p1, p2; layout=(2, 1), size = (1000, 400))

# We define the model loss functions
loss1 = DSVGPLoss(x, y)
loss2 = AVGPLoss(x, y)

# We call and time the `optimize` function to find optimal parameters
# The augmented GP leads to a considerable speedup, at least for this
# problem size.
@time θ1 = optimize(loss1)
@time θ2 = optimize(loss2)

# We evaluate the loss function after training
@show loss1(θ1)
@show loss2(θ2)

# We construct the posterior and residuals after training
post1, res1 = build_posterior(loss1, θ1)
post2, res2 = build_posterior(loss2, θ2)

# We calculate the mean squared errors
@show mean(abs2, res1) / var(y)
@show mean(abs2, res2) / var(y)

# We calculate Monte Carlo estimates of medians and quantiles of y
function approx_post_95_CI(post, x::AbstractVector, N::Int)
    samples = mapreduce(i -> rand(post(x)).y, hcat, 1:N)
    return quantile.(eachrow(samples), 0.025), 
        quantile.(eachrow(samples), 0.5), 
        quantile.(eachrow(samples), 0.975)
end

x_pr = vcat(x, 1.01:0.01:1.25) # input for posteriors
q11, m1, q12 = approx_post_95_CI(post1, x_pr, 10000)
q21, m2, q22 = approx_post_95_CI(post2, x_pr, 10000)




# Finally, we plot all results
p1 = scatter(x, y, label = "y", color = :black, markersize = 3, legend = :outerleft)

plot!(p1, x_pr, m1, label = "DSVGP posterior median", color = :blue)
plot!(
    p1, x_pr, q11;
    fillrange = q12, 
    label = "DSVGP posterior 95% CI", 
    color = :blue, lw = 0.1, fillalpha = 0.3, linealpha=0
)

plot!(p1, x_pr, m2, label = "AVGP posterior median", color = :green)
plot!(
    p1, x_pr, q21;
    fillrange = q22, 
    label = "AVGP posterior 95% CI", 
    color = :green, lw = 0.1, fillalpha = 0.3, linealpha=0
)

p2 = scatter(x, f, label = "f", legend = :outerleft, color = :black, lw = 1)
plot!(p2, x_pr, post1(x_pr).fx, label = "DSVGP posterior for f", color = :blue)
plot!(p2, x_pr, post2(x_pr).fx, label = "AVGP posterior for f", color = :green)

plt = plot(p1, p2; layout=(2, 1), size = (1000, 400))
