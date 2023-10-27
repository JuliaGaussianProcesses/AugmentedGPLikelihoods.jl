```@meta
CurrentModule = AugmentedGPLikelihoods
```

# AugmentedGPLikelihoods

Documentation for [AugmentedGPLikelihoods](https://github.com/theogf/AugmentedGPLikelihoods.jl).

Gaussian Processes (GPs) are great tools for working with function approximations
including uncertainty.
On top of Gaussian regression, GPs can be used as latent functions for a lot of
different tasks such as non-Gaussian regression, classification, multi-class
classification or event counting.
However, these tasks involve non-conjugate likelihoods and the GP posterior 
is then intractable.
The typical solution is to approximate the posterior by either sampling from it
or approximating it with another distribution.
However, both these methods are computationally involved, require gradient
and are not always guaranteed to converge.

## The augmentation

An alternative proposed in [galy20](@cite) is to represent these non-conjugate
likelihoods as scale-mixtures (sometimes requiring multiple steps) to obtain
a __conditionally conjugate likelihood__.
More concretely, some likelihoods can be written as:

```math
    p(x) = \int q(x|\omega)d\pi(\omega).
```

One can then get rid of the integral by __augmenting__ the model with the
auxiliary variable ``\omega``.
With the right choice of mixture, one can obtain a likelihood conjugate with a
GP ``f`` but only when conditioned on ``\omega``.
This means we go from

```math
    p(f|y) = \frac{p(y|f)p(f)}{p(y)},
```

which is intractable, to

```math
    p(f,\omega|y) = \frac{p(y,\omega|f)p(f)}{p(y)},
```

where the conditionals ``p(f|\omega,y)`` and ``p(\omega|f,y)`` are known
in closed-form.

## Bayesian Inference

This new formulation leads to easier, more stable and faster inference.
A natural algorithm following from this formulation is the
[__Blocked Gibbs Sampling__](https://en.wikipedia.org/wiki/Gibbs_sampling#Blocked_Gibbs_sampler)
but also the Coordinate Ascent VI (CAVI) algorithm where the conditionals are used
to compute the optimal variational distribution.

## What this package does

Although an automatic way is proposed in [galy20](@cite), most of the
augmentations require some hand derivations.
This package provides the results of these derivations (as well as the derivations)
and propose a unified framework to obtain the distributions of interest.
Generally values (either samples or distributions) of the auxiliary variables
need to be carried around. Since, each likelihood can have a different structure
of auxiliary variables, they are uniformly moved as ``\Omega``, which is
a `NamedTuple` containing `Vector`s of either samples or distributions.

## [Gibbs Sampling](@id gibbs-sampling-index)

We give an example in the Gibbs Sampling tutorial using `AbstractMCMC`.
But the API can be reduced to 5 main functions:

```@docs
init_aux_variables
aux_sample!
aux_sample
aux_full_conditional
auglik_potential
auglik_precision
auglik_potential_and_precision
```

First [`init_aux_variables`](@ref) initializes a `NamedTuple`
of `Vector`(s) of auxiliary variables to be modified in-place during sampling.
[`aux_sample!`](@ref) will sample the auxiliary variables from the full
conditional ``p(\Omega|y,f)``, given by [`aux_full_conditional`](@ref), and return the modified `NamedTuple`.
For generating a new `NamedTuple` every time, see [`aux_sample`](@ref).
The full-conditional from ``f`` are given by

```math
\begin{align*}
    p(f|y,\Omega) =& \mathcal{N}(f|m,S)\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(\lambda)\right)^{-1}\\
    m =& S \left(h + K^{-1}\mu_0\right),
\end{align*}
```

where ``\lambda`` is obtained via [`auglik_precision`](@ref) and ``h`` is
obtained via [`auglik_potential`](@ref).
For likelihoods requiring multiple latent GP (e.g. multi-class classification
or heteroscedastic likelihoods), [`auglik_potential`](@ref) and [`auglik_precision`](@ref)
return a `Tuple` with the respective `Vector`s ``\lambda`` and ``h``.

As a general rule, the augmented likelihood will have the form

```math
    p(y|f,\Omega) \propto \exp\left(h(\Omega,y)f + \frac{\lambda(\Omega,y)}{2}f^2\right),
```

with [`auglik_potential`](@ref) ``\equiv h(\Omega,y)`` while [`auglik_precision`](@ref)
``\equiv \lambda(\Omega,y)``.

## Coordinate Ascent Variational Inference

[CAVI](https://en.wikipedia.org/wiki/Coordinate_descent) updates work exactly
the same way as the Gibbs Sampling, except we are now working with posterior
distributions instead of samples.
We work with the variational family ``q(f)\prod_{i=1}^N q(\Omega_i)``, i.e.
we make the [__mean-field__](https://en.wikipedia.org/wiki/Mean-field_theory)
assumption that our new auxiliary variables are independent of each other
and independent of ``f``.

Like for [Gibbs Sampling](@ref gibbs-sampling-index), there are also 5 main functions

```@docs
init_aux_posterior
aux_posterior!
aux_posterior
expected_auglik_potential
expected_auglik_precision
expected_auglik_potential_and_precision
expected_aug_loglik
```

[`init_aux_posterior`](@ref) initializes the posterior
``\prod_{i=1}^Nq(\Omega_i)`` as a `NamedTuple`.
[`aux_posterior!`](@ref) updates the variational posterior distributions in-place
given the marginals ``q(f_i)`` and return the modified `NamedTuple`.
To get a new `NamedTuple` every time, use [`aux_posterior`](@ref).
Finally, [`expected_auglik_potential`](@ref) and [`expected_auglik_precision`](@ref)
give us the elements needed to update the variational distribution ``q(f)``.
Like for [Gibbs Sampling](@ref gibbs-sampling-index), we have the following optimal
variational distributions:

```math
\begin{align*}
    q^*(f) =& \mathcal{N}(f|m,S)\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(\lambda)\right)^{-1}\\
    m =& S \left(h + K^{-1}\mu_0\right)
\end{align*}
```

where ``\lambda`` is given by [`expected_auglik_precision`](@ref) and ``h`` is given by [`expected_auglik_potential`](@ref).
Note that if you work with __Sparse__ GPs, the updates should be:

```math
\begin{align*}
    S =& \left(K_{Z}^{-1} + \kappa\operatorname{Diagonal}(r)\kappa^\top\right)^{-1},\\
    m =& S \left(\kappa t + K_Z^{-1}\mu_0(Z)\right),
\end{align*}
```

where ``\kappa=K_{Z}^{-1}K_{Z,X}``.

## Likelihood and ELBO computations

You might be interested in computing the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound),
i.e. the lower bound on the evidence of the augmented model.
The ELBO can be written as:

```math
\begin{align*}
    \mathcal{L(q(f,\Omega)} =& E_{q(f)q(\Omega)}\left[\log p(y,\Omega|f)\right] - \operatorname{KL}\left(q(f)||p(f)\right)\\
    =& E_{q(f)q(\Omega)}\left[\log p(y|\Omega,f)\right] - \operatorname{KL}\left(q(\Omega)||p(\Omega)\right) -\operatorname{KL}\left(q(f)||p(f)\right)
\end{align*}
```

To work with all these terms, `AugmentedGPLikelihoods.jl` provide a series of
helping functions:

```@docs
aug_loglik
aux_prior
aux_kldivergence
logtilt
expected_logtilt
```

[`aug_loglik`](@ref) returns the augmented log-likelihood ``\log p(y,\Omega|f)``,
but one should avoid using it as computing ``p(\Omega)`` can be expensive.
[`aux_prior`](@ref) returns the prior on the auxiliary variables, note that it
can depend on the observations ``y``.
[`logtilt`](@ref) returns the log of the exponential part of the augmented likelihood, which is conjugate with ``f``.
[`aug_loglik`](@ref) is computed by default using [`logtilt`](@ref) + `logpdf(aux_prior, x)`.
Finally, for variational inference purposes, [`aux_kldivergence`](@ref) computes
the KL divergence ``\operatorname{KL}(q(\Omega)||p(\Omega))`` and [`expected_logtilt`](@ref) computes the expectation of [`logtilt`] analytically.
