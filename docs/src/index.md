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
However, both these methosd are computationally involved, require gradient
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
auxiliary variable $$\omega$$.
With the right choice of mixture, one can obtain a likelihood conjugate with a 
GP $$f$$ but only when conditioned on $$\omega$$.
This means we go from
```math
    p(f|y) = \frac{p(y|f)p(f)}{p(y)},
```
which is intractable, to
```math
    p(f,\omega|y) = \frac{p(y,\omega|f)p(f)}{p(y)},
```
where the conditionals $$p(f|\omega,y)$$ and $$p(\omega|f,y)$$ are known
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
of auxiliary variables, they are uniformly moved as $$\Omega$$, which is 
a `NamedTuple` containing `Vector`s of either samples or distributions.

## Gibbs Sampling

We give an example in the Gibbs Sampling tutorial using `AbstractMCMC`.
But the API can be reduced to 5 main functions:
```@docs
init_aux_variables
aux_sample!
aux_sample
sample_shift
sample_rate
```
First [`init_aux_variables`](@ref) initializes a `NamedTuple` 
of `Vector`(s) of auxiliary variables to be modified in-place during sampling.
[`aux_sample!`](@ref) will sample the auxiliary variables from the full 
conditional $$p(\Omega|y,f)$$ and return the modified `NamedTuple`.
For generating a new `NamedTuple` every time, see [`aux_sample`](@ref).
The full-conditional from $$f$$ are given by
```math
\begin{align*}
    p(f|y,\Omega) =& \mathcal{N}(f|m,S)\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(r)\right)^{-1}\\
    m =& S \left(t + K^{-1}\mu_0\right)
\end{align*}
```
[`sample_shift`](@ref) returns a `Tuple` containing the `Vector` $$t$$,
and [`sample_rate`](@ref) returns a `Tuple` containing the `Vector` $$r$$.
For likelihoods requiring multiple latent GP (e.g. multi-class classification 
or heteroscedastic likelihoods), [`sample_shift`](@ref) and [`sample_rate`](@ref)
return a `Tuple` with the respective `Vector`s $$t$$ and $$r$$.

As a general rule, the augmented likelihood will have the form
```math
    p(y|f,\Omega) \propto \exp\left(a(\Omega,y)f + b(\Omega,y)f^2\right),
```
and [`sample_shift`](@ref) $$\equiv a(\Omega,y)$$ while [`sample_rate`](@ref) 
$$\equiv 2b(\Omega,y)$$.

## Coordinate Ascent Variational Inference

[CAVI](https://en.wikipedia.org/wiki/Coordinate_descent) updates work exactly 
the same way as the Gibbs Sampling, except we are now working with posterior 
distributions instead of samples.
We work with the variational family $$q(f)\prod_{i=1}^N q(\Omega_i)$$, i.e. 
we make the [__mean-field__](https://en.wikipedia.org/wiki/Mean-field_theory) 
assumption that our new auxiliary variables are independent of each other 
and independent of $$f$$.

Like for [Gibbs Sampling](@ref), there are also 5 main functions

```@docs
init_aux_posterior
aux_posterior!
aux_posterior
vi_shift
vi_rate
```

[`init_aux_posterior`](@ref) initializes the posterior 
$$\prod_{i=1}^Nq(\Omega_i)$$ as a `NamedTuple`.
[`aux_posterior!`](@ref) updates the variational posterior distributions in-place
given the marginals $$q(f_i)$$ and return the modified `NamedTuple`.
To get a new `NamedTuple` every time, use [`aux_posterior`](@ref).
Finally, [`vi_shift`](@ref) and [`vi_rate`](@ref) 
give us the elements needed to update the variational distribution $$q(f)$$.
Like for [Gibbs Sampling](@ref), we have the following optimal
variational distributions:
```math
\begin{align*}
    q^*(f) =& \mathcal{N}(f|m,S)\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(r)\right)^{-1}\\
    m =& S \left(t + K^{-1}\mu_0\right)
\end{align*}
```
where $$r$$ is given by [`vi_rate`](@ref) and $$t$$ is given by [`vi_shift`](@ref).
Note that if you work with __Sparse__ GPs, the updates should be:
```math
\begin{align*}
    S =& \left(K_{Z}^{-1} + \kappa\operatorname{Diagonal}(r)\kappa^\top\right)^{-1},\\
    m =& S \left(\kappa t + K_Z^{-1}\mu_0(Z)\right),
\end{align*}
```
where $$\kappa=K_{Z}^{-1}K_{Z,X}$$.

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
aug_expected_loglik
aux_prior
```
[`aug_loglik`](@ref) is the equivalent of $$\log p(y|\Omega,f)$$, 
[`aug_expected_loglik`](@ref) is the equivalent of 
$$E_{q(f)q(\Omega)}\left[\log p(y|\Omega,f)\right]$$ and finally, [`aux_prior`](@ref)
will return $$p(\Omega)$$ with the same structure as $$q(\Omega)$$.
