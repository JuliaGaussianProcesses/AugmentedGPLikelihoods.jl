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
A natural algorithm following from this formulation is the [__Blocked Gibbs Sampling__](https://en.wikipedia.org/wiki/Gibbs_sampling#Blocked_Gibbs_sampler)
but also the Coordinate Ascent VI (CAVI) algorithm where the conditionals are used
to compute the optimal variational distribution.

## What this package does

Although an automatic way is proposed in [galy20](@cite), most of the 
augmentations require some hand derivations.
This package provides the results of these derivations (as well as the derivations)
and propose a unified framework to obtain the distributions of interest.
Generally values (either samples or distributions) of the auxiliary variables need to be carried around.
Since, each likelihood can have a different structure of auxiliary variables, they are uniformly moved as $$\Omega$$, which is a `NamedTuple` containing `Vector`s of either samples or distributions.

### Gibbs Sampling

We give an example in the Gibbs Sampling tutorial using `AbstractMCMC`.
But the API can be reduced to 4 main functions:
```@docs
init_aux_variables
aux_sample!
aux_sample
sample_shift
sample_rate
```
First [`init_aux_variables`](@ref) properly initializes a `NamedTuple` of `Vector`(s) of
auxiliary variables to be modified in place during sampling.
[`aux_sample!`](@ref) will sample in-place from the full conditional $$p(\Omega|y,f)$$ and return another `NamedTuple`, for no-inplace operation see [`aux_sample`].
The full-conditional from $$f$$ are given by
```math
\begin{align*}
    p(f|y,\Omega) =& \mathcal{N}(f|m,S)\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(r)\right)^{-1}\\
    m =& S \left(t + K^{-1}\mu_0\right)
\end{align*}
```
which can be adapted in function of the context.
[`sample_shift`](@ref) returns the `Tuple` representing the `Vector`(s) $$\{t_i\}$$
and [`sample_rate`](@ref) returns the `Tuple` representing the `Vector`(s) $$\{r_i\}$$

The general rule, is that the augmented likelihood will have the form
```math
    p(y|f,\Omega) \propto \exp\left(a(\Omega,y)f + b(\Omega,y)f^2\right),
```
and we have [`sample_shift`](@ref) $$\equiv a(\Omega,y)$$ and [`sample_rate`](@ref) $$\equiv 2b(\Omega,y)$$.

### Coordinate Ascent Variational Inference

CAVI updates work exactly the same way as the Gibbs Sampling, except we 
are now working with posterior distributions instead of samples.
We work with the variational family $$q(f)\prod_{i=1}^N q(\Omega_i)$$, i.e. we make the [__mean-field__](https://en.wikipedia.org/wiki/Mean-field_theory) assumption that our new auxiliary variables are independent of each other and independent of $$f$$.

Similarly there are also 4 main functions

```@docs
init_aux_posterior
aux_posterior!
aux_posterior
vi_shift
vi_rate
```

[`init_aux_posterior`](@ref) initializes the posterior $$\prod_{i=1}^Nq(\Omega_i)$$ as a `NamedTuple`.
[`aux_posterior!`](@ref) updates the variational posterior distributions given the marginals $$q(f)$$.
Finally [`vi_shift`](@ref) and [`vi_rate`](@ref) gives us the elements needed to update the variational distribution $$q(f)$$.
In a similar fashion to [Gibbs Sampling](@ref), we have the following optimal
variational distribution:
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