@doc raw"""
    init_aux_variables([rng::AbstractRNG], ::Likelihood, n::Int) -> NamedTuple

Initialize collections of `n` auxiliary variables in  a `NamedTuple` to be used
in the context of sampling.
`n` should be the number of data inputs.
See also [`init_aux_posterior`](@ref) for variational inference.
"""
init_aux_variables

@doc raw"""
    init_aux_posterior(::Likelihood, n::Int) -> NamedTuple

Initialize collections of `n` (independent) posteriors for the auxiliary
variables in the context of variational inference.
`n` should be the size of the data used every iteration.
See also [`init_aux_posterior`](@ref) for sampling.
"""
init_aux_posterior


@doc raw"""
    aux_sample!([rng::AbstractRNG], Ω, lik::Likelihood, y, f) -> NamedTuple

Sample the auxiliary variables `Ω` in-place based on the full-conditional
associated with the augmented likelihood:
$$p(\Omega|y,f) \propto p(\Omega,y|f)$$
See also [`aux_sample`](@ref) for an allocating version and [`aux_posterior!`](@ref)
for variational inference.
"""
aux_sample!

@doc raw"""
    aux_sample([rng::AbstractRNG], lik::Likelihood, y, f) -> NamedTuple

Sample and allocate the auxiliary variables `Ω` in a `NamedTuple` based
on the full-conditional associated with the likelihood.
See als [`aux_sample!`](@ref) for an in-place version.
"""
aux_sample


@doc raw"""
    aux_posterior!(qΩ, lik::Likelihood, y, qf) -> NamedTuple

Compute the optimal posterior of the auxiliary variables $$q^*(\Omega)$$ given the marginal
distributions `qf` in-place using the formula
$$q^*(\Omega) \propto \exp\left(E_{q(f)}\left[p(\Omega|f,y)\right]\right)$$
See also [`aux_posterior`](@ref) and [`aux_sample!`](@ref)
"""
aux_posterior!

@doc raw"""
    aux_posterior(lik::Likelihood, y, qf) -> NamedTuple

Compute the optimal posterior of the auxiliary variables in a new
`NamedTuple`.
See also [`aux_posterior!`](@ref)
"""
aux_posterior

@doc raw"""
    auglik_potential(lik::Likelihood, Ω, y) -> Tuple

Given the augmented likelihood $$l(\Omega,y,f) \propto \exp(\beta(\Omega,y) f + \gamma(\Omega,y)f^2)$$,
return the potential, $$\beta(\Omega,y)$$, note that this equivalent to the
shift of the first natural parameter $\eta_1 = \Sigma^{-1}\mu$.
The `Tuple` contains a `Vector` for each latent.
See also [`expected_auglik_potential`](@ref) for variational inference.
"""
auglik_potential

@doc raw"""
    auglik_precision(lik::Likelihood, Ω, y) -> Tuple

Given the augmented likelihood $$l(\Omega,y,f) \propto \exp(\beta(\Omega,y) f + \frac{\gamma(\Omega,y)}{2}f^2)$$,
return the precision, $$\gamma(\Omega,y)$$, note that this equivalent to the
shift of the precision $\Lambda = \Sigma^{-1}$.
The `Tuple` contains a `Vector` for each latent.
See also [`expected_auglik_precision`](@ref) for variational inference.
"""
auglike_precision

@doc raw"""
    expected_auglik_potential(lik::Likelihood, qΩ, y) -> Tuple

Given the augmented likelihood $$l(\Omega,y,f) \propto \exp(\beta(\Omega,y) f + \frac{\gamma(\Omega,y)}{2}f^2)$$,
return the expected potential, $$E_{q(\Omega)}[\beta(\Omega,y)]$$, note that this equivalent to the
shift of the first variational natural parameter $\eta_1 = \Sigma^{-1}\mu$.
The `Tuple` contains a `Vector` for each latent.
See also [`auglik_potential`](@ref) for sampling.
"""
expected_auglik_potential


@doc raw"""
    expected_auglik_precision(lik::Likelihood, qΩ, y) -> Tuple

Given the augmented likelihood $$l(\Omega,y,f) \propto \exp(\beta(\Omega,y) f + \frac{\gamma(\Omega,y)}{2}f^2)$$,
return the expected precision, $$E_{q(\Omega)}[\gamma(\Omega,y)]$$, note that this equivalent to the
shift of the variational precision $\Lambda = \Sigma^{-1}$.
The `Tuple` contains a `Vector` for each latent.
See also [`auglik_precision`](@ref) for sampling.
"""
expected_auglike_precision


@doc raw"""
    aug_loglik(lik::Likelihood, Ω, y, f) -> Real

Return the augmented log-likelihood with the given parameters.
The augmented log-likelihood is of the form 
$$\log p(y,\Omega|f) = \log l(y,\Omega,f) + \log p(\Omega|y).$$
To only obtain $$p(\Omega|y)$$ part see [`aux_prior`](@ref) and see [`logtilt`](@ref)
for $$\log l(y, \Omega, f)$$.
See also [`aug_expected_loglik`](@ref) for the expectation of $$\log p(y,\Omega|f)$$ 
given $$q(f,\Omega)$$.
A generic fallback exists based on [`logtilt`](@ref) and [`aux_prior`](@ref) but
specialized implementations are encouraged
"""
aug_expected_loglik

@doc raw"""
    aug_expected_loglik(lik::Likelihood, qΩ, y, qf) -> Real

Compute the analytical expectation of the augmented likelihood [`aug_loglik`](@ref) given $$q(f)$$ and $$q(\Omega)$$.
As mentionned in [`aug_loglik`](@ref), the prior part ([`aux_prior`](@ref)) is not used.
To compute the KL divergence between $$q(\Omega)$$ and $$p(\Omega)$$ use
[`kl_term`](@ref) instead.
"""
aug_expected_loglik

@doc raw"""
    aux_prior(lik::Likelihood, y) -> NamedTuple

Returns a `NamedTuple` of distributions with the same structure as [`aux_posterior`](@ref),
[`init_aux_posterior`](@ref) and [`init_aux_variables`](@ref).
"""
aux_prior

@doc raw"""
    logtilt(lik::Likelihood, Ω, y, f) -> Real

Compute the quadratic part on $$f$$ of the augmented likelihood:
$$\log C(\Omega, y) + \alpha(\Omega,y) + \beta(\Omega,y)f + \frac{\gamma(\Omega,y)}{2}f^2.$$
See also [`expected_logtilt`](@ref) for variational inference.
"""
logtilt

@doc raw"""
    expected_logtilt(lik::Likelihood, qΩ, y, qf) -> Real

Compute the expectation of the quadratic part on $$f$$ of the augmented likelihood.
$$E_{q(\Omega,f)}\left[\log C(\Omega, y) + \alpha(\Omega,y) + \beta(\Omega,y)f + \frac{\gamma(\Omega,y)}{2}f^2\right].$$
See also [`logtilt`](@ref).
"""
expected_logtilt
