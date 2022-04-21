@doc raw"""
    init_aux_variables([rng::AbstractRNG], ::Likelihood, n::Int) -> TupleVector

Initialize collections of `n` auxiliary variables in  a `TupleVector` to be used
in the context of sampling.
`n` should be the number of data inputs.

See also [`init_aux_posterior`](@ref) for variational inference.
"""
init_aux_variables

@doc raw"""
    init_aux_posterior([T::DataType], lik::Likelihood, n::Int) -> AbstractProductMeasure

Initialize collections of `n` (independent) posteriors for the auxiliary
variables in the context of variational inference.
`n` should be the size of the data used every iteration.
The real variational parameters will be given type `T` (`Float64` by default)

See also [`init_aux_variables`](@ref) for sampling.
"""
init_aux_posterior

@doc raw"""
    aux_full_conditional(lik::AbstractLikelihood, yᵢ, fᵢ) -> AbstractNTDist

Return the full conditional distribution of `Ωᵢ`.
"""
aux_full_conditional

@doc raw"""
    aux_sample!([rng::AbstractRNG], Ω, lik::Likelihood, y, f) -> TupleVector

Sample the auxiliary variables `Ω` in-place based on the full-conditional [`aux_full_conditional`](@ref)
associated with the augmented likelihood:
```math
    p(\Omega|y,f) \propto p(\Omega,y|f).
```

See also [`aux_sample`](@ref) for an allocating version and [`aux_posterior!`](@ref)
for variational inference.
"""
aux_sample!

@doc raw"""
    aux_sample([rng::AbstractRNG], lik::Likelihood, y, f) -> TupleVector

Sample and allocate the auxiliary variables `Ω` in a `TupleVector` based
on the full-conditional associated with the likelihood.

See [`aux_sample!`](@ref) for an in-place version.
"""
aux_sample

@doc raw"""
    aux_full_conditional(lik::Likelihood, y, f::Real) -> Distribution

Given the observation `y` and latent `f`, returns the full conditional on the
auxiliary variables `Ω`.
"""

@doc raw"""
    aux_posterior!(qΩ, lik::Likelihood, y, qf) -> AbstractProductMeasure

Compute the optimal posterior of the auxiliary variables ``q^*(\Omega)`` given 
the marginal distributions `qf` by updating the variational parameters
in-place using the formula
```math
    q^*(\Omega) \propto \exp\left(E_{q(f)}\left[p(\Omega|f,y)\right]\right)
```

See also [`aux_posterior`](@ref) and [`aux_sample!`](@ref)
"""
aux_posterior!

@doc raw"""
    aux_posterior(lik::Likelihood, y, qf) -> AbstractProductMeasure

Compute the optimal posterior of the auxiliary variables in a new
`AbstractProductMeasure` (`For` by default).

See [`aux_posterior!`](@ref) for more details
"""
aux_posterior

@doc raw"""
    auglik_potential(lik::Likelihood, Ω, y) -> Tuple

Given the augmented likelihood ``l(\Omega,y,f) \propto \exp(\beta(\Omega,y) f + \gamma(\Omega,y)f^2)``,
return the potential, ``\beta(\Omega,y)``.
Note that this equivalent to the shift of the first natural parameter ``\eta_1 = \Sigma^{-1}\mu``.
The `Tuple` contains a `Vector` for each latent.

See also [`expected_auglik_potential`](@ref) for variational inference.
"""
auglik_potential

@doc raw"""
    auglik_precision(lik::Likelihood, Ω, y) -> Tuple

Given the augmented likelihood ``l(\Omega,y,f) \propto \exp(\beta(\Omega,y) f + \frac{\gamma(\Omega,y)}{2}f^2)``,
return the precision, $$\gamma(\Omega,y)$$, note that this equivalent to the
shift of the precision ``\Lambda = \Sigma^{-1}``.
The `Tuple` contains a `Vector` for each latent.

See also [`expected_auglik_precision`](@ref) for variational inference.
"""
auglik_precision

@doc raw"""
    auglik_potential_and_precision(lik::Likelihood, Ω, y) -> Tuple{Tuple, Tuple}

Returns both [`auglik_potential`](@ref) and [`auglik_precision`](@ref) when some 
computation can be saved by doing both at the same time.
"""
auglik_potential_and_precision

@doc raw"""
    expected_auglik_potential(lik::Likelihood, qΩ, y) -> Tuple

Given the augmented likelihood ``l(\Omega,y,f) \propto \exp(\beta(\Omega,y) f + \frac{\gamma(\Omega,y)}{2}f^2)``,
return the expected potential, ``E_{q(\Omega)}[\beta(\Omega,y)]``, note that this equivalent to the
shift of the first variational natural parameter ``\eta_1 = \Sigma^{-1}\mu``.
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
expected_auglik_precision

@doc raw"""
    expected_auglik_potential_and_precision(lik::Likelihood, Ω, y) -> Tuple{Tuple, Tuple}

Returns both [`expected_auglik_potential`](@ref) and [`expected_auglik_precision`](@ref) when some 
computation can be saved by doing both at the same time.
"""
expected_auglik_potential_and_precision

@doc raw"""
    aug_loglik(lik::Likelihood, Ω, y, f) -> Real

Return the augmented log-likelihood with the given parameters.
The augmented log-likelihood is of the form 
```math
    \log p(y,\Omega|f) = \log l(y,\Omega,f) + \log p(\Omega|y).
```
To only obtain the $$p(\Omega|y)$$ part see [`aux_prior`](@ref) and see [`logtilt`](@ref)
for $$\log l(y, \Omega, f)$$.

A generic fallback exists based on [`logtilt`](@ref) and [`aux_prior`](@ref) but
specialized implementations are encouraged.
"""
aug_loglik

@doc raw"""
    aux_kldivergence(lik::Likelihood, qΩ::For, pΩ::For) -> Real
    aux_kldivergence(lik::Likelihood, qΩ::For, y) -> Real

Compute the analytical KL divergence between the auxiliary variables posterior
``q(\Omega)``, obtained with [`aux_posterior`](@ref) and prior
``p(\Omega)``, obtained with [`aux_prior`](@ref).
"""
aux_kldivergence

@doc raw"""
    aux_prior(lik::Likelihood, y) -> AbstractProductMeasure

Returns a `NamedTuple` of distributions with the same structure as [`aux_posterior`](@ref),
[`init_aux_posterior`](@ref) and [`init_aux_variables`](@ref).
"""
aux_prior

@doc raw"""
    logtilt(lik::Likelihood, Ω, y, f) -> Real

Compute the quadratic part on ``f`` of the augmented likelihood:
```math
    \log C(\Omega, y) + \alpha(\Omega,y) + \beta(\Omega,y)f + \frac{\gamma(\Omega,y)}{2}f^2.
```

See also [`expected_logtilt`](@ref) for variational inference.
"""
logtilt

@doc raw"""
    expected_logtilt(lik::Likelihood, qΩ, y, qf) -> Real

Compute the expectation of the quadratic part on $$f$$ of the augmented likelihood.
```math
    E_{q(\Omega,f)}\left[\log C(\Omega, y) + \alpha(\Omega,y) + \beta(\Omega,y)f + \frac{\gamma(\Omega,y)}{2}f^2\right].
```

See also [`logtilt`](@ref).
"""
function expected_logtilt(lik::AbstractLikelihood, qΩ, y, qf::AbstractVector{<:Normal})
    return mapreduce(+, y, qf, @ignore_derivatives marginals(qΩ)) do yᵢ, qfᵢ, qΩᵢ
        expected_logtilt(lik, qΩᵢ, yᵢ, qfᵢ)
    end
end
