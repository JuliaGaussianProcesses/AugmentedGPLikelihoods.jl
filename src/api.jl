"""
    init_aux_variables([rng::AbstractRNG], ::Likelihood, n::Int) -> NamedTuple

Initialize the `n` auxiliary variables in the form of a `NamedTuple`
in the context of sampling.
`n` should be the size of the data used every iteration.
See also [`init_aux_posterior`](@ref)
"""
init_aux_variables

"""
    init_aux_posterior(::Likelihood, n::Int) -> NamedTuple

Initialize the series of `n` (independent) posteriors for the auxiliary
variables in the context of variational inference.
`n` should be the size of the data used every iteration.
See also [`init_aux_posterior`](@ref)
"""
init_aux_posterior


"""
    aux_sample!([rng::AbstractRNG], Ω, lik::Likelihood, y, f) -> NamedTuple

Sample the auxiliary variables `Ω` inplace based on the full-conditional
associated with the likelihood and return `Ω`
See also [`aux_sample`](@ref) and [`aux_posterior!`](@ref)
"""
aux_sample!

"""
    aux_sample([rng::AbstractRNG], lik::Likelihood, y, f) -> NamedTuple

Sample the auxiliary variables `Ω` in a new `NamedTuple`.
For the explanation see [`aux_sample!`].
"""
aux_sample


"""
    aux_posterior!(Ω, lik::Likelihood, y, q_f) -> NamedTuple

Compute the optimal posterior of the auxiliary variables given the marginal
distributions `q_f` and replace the existing ones.
See also [`aux_posterior`](@ref) and [`aux_sample!`](@ref)
"""
aux_posterior!

"""
    aux_posterior(lik::Likelihood, y, q_f) -> NamedTuple

Compute the optimal posterior of the auxiliary variables in a new
`NamedTuple`.
See also [`aux_posterior!`](@ref)
"""
aux_posterior

@doc raw"""
    sample_shift(lik::Likelihood, Ω, y) -> Tuple

Return the shift of the first variational natural parameter
$\eta_1=\Sigma^{-1}\mu$ given the auxiliary variables in `\Omega`
and the observations `y`
"""
sample_shift

@doc raw"""
    sample_rate(lik::Likelihood, Ω, y) -> Tuple

Return two times the shift of the second natural parameter
$\eta_2=-\frac{1}{2}\Sigma^{-1}$ given the auxiliary variables
in `\Omega` and the observations `y`
"""
sample_rate

@doc raw"""
    vi_shift(lik::Likelihood, Ω, y) -> Tuple

Return the shift of the first natural parameter $\eta_1=\Sigma^{-1}\mu$ given the auxiliary
posteriors contained in `\Omega` and the observations `y`
"""
vi_shift


@doc raw"""
    vi_rate(lik::Likelihood, Ω, y) -> Tuple

Return two times the shift of the second natural parameter $\eta_2=-\frac{1}{2}\Sigma^{-1}$
given the auxiliary posteriors contained in `\Omega` and the observations `y`
"""
vi_rate
