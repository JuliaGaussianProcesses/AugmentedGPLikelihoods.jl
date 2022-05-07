# Negative Binomial Likelihood (Logistic Link)

The [`NegativeBinomialLikelihood`](@ref) with [the failure parametrization](https://en.wikipedia.org/wiki/Negative_binomial_distribution) with a [logistic](https://en.wikipedia.org/wiki/Logistic_function) link ``\sigma`` is defined as

```math
    p(y|f,\text{failures}) = \operatorname{NB}(y|\sigma(f),\text{failures}) = {y + \text{failures} - 1 \choose y} (1 - \sigma(f))^{\text{failures}} \sigma^y(f)
```

To understand more about the different Negative Binomial parametrizations, please [read the `GPLikelihoods.jl` docs](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/stable/api/#GPLikelihoods.NegativeBinomialLikelihood).

To build such a likelihood, you can do `NegativeBinomialLikelihood(NBParamFailures(failures))`.

## The augmentation

Reworking the sigmoids, and using ``r=\text{failures}`` we obtain the following likelihood:

```math
p(y|f,r) = C(y,r) \sigma^r(-f)\sigma^y(f)
```

We are reusing the augmentation from the [Bernoulli Likelihood (Logistic Link)](@ref bernoulli_section) section and use the additivity properties of the Polya-Gamma variables:

```math
\begin{align*}
    p(y|f,r) &= C(y,r)\int_0^\infty \frac{1}{2^r}\exp\left(-\frac{f}{2}r - \frac{f^2}{2}\omega\right)\operatorname{PG}(\omega|r, 0)d\omega\int_0^\infty\frac{1}{2^y}\exp\left(\frac{f}{2}y - \frac{f^2}{2}\omega\right)\operatorname{PG}(\omega|y,0)d\omega\\ 
    &= C(y,r)\int_0^\infty\frac{1}{2^{r+y}}\exp\left(\frac{f}{2}(y-r) -\frac{f^2}{2}\omega\right)\operatorname{PG}(\omega|r+y,0)d\omega,
\end{align*}
```

where ``C(y,r) = {y + r - 1 \choose y}``.
We can now augment with the new variable ``\omega``:

```math
p(y,\omega|f,r) = C(y, r) 2^{-(y+r)}\exp\left(\frac{f}{2}(y-r) -\frac{f^2}{2}\omega\right)\operatorname{PG}(\omega|r+y,0)
```

## Conditional distributions (Sampling)

We are interested in the full-conditionals ``p(f|y,\omega,r)`` and ``p(\omega|y,f,r)``:

```math
\begin{align*}
    p(f|y,\omega) =& \mathcal{N}(f|\mu,\Sigma)\\
    \Sigma =& \left(K^{-1} + \operatorname{Diagonal}(\omega)\right)^{-1}\\
    \mu =& \Sigma\left(\frac{y - r}{2} + K^{-1}\mu_0\right)\\
    p(\omega_i|y_i,f_i) \propto& \exp(-\frac{f_i^2}{2}\omega)\operatorname{PG}(\omega_i|1,0)\\
    =& \operatorname{PG}(\omega_i|y_i+r,|f_i|)
\end{align*}
```

## Variational distributions (Variational Inference)

We define the variational distribution with a block mean-field approximation:

```math
    q(f,\omega) = q(f)\prod_{i=1}^Nq(\omega_i) = \mathcal{N}(f|m,S)\prod_{i=1}^N \operatorname{PG}(\omega_i|y_i + r, c_i).
```

The optimal variational parameters are given by:

```math
\begin{align*}
    c_i =& \sqrt{\mu_i^2 + S_{ii}},\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(\theta)\right)^{-1},\\
    m =& \Sigma\left(\frac{y - r}{2} + K^{-1}\mu_0\right),
\end{align*}
```

where ``\theta_i = E_{q(\omega_i)}[\omega_i] = \frac{y_i + r}{2c_i}\tanh\left(\frac{c_i}{2}\right)``.

We get the ELBO as

```math
    \mathcal{L} = \sum_{i=1}^N -(y_i + r)\log 2 + \frac{(y_i - r) m_i}{2} - \frac{m_i^2 + S_{ii}}{2}\theta_i - \operatorname{KL}(q(\omega)||p(\omega)) - \operatorname{KL}(q(f)||p(f)),
```

where

```math
    \operatorname{KL}(q(\omega_i|y_i+r,c)||p(\omega_i|y_i+r,0)) = (y_i + r) \log \cosh \left(\frac{c_i}{2}\right) - c_i^2\frac{\theta_i}{2}
```
