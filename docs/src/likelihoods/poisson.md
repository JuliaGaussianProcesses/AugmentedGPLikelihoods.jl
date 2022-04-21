# Poisson Likelihood (Scaled Logistic Link)

The [`PoissonLikelihood`](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/dev/#GPLikelihoods.PoissonLikelihood) with a scaled [logistic](https://en.wikipedia.org/wiki/Logistic_function) link ``\sigma`` and scaling ``\lambda`` is defined as

```math
    p(y|f) = \operatorname{Po}(y|\lambda\sigma(f)),
```

where ``\operatorname{Po}`` is the [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution)

More explicitly we have:

```math
    p(y|f, \lambda) = \frac{\left(\lambda \sigma(f)\right)^y}{y!}\exp\left(-\lambda \sigma(f)\right)
```

## The augmentation

We first use the symmetry of the logistic function ``\sigma(f) = 1 - \sigma(-f)``:

```math
    p(y|f, \lambda) = \frac{\left(\lambda \sigma(f)\right)^y}{y!}\exp\left(-\lambda (1 - \sigma(f))\right)
```

The last exponential term corresponds to the [probability generating function](https://en.wikipedia.org/wiki/Probability-generating_function) (the discrete version of the moment generating function) of a Poisson distribution with parameter ``\lambda``.
So this can be rewritten as:

```math
    p(y|f, \lambda) = \frac{\left(\lambda \sigma(f)\right)^y}{y!}\sum_{i=1}^\infty \sigma^n(-f)\operatorname{Po}(n|\lambda)
```

We can now augment the likelihood as:

```math
    p(y, n|f, \lambda) = \frac{\left(\lambda \sigma(f)\right)^y}{y!}\sigma^n(-f)\operatorname{Po}(n|\lambda)
```

Like for the [Bernoulli Likelihood](@ref bernoulli_section) work, we can use the Polya-Gamma augmentation, and use the fact that independent Polya-Gamma variables are additive, i.e. if ``\omega_1 \sim \operatorname{PG}(a, 0)`` and ``\omega_2 \sim \operatorname{PG}(b, 0)`` then ``\omega_1 + \omega_2 \sim \operatorname{PG}(a + b, 0)``
This results in the final augmented likelihood

```math
    p(y, n, \omega| f, \lambda) = \lambda^y\left(2^{y + n}y!\right)^{-1}\exp\left(\frac{(y-n)}{2}f - \frac{f^2}{2}\omega\right)\operatorname{PG}(\omega|y+n, 0)\operatorname{Po}(n|\lambda)
```

## Conditional distributions (Sampling)

We are interested in the full-conditionals ``p(f|y,\omega, n, \lambda)`` and ``p(\omega, n|y,f, \lambda)``:

```math
\begin{align*}
    p(f|y,\omega, n, \lambda) =& \mathcal{N}(f|\mu,\Sigma)\\
    \Sigma =& \left(K^{-1} + \operatorname{Diagonal}(\omega)\right)^{-1}\\
    \mu =& \Sigma\left(\frac{y - n}{2} + K^{-1}\mu_0\right)\\
    p(\omega_i, n_i|y_i,f_i,\lambda) =& \operatorname{PG}(\omega_i|y_i + n_i, |f_i|)\operatorname{Po}(n_i|\lambda\sigma(f_i))
\end{align*}
```

Note that ``p(\omega,n |y,f,\lambda)`` is defined in the package as a [`PolyaGammaPoisson`](@ref) distribution.

## Variational distributions (Variational Inference)

We define the variational distribution with a block mean-field approximation:

```math
    q(f,\omega, n) = q(f)\prod_{i=1}^Nq(\omega_i, n_i) = \mathcal{N}(f|m,S)\prod_{i=1}^N \operatorname{PG}(\omega_i|y + \gamma_i, c_i)\operatorname{Po}(n_i|\gamma_i).
```

The optimal variational parameters are given by:

```math
\begin{align*}
    c_i =& \sqrt{\mu_i^2 + S_{ii}},\\
    \gamma_i =& \frac{\exp(-\frac{\mu_i}{2})}{2\cosh(\frac{c_i}{2})},\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(\theta)\right)^{-1},\\
    m =& \Sigma\left(\frac{y}{2} + K^{-1}\mu_0\right),
\end{align*}
```

where ``\theta_i = E_{q(\omega_i,n_i)}[\omega_i] = \frac{y+\gamma_i}{2c_i}\tanh\left(\frac{c_i}{2}\right)``.

We get the ELBO as

```math
\begin{align*}
    \mathcal{L} =& \sum_{i=1}^N -(y_i + \gamma_i) \log 2+ y_i \log \lambda + \frac{(y_i - \gamma_i) m_i}{2} - \frac{m_i^2 + S_{ii}}{2}\theta_i\\ 
    &- \operatorname{KL}(q(\omega,n)||p(\omega,n|y)) - \operatorname{KL}(q(f)||p(f)),
\end{align*}
```

where

```math
\begin{align*}
    \operatorname{KL}(q(\omega_i|n_i)q(n_i)||p(\omega_i|n_i, y)p(n_i)) =& \operatorname{KL}(q(n_i)||p(n_i)) + E_{q(n_i)}\left[\operatorname{KL}(q(\omega_i|n_i)||p(\omega_i|n_i, y)\right],\\
    \operatorname{KL}(q(n_i)||p(n_i)) =& \lambda - \gamma_i + \gamma_i \log \frac{\gamma_i}{\lambda},\\
    E_{q(n_i)}\left[\operatorname{KL}(q(\omega_i|n_i)||p(\omega_i|n_i, y)\right] =& (y_i + \gamma_i)\log\cosh \left(\frac{c_i}{2}\right) - c_i^2 \frac{\theta_i}{2}.
\end{align*}
```
