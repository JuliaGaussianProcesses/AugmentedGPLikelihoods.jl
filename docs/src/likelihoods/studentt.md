# Student-T Likelihood

The [`StudentTLikelihood`](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/dev/#GPLikelihoods.BernoulliLikelihood) with a [logistic](https://en.wikipedia.org/wiki/Logistic_function) link ``\sigma`` is defined as

```math
\begin{aligned}
    p(y|f) =& \operatorname{Student-t}(y|f,\sigma,\nu)\\
    =& \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)\sqrt{\pi\nu}\sigma}\left(1 + \frac{1}{\nu}\left(\frac{x-\nu}{\sigma}\right)^2\right)^{-\frac{\nu+1}{2}}.
\end{aligned}
```

## The augmentation

We use the approach introduced by [liu1995ml](@cite), but also in [jylanki2011robust](@cite) and [galy20](@cite).

The Student-T distribution is defined as Gaussian scale-mixture with a [Gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) prior on the precision, i.e.

```math
    \operatorname{Student-t}(y|f,\sigma,\nu) = \int_0^\infty \mathcal{N}(y|f,\sigma^2\omega^{-1})\mathcal{G}(\omega|\frac{\nu}{2},\frac{\nu}{2}),
```

where ``\mathcal{G}`` is the Gamma distribution.
Using the properties that if ``X \sim \mathcal{G}(a, b)`` then ``c X \sim \mathcal{G}(a, c b)``, we reformulate the scale-mixture as the following augmented likelihood

```math
    p(y,\omega|f,\sigma,\nu) = \mathcal{N}(y|f, \omega^{-1})\mathcal{G}\left(\omega|\frac{\nu}{2},\frac{\nu}{2\sigma^2}\right)
```

## Conditional distributions (Sampling)

We are interested in the full-conditionals ``p(f|y,\omega,\sigma,\nu)`` and ``p(\omega|y,f,\sigma,\nu)``:

```math
\begin{align*}
    p(f|y,\omega,\sigma,\nu) =& \mathcal{N}(f|\mu,\Sigma)\\
    \Sigma =& \left(K^{-1} + \operatorname{Diagonal}(\omega^{-1})\right)^{-1}\\
    \mu =& \Sigma\left(\omega^{-1} y + K^{-1}\mu_0\right)\\
    p(\omega_i|y_i,f_i,\sigma,\nu) =& \mathcal{G}\left(\omega_i|\frac{\nu + 1}{2}, \frac{\nu}{2\sigma^2} + \frac{(y_i - f_i)^2}{2}\right)
\end{align*}
```

## Variational distributions (Variational Inference)

We define the variational distribution with a block mean-field approximation:

```math
    q(f,\omega) = q(f)\prod_{i=1}^Nq(\omega_i) = \mathcal{N}(f|m,S)\prod_{i=1}^N \mathcal{G}(\omega_i|\alpha, \beta_i),
```

where ``\alpha=\frac{\nu+1}{2}`` for all ``i``.
The optimal variational parameters are given by:

```math
\begin{align*}
    \beta_i =& \frac{\nu}{2\sigma^2} + \frac{(y_i - \mu_i)^2 + S_{ii}}{2},\\
    m =& \Sigma\left(\theta y + K^{-1}\mu_0\right),\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(\theta)\right)^{-1},
\end{align*}
```

where ``\theta_i = E_{q(\omega_i)}[\omega_i] = \alpha_i / \beta_i``.

We get the ELBO as

```math
    \mathcal{L} = -\frac{N}{2}\log (2\pi) + \sum_{i=1}^N \frac{1}{2}E_{q(\omega_i)}[-\log \omega_i] -\frac{1}{2} \left(-\frac{(y_i-m_i)^2 + S_{ii}}{2}\theta_i} - \operatorname{KL}(q(\omega)||p(\omega)) - \operatorname{KL}(q(f)||p(f)),
```

where

```math
\begin{align*}
    E_{q(\omega_i)}\left[-\log\omega_i\right] =& \alpha_i + \log (\beta_i(\Gamma(\alpha_i))) - (\alpha_i + 1)\psi(\alpha_i)\\
    \operatorname{KL}(q(\omega_i|\alpha_i,\beta_i)||p(\omega_i|\alpha,\beta)) =& (\alpha_i - \alpha)\psi(\alpha_i) - \log\Gamma(\alpha_i) + \log \Gamma(\alpha) + \alpha(\log \beta_i - \log \beta) + \alpha_i \frac{\beta - \beta_i}{\beta_i}
\end{align*}
```

where ``\psi(\alpha)`` is the [digamma function](https://en.wikipedia.org/wiki/Digamma_function).
