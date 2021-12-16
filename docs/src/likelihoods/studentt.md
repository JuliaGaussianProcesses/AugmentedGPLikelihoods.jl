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

The Student-T distribution is defined as Gaussian scale-mixture with an [Inverse-Gamma](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) prior, i.e.
```math
    \operatorname{Student-t}(y|f,\sigma,\nu) = \int_0^\infty \mathcal{N}(y|f,\sigma^2\omega)\mathcal{IG}(\omega|\frac{\nu}{2},\frac{\nu}{2}),
```
where ``\mathcal{IG}`` is the Inverse-Gamma distribution.
Using the properties that if ``X \sim \mathcal{IG}(a, b)`` then ``c X \sim \mathcal{G}(a, c b)``, we reformulate the scale-mixture as the following augmented likelihood

```math
    p(y,\omega|f,\sigma,\nu) = \mathcal{N}(y|f, \omega)\mathcal{G}\left(\omega|\frac{\nu}{2},\frac{\sigma^2\nu}{2}\right)
```

## Conditional distributions (Sampling)

We are interested in the full-conditionals ``p(f|y,\omega,\sigma,\nu)`` and ``p(\omega|y,f,\sigma,\nu)``:
```math
\begin{align*}
    p(f|y,\omega,\sigma,\nu) =& \mathcal{N}(f|\mu,\Sigma)\\
    \Sigma =& \left(K^{-1} + \operatorname{Diagonal}(\omega^{-1})\right)^{-1}\\
    \mu =& \Sigma\left(\omega^{-1} y + K^{-1}\mu_0\right)\\
    p(\omega_i|y_i,f_i,\sigma,\nu) =& \mathcal{IG}\left(\omega_i|\frac{\nu + 1}{2}, \frac{\sigma^2\nu}{2\sigma^2} + \frac{(y_i - f_i)^2}{2}\right)
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
    \beta_i =& \frac{\sigma^2\nu}{2} + \frac{(y_i - \mu_i)^2 + S_{ii}}{2},\\
    m =& \Sigma\left(\theta y + K^{-1}\mu_0\right),\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(\theta)\right)^{-1},\\
\end{align*}
```
where ``\theta_i = E_{q(\omega_i)}[\omega_i^{-1}] = \alpha_i / \beta_i``.

We get the ELBO as
```math
    \mathcal{L} = -\frac{N}{2}\log (2\pi) + \sum_{i=1}^N -\frac{1}{2} \frac{y_i m_i}{2} + \frac{m_i^2 + S_{ii}}{2}\theta_i - KL(q(\omega)||p(\omega)) - KL(q(f)||p(f)),
```
where
```math
    KL(q(\omega_i|1,c)||p(\omega_i|1,0)) = \log \cosh \left(\frac{c_i}{2}\right) - c_i^2\frac{\theta_i}{2}
```
