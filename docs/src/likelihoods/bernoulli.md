# [Bernoulli Likelihood (Logistic Link)](@id bernoulli_section)

The [`BernoulliLikelihood`](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/dev/#GPLikelihoods.BernoulliLikelihood) with a [logistic](https://en.wikipedia.org/wiki/Logistic_function) link ``\sigma`` is defined as

```math
    p(y|f) = \operatorname{Bernoulli}(y|\sigma(f)).
```

In other terms we have that ``p(y=1|f) = \sigma(yf) = \frac{\exp\left(\frac{yf}{2}\right)}{2\cosh\left(\frac{yf}{2}\right)}``,
where we set ``y\in\{-1,1\}``.

## The augmentation

The technique comes from [polson2013bayesian](@cite) and was expanded in [wenzel2019efficient](@cite)

We can rewrite the sigmoid function as:

```math
    \sigma(yf) = \frac{1}{2}\int_0^\infty \exp\left(\frac{yf}{2}-\frac{yf^2}{2}\omega\right)\operatorname{PG}(\omega|1,0)d\omega,
```

where ``\operatorname{PG}(\omega|1,0)`` is the Polya-Gamma distribution.
We can augment the likelihood as:

```math
    p(y,\omega|f) = \frac{1}{2}\exp\left(\frac{yf}{2}-\frac{yf^2}{2}\omega\right)\operatorname{PG}(\omega|1,0).
```

## Conditional distributions (Sampling)

We are interested in the full-conditionals ``p(f|y,\omega)`` and ``p(\omega|y,f)``:

```math
\begin{align*}
    p(f|y,\omega) =& \mathcal{N}(f|\mu,\Sigma)\\
    \Sigma =& \left(K^{-1} + \operatorname{Diagonal}(\omega)\right)^{-1}\\
    \mu =& \Sigma\left(\frac{y}{2} + K^{-1}\mu_0\right)\\
    p(\omega_i|y_i,f_i) \propto& \exp(-\frac{f_i^2}{2}\omega)\operatorname{PG}(\omega_i|1,0)\\
    =& \operatorname{PG}(\omega_i|1,|f_i|)
\end{align*}
```

## Variational distributions (Variational Inference)

We define the variational distribution with a block mean-field approximation:

```math
    q(f,\omega) = q(f)\prod_{i=1}^Nq(\omega_i) = \mathcal{N}(f|m,S)\prod_{i=1}^N \operatorname{PG}(\omega_i|1, c_i).
```

The optimal variational parameters are given by:

```math
\begin{align*}
    c_i =& \sqrt{\mu_i^2 + S_{ii}},\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(\theta)\right)^{-1},\\
    m =& \Sigma\left(\frac{y}{2} + K^{-1}\mu_0\right),
\end{align*}
```

where ``\theta_i = E_{q(\omega_i)}[\omega_i] = \frac{1}{2c_i}\tanh\left(\frac{c_i}{2}\right)``.

We get the ELBO as

```math
    \mathcal{L} = -N\log(2) + \sum_{i=1}^N \frac{y_i m_i}{2} - \frac{m_i^2 + S_{ii}}{2}\theta_i - \operatorname{KL}(q(\omega)||p(\omega)) - \operatorname{KL}(q(f)||p(f)),
```

where

```math
    \operatorname{KL}(q(\omega_i|1,c)||p(\omega_i|1,0)) = \log \cosh \left(\frac{c_i}{2}\right) - c_i^2\frac{\theta_i}{2}
```
