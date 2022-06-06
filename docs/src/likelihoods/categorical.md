# Categorical Likelihood (LogisticSoftmax Link)

The [`CategoricalLikelihood`](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/stable/api/#GPLikelihoods.CategoricalLikelihood) with ``K`` categories with a logistic-softmax link is defined as

```math
p(y=k|\{f_j\}_{j=1}^K) = \frac{\theta_k\sigma(f_k)}{\sum_{j=1}^K \theta_j\sigma(f_j)},
```

Note that two versions are possible:

- The bijective one where we have ``K-1`` latent Gaussian Processes and set the last ``f_K`` to a fixed value ``C``.
- The non-bijective, or over-parametrized, version where we have ``K`` latent GPs.

To call:

- The first one, build: `CategoricalLikelihood(BijectiveSimplex(LogisticSoftmaxLink(zeros(nclass))))`
- The second one : `CategoricalLikelihood(LogisticSoftmaxLink(zeros(nclass)))`.

For ease of computation, we one-hot encode the labels as ``\boldsymbol{Y}`` where ``Y_j^i = y^i == j``.

## The augmentation

### Bijective version

We have ``\sigma(f_K) = \sigma(C) = D \in [0, 1]``.

```math
\begin{align*}
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K-1},\bf \theta) =& \frac{\theta_k\sigma(f^i_k)}{\theta_K D + \sum_{j=1}^{K-1}\theta_j\sigma(f^i_j)}\\
    =&\frac{\theta_k\sigma(f^i_k)}{D}\frac{1}{1 + \frac{1}{\theta_K D} \sum_{j=1}^{K-1}\theta_j\sigma(f^i_j)}\\
    =& \frac{\sigma(f^i_k)}{\theta_k D}\int_{0}^\infty \exp(-\lambda \sum_{j=1}^{K-1}\theta_j\sigma(f^i_j))\operatorname{Ga}(\lambda|1,\frac{1}{\theta_K D})d\lambda\\
    p(y^i=k|\{f^i_j\}_{j=1}^{K-1},\lambda,\bf \theta) =& \theta_k\sigma(f^i_k)\exp(\lambda \sum_{i=j}^{K-1}\theta_j\sigma(f^i_j))\operatorname{Ga}(\lambda|1,\frac{1}{\theta_K D})\\
    =& \theta_k \sigma(f^i_k)\prod_{j=1}^{K-1}\sum_{n^i_j=0}^\infty\sigma^{n^i_j}(-f^i_j)\operatorname{Po}(n^i_j|\theta_j\lambda)\operatorname{Ga}(\lambda|1,\frac{1}{\theta_K D})\\
    p(y=k|\{\boldsymbol{f}_j\}_{j=1}^{K-1},\lambda, \{n^i_j\}_{j=1}^{K-1}) =&\theta_k\sigma(f^i_k)\prod_{j=1}^{K-1}\sigma^{n^i_j}(-f^i_j)\operatorname{Po}(n^i_j|\theta_j\lambda)\operatorname{Ga}(\lambda|1,\frac{1}{\theta_K D})
\end{align*}
```

We could continue with this, but one can actually notice that a product of independent Poisson variables with a Gamma prior will produce a Negative Multinomial variable by marginalizing out ``\lambda``:

```math
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K-1},\boldsymbol{n}^i) =\sigma(f^i_k)\prod_{j=1}^{K-1}\sigma^{n^i_j}(-f^i_j)\operatorname{NM}(\boldsymbol{n}^i|1, \left\{\frac{\theta_j}{\theta_K D + \sum_{l=1}^{K-1} \theta_l}\right\}_{j=1}^{K-1})
```

It's one less variable!
Now we can easily perform the last augmentation using the Pólya-Gamma augmentations.

```math
\begin{align*}
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K=1},\boldsymbol{n}^i, \boldsymbol{\omega}^i,\bf \theta) =&\operatorname{NM}(\boldsymbol{n}^i|1, \left\{\frac{\theta_j}{D + \sum_{l=1}^{K=1}\theta_l}\right\}_{j=1}^{K-1})\\
    &\times \prod_{j=1}^{K-1}2^{-(y^i_j + n^i_j)}e^{\frac{1}{2}\left((y^i_j - n^i_j) f^i_j - \omega^i_j (f^i_j)^2\right)}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j,0)
\end{align*}
```

### Non-bijective version

This time there is no constant to take care of!

```math
\begin{align*}
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K},\bf \theta) =& \frac{\theta_k\sigma(f^i_k)}{\sum_{i=1}^{K}\theta_j\sigma(f^i_j)}\\
    =&\theta_k\sigma(f^i_k)\int_0^\infty e^{-\lambda \sum_{j=1}^{K}\theta_j\sigma(f^i_j)}d\lambda\\
    p(y^i=k|\{f^i_j\}_{j=1}^{K},\bf \theta, \lambda) =& \theta_k\sigma(f^i_k)\exp(\lambda \sum_{j=1}^{K-1}\theta_j\sigma(f^i_j))\\
    =& \theta_k\sigma(f^i_k)\prod_{j=1}^{K}\sum_{n^i_j=0}^\infty\sigma^{n^i_j}(-f^i_j)\operatorname{Po}(n^i_j|\theta_j\lambda)\\
    p(y=k|\{\boldsymbol{f}_j\}_{j=1}^{K}, \bf \theta, \lambda, \{n^i_j\}_{j=1}^{K}) =&\theta_k \sigma(f^i_k)\prod_{j=1}^{K}\sigma^{n^i_j}(-f^i_j)\operatorname{Po}(n^i_j|\theta_j\lambda)
\end{align*}
```

Note that as opposed to the bijective version, the prior on ``\lambda`` is the improper prior ``1_{[0, \infty]}``.
Similarly to the bijective version, we can also recover a Negative Multinomial

```math
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K},\boldsymbol{n}^i) =\sigma(f^i_k)\prod_{j=1}^K \widetilde{\operatorname{NM}}(\boldsymbol{n}^i|1, \left\{\frac{\theta_j\sigma(-f_j^i)}{\sum_j \theta_j}\right\}_{j=1}^{K})
```

``\widetilde{\operatorname{NM}}`` is normalizable but since ``p_0 = 1 - \sum_j \frac{\sigma(-f_j^i)}{K}`` we simply consider the non-normalized version.
For the last step:

```math
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K},\boldsymbol{n}^i, \boldsymbol{\omega}^i) =\widetilde{\operatorname{NM}}(\boldsymbol{n}^i|1, \left\{\frac{\theta_j}{\sum_l \theta_l}\right\}_{j=1}^{K})\prod_{j=1}^{K}2^{-(y^i_j + n^i_j)}e^{\frac{1}{2}\left((y^i_j - n^i_j) f^i_j - \omega^i_j (f^i_j)^2\right)}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j,0),
```

where we extracted the ``\sigma(-f_j^i)`` terms from the Negative Multinomial.
Note that it is the same as the bijective version but with a different prior on ``\boldsymbol{n}``.

## Conditional distributions (Sampling)

Let's define the set of all variables ``\boldsymbol{F} = \{\boldsymbol{f}_j\}_{j=1}^{K}``, ``\boldsymbol{N} = \{\boldsymbol{n}^i\}_{i=1}^N`` and ``\boldsymbol{\Omega} = \{\boldsymbol{\omega}^i\}_{i=1}^N``.

We are interested in the full-conditionals ``p(\boldsymbol{f}_j|\boldsymbol{Y}, \boldsymbol{\Omega}, \boldsymbol{N})`` and ``p(\boldsymbol{\omega}^i, \boldsymbol{n}^i|\boldsymbol{Y},\boldsymbol{F})``

```math
\begin{align*}
    p(\boldsymbol{f}_j|\boldsymbol{Y},\boldsymbol{\Omega}, \boldsymbol{N}) =& \mathcal{N}(\boldsymbol{f}_j|\mu_j,\Sigma_j)\\
    \Sigma_j =& \left(K^{-1} + \operatorname{Diagonal}(\boldsymbol{\omega}_j)\right)^{-1}\\
    \mu_j =& \Sigma_j\left(\frac{\boldsymbol{y}_j - \boldsymbol{n}_j}{2} + K^{-1}\boldsymbol{\mu}_0\right)\\
\end{align*}
```

For the **bijective version**

```math
\begin{align*}
p(\boldsymbol{\omega}^i, \boldsymbol{n}^i|\boldsymbol{y}^i, \boldsymbol{F}) =& \prod_{j=1}^{K-1}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j, |f^i_j|)\operatorname{NM}\left(\boldsymbol{n}^i|1, \left\{\frac{\sigma(-f^i_j)}{D + K - 1}\right\}_{j=1}^{K-1}\right)
\end{align*}
```

For the **non-bijective version**

```math
\begin{align*}
    p(\boldsymbol{\omega}^i, \boldsymbol{n}^i|\boldsymbol{y}^i, \boldsymbol{F}) =& \prod_{j=1}^{K-1}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j, |f^i_j|)\operatorname{NM}\left(\boldsymbol{n}^i|1, \left\{\frac{\sigma(-f^i_j)}{K}\right\}_{j=1}^{K-1}\right)
\end{align*}
```

Note that ``p(\boldsymbol{\omega}^i, \boldsymbol{n}^i|
\boldsymbol{Y},\boldsymbol{F})`` is defined in the package as a [`AugmentedGPLikelihoods.SpecialDistributions.PolyaGammaNegativeMultinomial`](@ref) distribution.

## Variational distributions (Variational Inference)

We define the variational distribution with a block mean-field approximation:

```math
    q(\boldsymbol{F}, \boldsymbol{\Omega}, \boldsymbol{N}) = \prod_{j=1}^{K-1} \mathcal{N}(\boldsymbol{f}_j|\boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)\prod_{i=1}^N \operatorname{NM}(\boldsymbol{n}^i|1, \boldsymbol{p}^i)\prod_{j=1}^{K-1}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j, c^i_j).
```

The optimal variational parameters are given by:

```math
\begin{align*}
    c^i_j =& \sqrt{(\mu^i_j)^2 + S^{ii}_j},\\
    \Sigma_j =& \left(K^{-1} + \operatorname{Diagonal}(\theta_j)\right)^{-1},\\
    \mu_j =& \Sigma_j\left(\frac{y - \gamma_j}{2} + K^{-1}\mu_0\right),
\end{align*}
```

where ``\gamma_j^i = E_{q(\boldsymbol{p})}[n_j^i] = \frac{p_j^i}{1 - \sum_{j=1}^{K-1} p_j^i}``, ``\theta_j^i = E_{q(\omega^i_j,n^i_j)}[\omega_j^i] = \frac{y_j^i+\gamma^i_j}{2c_j^i}\tanh\left(\frac{c_j^i}{2}\right)`` and ``\widetilde{\sigma}(q(f_j^i)) = \frac{e^{-\mu_j^i}/2}{2\cosh(c_j^i/2)}`` which is an approximation of the expectation of ``E_{q(f_j^i)}[\sigma(-f_j^i)]``.

For the **bijective version**

```math
p^i_j = \frac{\widetilde{\sigma}(q(f_i^j))}{D + K - 1},
```

and for the **non-bijective version**

```math
p^i_j = \frac{\widetilde{\sigma}(q(f_i^j))}{K},
```

We get the ELBO as

```math
\begin{align*}
    \mathcal{L} =& \sum_{i=1}^N\sum_{j=1}^K -  (y^i + \gamma^i_j) \log 2 + \frac{(y_j^i - \gamma^i_j) m^i_j}{2} - \frac{(m^i_j)^2 + S_j^{ii}}{2}\theta^i_j\\ 
    &- \operatorname{KL}(q(\boldsymbol{\Omega},\boldsymbol{N})||p(\boldsymbol{\Omega},\boldsymbol{N}|\boldsymbol{Y})) - \operatorname{KL}(q(\boldsymbol{F})||p(\boldsymbol{F})),
\end{align*}
```

where

```math
\begin{align*}
    \operatorname{KL}(\prod_{j} q(\omega^i_j|n^i_j)q(\bm n^i)||\prod_j p(\omega^i_j|\bm n^i, \boldsymbol{y}^i)p(\bm n^i)) =& \operatorname{KL}(q(\bm n^i)||p(\bm n^i)) + \sum_j E_{q(\bm n^i)}\left[\operatorname{KL}(q(\omega^i_j|\bm n^i)||p(\omega^i_j|\bm n^i, \boldsymbol{y}^i)\right],\\
    E_{q(\bm n^i)}\left[\operatorname{KL}(q(\omega^i_j|\bm n^i)||p(\omega^i_j|n^i_j, y^i)\right] =& (y^i + \gamma^i_j)\log\cosh \left(\frac{c^i_j}{2}\right) - (c^i_j)^2 \frac{\theta^i_j}{2}.
\end{align*}
```

For the **bijective version**

```math
\operatorname{KL}(q(\bm n_i)||p(\bm n_i)) = \log p_0^q - \log p_0^p + \sum_j \gamma_i (\log p_i^q - \log p_i^p),
```

while for the **non-bijective version**

```math
\operatorname{KL}(q(\bm n_i)||\tilde{p}(\bm n_i)) = \log p_0^q + \sum_j \gamma_i (\log p_i^q - \log p_i^p),
```

Note that we ignored the ``p_0^p`` term since we were working with the unnormalized version of the distribution.
