# Categorical Likelihood (LogisticSoftmax Link)

The [`CategoricalLikelihood`](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/stable/api/#GPLikelihoods.CategoricalLikelihood) with ``K`` categories with a logistic-softmax link is defined as
```math
    p(y=k|\{f_j\}_{j=1}^K) = \frac{\sigma(f_k)}{\sum_{j=1}^K \sigma(f_j)},
```
Note that two versions are possible:
- The bijective one where we have ``K-1`` latent Gaussian Processes and set the last ``f_K`` to a fixed value ``C``.
- The non-bijective, or over-parametrized, version where we have ``K`` latent GPs.

To call the first one, build: `CategoricalLikelihood(nclass, BijectiveSimplex(LogisticSoftmaxLink))` for the second one : `CategoricalLikelihood(nclass, LogisticSoftmaxLink)`.

For ease of computation, we one-hot encode the labels as ``\boldsymbol{Y}`` where ``Y_j^i = y^i == j``.
## The augmentation
### Bijective version

We have ``\sigma(f_K) = \sigma(C) = D \in [0, 1]``.
```math
\begin{align*}
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K-1}) =& \frac{\sigma(f^i_k)}{D + \sum_{i=1}^{K-1}\sigma(f^i_j)}\\
    =&\frac{\sigma(f^i_k)}{D}\frac{1}{1 + \frac{1}{D} \sum_{i=1}^{K-1}\sigma(f^i_j)}\\
    =& \sigma(f^i_k)\int_{0}^\infty \exp(-\lambda \sum_{i=1}^{K-1}\sigma(f^i_j))\operatorname{Ga}(\lambda|1,\frac{1}{D})d\lambda\\
    p(y^i=k|\{f^i_j\}_{j=1}^{K-1},\lambda) =& \sigma(f^i_k)\exp(\lambda \sum_{i=1}^{K-1}\sigma(f^i_j))\operatorname{Ga}(\lambda|1,\frac{1}{D})\\
    =& \sigma(f^i_k)\prod_{j=1}^{K-1}\sum_{n^i_j=0}^\infty\sigma^{n^i_j}(-f^i_j)\operatorname{Po}(n^i_j|\lambda)\operatorname{Ga}(\lambda|1,\frac{1}{D})\\
    p(y=k|\{\boldsymbol{f}_j\}_{j=1}^{K-1},\lambda, \{n^i_j\}_{j=1}^{K-1}) =&\sigma(f^i_k)\prod_{j=1}^{K-1}\sigma^{n^i_j}(-f^i_j)\operatorname{Po}(n^i_j|\lambda)\operatorname{Ga}(\lambda|1,\frac{1}{D})
\end{align*}
```
We could continue with this, but one can actually notice that a product of independent Poisson variables with a Gamma prior will produce a Negative Multinomial variable by marginalizing out ``\lambda``:
```math
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K-1},\boldsymbol{n}^i) =\sigma(f^i_k)\prod_{j=1}^{K-1}\sigma^{n^i_j}(-f^i_j)\operatorname{NM}(\boldsymbol{n}^i|1, \left\{\frac{1}{D + K - 1}\right\}_{j=1}^{K-1})
```

It's one less variable!
Now we can easily perform the last augmentation using the Pólya-Gamma augmentations.
```math
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K=1},\boldsymbol{n}^i, \boldsymbol{\omega}^i) =\operatorname{NM}(\boldsymbol{n}^i|1, \{\frac{1}{D + K - 1}\}_{j=1}^{K-1})\prod_{j=1}^{K-1}2^{-(y^i_j + n^i_j)}e^{\frac{1}{2}\left((y^i_j - n^i_j) f^i_j - \omega^i_j (f^i_j)^2\right)}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j,0)
```

### Non-bijective version

This time there is no constant to take care of!

```math
\begin{align*}
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K}) =& \frac{\sigma(f^i_k)}{\sum_{i=1}^{K}\sigma(f^i_j)}\\
    =&\sigma(f^i_k)\int_0^\infty e^{-\lambda \sum_{i=1}^{K}\sigma(f^i_j)}d\lambda\\
    p(y^i=k|\{f^i_j\}_{j=1}^{K},\lambda) =& \sigma(f^i_k)\exp(\lambda \sum_{i=1}^{K-1}\sigma(f^i_j))\\
    =& \sigma(f^i_k)\prod_{j=1}^{K}\sum_{n^i_j=0}^\infty\sigma^{n^i_j}(-f^i_j)\operatorname{Po}(n^i_j|\lambda)\\
    p(y=k|\{\boldsymbol{f}_j\}_{j=1}^{K},\lambda, \{n^i_j\}_{j=1}^{K}) =&\sigma(f^i_k)\prod_{j=1}^{K}\sigma^{n^i_j}(-f^i_j)\operatorname{Po}(n^i_j|\lambda)
\end{align*}
```
Note that as opposed to the bijective version, the prior on ``\lambda`` is the improper prior ``1_{[0, \infty]}``.
Similarly to the bijective version we can also recover a Negative Multinomial
```math
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K},\boldsymbol{n}^i) =\sigma(f^i_k)\prod_{j=1}^K \sigma^{n_j^i}(-f_j^i)\widetilde{\operatorname{NM}}(\boldsymbol{n}^i|1, \left\{\frac{1}{K}\right\}_{j=1}^{K-1})
```

Note that ``\widetilde{\operatorname{NM}}`` is not normalizable (as expected because of the improper prior), however the joint distribution is and results in a valid posterior. 
For the last step:
```math
    p(y^i=k|\{\boldsymbol{f}_j\}_{j=1}^{K},\boldsymbol{n}^i, \boldsymbol{\omega}^i) =\widetilde{\operatorname{NM}}(\boldsymbol{n}^i|1, \{\frac{1}{K}\}_{j=1}^{K})\prod_{j=1}^{K}2^{-(y^i_j + n^i_j)}e^{\frac{1}{2}\left((y^i_j - n^i_j) f^i_j - \omega^i_j (f^i_j)^2\right)}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j,0)
```

Which is the same as the bijective version but with a different prior on ``\boldsymbol{n}``.
## Conditional distributions (Sampling)

Let's define the set of all variables ``\boldsymbol{F} = \{\boldsymbol{f}_j\}_{j=1}^{K}``, ``\boldsymbol{N} = \{\boldsymbol{n}^i\}_{i=1}^N`` and ``\boldsymbol{\Omega} = \{\boldsymbol{\omega}^i\}_{i=1}^N``.

We are interested in the full-conditionals ``p(\boldsymbol{f}_j|\boldsymbol{Y}, \boldsymbol{\Omega}, \boldsymbol{N})`` and ``p(\boldsymbol{\omega}^i, \boldsymbol{n}^i|\boldsymbol{Y},\boldsymbol{F})``
### Bijective version


```math
\begin{align*}
    p(\boldsymbol{f}_j|\boldsymbol{Y},\boldsymbol{\Omega}, \boldsymbol{N}) =& \mathcal{N}(\boldsymbol{f}_j|\mu_j,\Sigma_j)\\
    \Sigma_j =& \left(K^{-1} + \operatorname{Diagonal}(\boldsymbol{\omega}_j)\right)^{-1}\\
    \mu_j =& \Sigma_j\left(\frac{\boldsymbol{y}_j - \boldsymbol{n}_j}{2} + K^{-1}\boldsymbol{\mu}_0\right)\\
    p(\boldsymbol{\omega}^i, \boldsymbol{n}^i|\boldsymbol{y}^i, \boldsymbol{F}) =& \prod_{j=1}^{K-1}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j, |f^i_j|)\operatorname{NM}(\boldsymbol{n}^i|1, \left\{\frac{\sigma(-f^i_j)}{D + \sum_{k=1}^{K-1}\sigma(-f^i_k)}\right\}_{j=1}^{K-1})
\end{align*}
```

Note that ``p(\boldsymbol{\omega}^i, \boldsymbol{n}^i|
\boldsymbol{Y},\boldsymbol{F})`` is defined in the package as a [`AugmentedGPLikelihoods.SpecialDistributions.PolyaGammaNegativeMultinomial`](@ref) distribution.

### Non-bijective version

```math
\begin{align*}
    p(\boldsymbol{f}_j|\boldsymbol{Y},\boldsymbol{\Omega}, \boldsymbol{N}) =& \mathcal{N}(\boldsymbol{f}_j|\mu_j,\Sigma_j)\\
    \Sigma_j =& \left(K^{-1} + \operatorname{Diagonal}(\boldsymbol{\omega}_j)\right)^{-1}\\
    \mu_j =& \Sigma_j\left(\frac{\boldsymbol{y}_j - \boldsymbol{n}_j}{2} + K^{-1}\boldsymbol{\mu}_0\right)\\
    p(\boldsymbol{\omega}^i, \boldsymbol{n}^i|\boldsymbol{y}^i, \boldsymbol{F}) =& \prod_{j=1}^{K-1}\operatorname{PG}(\omega^i_j|y^i_j + n^i_j, |f^i_j|)\operatorname{NM}(\boldsymbol{n}^i|1, \left\{\frac{\sigma(-f^i_j)}{K}\right\}_{j=1}^{K-1})
\end{align*}
```


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
    \mu_j =& \Sigma_j\left(\frac{y + \gamma_j}{2} + K^{-1}\mu_0\right),
\end{align*}
```
where ``\gamma_j^i = E_{q(\boldsymbol{p})}[n_j^i] = \frac{p_j^i}{1 - \sum_{j=1}^{K-1} p_j^i}``, ``\theta_j^i = E_{q(\omega^i_j,n^i_j)}[\omega_j^i] = \frac{y_j^i+\gamma^i_j}{2c_j^i}\tanh\left(\frac{c_j^i}{2}\right)`` and ``\widetilde{\sigma}(q(f_j^i)) = \frac{e^{-\mu_j^i}/2}{2\cosh(c_j^i/2)}`` which is an approximation of the expectation of ``E_{q(f_j^i)}[\sigma(-f_j^i)]``.

For the **bijective version**
```math
p^i_j = \frac{\widetilde{\sigma}(q(f_i^j))}{D + \sum_{j=1}^{K-1}\widetilde{\sigma}(q(f_i^j))},
```
and for the **non-bijective version**

```math
p^i_j = \frac{\widetilde{\sigma}(q(f_i^j))}{K},
```

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

### Non-Bijective version