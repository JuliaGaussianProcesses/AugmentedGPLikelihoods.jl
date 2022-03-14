# Categorical Likelihood (LogisticSoftmax Link)

The [`CategoricalLikelihood`](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/stable/api/#GPLikelihoods.CategoricalLikelihood) with ``K`` categories with a logistic-softmax link is defined as
```math
    p(y=k|\{f_j\}_{j=1}^K) = \frac{\sigma(f_k)}{\sum_{j=1}^K \sigma(f_j)},
```
Note that two versions are possible:
- The bijective one where we have ``K-1`` latent Gaussian Processes and set the last ``f_K`` to a fixed value ``C``.
- The non-bijective, or overparametrized version where we have ``K`` latent GPs.


## The augmentation

### Bijective version

We have ``\sigma(f_K) = \sigma(C) = D \in [0, 1]``.
```math
\begin{align*}
    p(y=k|\{f_j\}_{j=1}^{K-1}) =& \frac{\sigma(f_k)}{D + \sum_{i=1}^{K-1}\sigma(f_j)}\\
    =&\sigma(f_k)\frac{1}{D}\frac{1}{1 + \frac{1}{D} \sum_{i=1}^{K-1}\sigma(f_j)}\\
    =& \sigma(f_k)\int_{0}^\infty \exp(\lambda \sum_{i=1}^{K-1}\sigma(f_j))\operatorname{Ga}(\lambda|1,\frac{1}{D})d\lambda\\
    p(y=k|\{f_j\}_{j=1}^{K=1},\lambda) =& \sigma(f_k)\exp(\lambda \sum_{i=1}^{K-1}\sigma(f_j))\operatorname{Ga}(\lambda|1,\frac{1}{D})\\
    = \sigma(f_k)\prod_{j=1}^{K-1}\sum_{n_j=0}^\infty\sigma^{n_j}(-f_j)\operatorname{Po}(n_j|\lambda)\operatorname{Ga}(\lambda|1,\frac{1}{D})\\
    p(y=k|\{f_j\}_{j=1}^{K=1},\lambda, \{n_j\}_{j=1}^{K-1}) =&\sigma(f_k)\prod_{j=1}^{K-1}\sigma^{n_j}(-f_j)\operatorname{Po}(n_j|\lambda)\operatorname{Ga}(\lambda|1,\frac{1}{D})\\
\end{align*}
```
We could continue with this, but one can actually notice that a product of independent Poisson variables with a Gamma prior will produce a Negative Multinomial variable by marginalizing out ``\lambda``:
```math
    p(y=k|\{f_j\}_{j=1}^{K=1},\{n_j\}_{j=1}^{K-1}) =\sigma(f_k)\prod_{j=1}^{K-1}\sigma^{n_j}(-f_j)\operatorname{NM}(\boldsymbol{n}|1, \{\frac{1}{D + K - 1}\}_{j=1}^{K-1})
```

It's one less variable!
Now we can easily perform the last augmentation using the Pólya-Gamma augmentations.
```math
    p(y=k|\{f_j\}_{j=1}^{K=1},\{n_j\}_{j=1}^{K-1}, \{\omega_j\}_{j=1}^{K-1}) =\operatorname{NM}(\boldsymbol{n}|1, \{\frac{1}{D + K - 1}\}_{j=1}^{K-1})\prod_{j=1}^{K-1}2^{-(y_j + n_j)}\exp\left(\frac{1}{2}\left((y_j - n_j) f_j - \omega_j f_j^2\right)\right)\sigma(f_k)\operatorname{PG}(\omega_j|y_j + n_j,0)
```

### Non-bijective version


## Conditional distributions (Sampling)


### Bijective version

We are interested in the full-conditionals ``p(\boldsymbol{f}_j|y,\boldsymbol{\omega}, \boldsymbol{n})`` and ``p(\boldsymbol{\omega}, \boldsymbol{n}|y,\boldsymbol{f}_j)``:
```math
\begin{align*}
    p(\boldsymbol{f}_j|y,\boldsymbol{\Omega}, \boldsymbol{N}) =& \mathcal{N}(f|\mu_j,\Sigma_j)\\
    \Sigma_j =& \left(K^{-1} + \operatorname{Diagonal}(\boldsymbol{\omega}_j)\right)^{-1}\\
    \mu_j =& \Sigma\left(\frac{y_j - \boldsymbol{n}_j}{2} + K^{-1}\mu_0\right)\\
    p(\omega^i, \boldsymbol{n}^i|\boldsymbol{y}^i, \{f^i\}_{i=1}^N) =& \prod_{j=1}^{K-1}\operatorname{PG}(\omega_j|y^i_j + n^i_j, |f^i_j|)\operatorname{NM}(\boldsymbol{n}^i|1, \{\frac{\sigma(-f^i_j)}{D + \sum_{k=1}^{K-1}\sigma(-f^i_k)}\}_{j=1}^{K-1})
\end{align*}
```

Note that ``p(\boldsymbol{omega}^i,\boldsymbol{n}^i |y,f)`` is defined in the package as a [`PolyaGammaNegativeMultinomial`](@ref) distribution.

### Non-bijective version

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
