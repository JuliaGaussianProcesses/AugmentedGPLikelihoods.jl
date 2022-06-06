# Heteroscedastic Gaussian (Inverse Scaled Logistic Link)

The [`HeteroscedasticGaussian`](https://juliagaussianprocesses.github.io/GPLikelihoods.jl/dev/api/#GPLikelihoods.HeteroscedasticGaussianLikelihood) with an inverse scaled logistic link is defined as

```math
p(y|f, g, \lambda) = \frac{\sqrt{\lambda \sigma(g)}}{2\pi}\exp\left(-\frac{\lambda\sigma(g)}{2}\left(y - f\right)^2\right).
```

## The augmentation

```math
\begin{align*}
    p(y|f, g, \lambda) =& \frac{\sqrt{\lambda \sigma(g)}}{2\pi}\exp\left(-\frac{\lambda\sigma(g)}{2}\left(y - f\right)^2\right),\\
    =& \frac{\sqrt{\lambda \sigma(g)}}{2\pi}\exp\left(\left(\sigma(-g) - 1\right)\frac{\lambda}{2}\left(y - f\right)^2\right).
\end{align*}
```

We use the identity with the [Moment Generating Function](https://en.wikipedia.org/wiki/Moment-generating_function) of the [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution).

```math
\begin{align*}
    p(y|f, g, \lambda) =& \frac{\sqrt{\lambda \sigma(g)}}{2\pi}\sum_{n=0}^\infty \sigma^{-n}(-g)\operatorname{Po}\left(n|\frac{\lambda}{2}\left(y - f\right)^2\right),
    =& \left(\frac{\lambda}{2\pi}\right)^{-\frac{1}{2}}
\end{align*}
```

We augment with the variable $n$.

```math
\begin{align*}
    p(y, n|f, g, \lambda) =& \frac{\sqrt{\lambda \sigma(g)}}{2\pi}\sigma^{-n}(-g)\operatorname{Po}\left(n|\frac{\lambda}{2}\left(y - f\right)^2\right)\\
    =& \left(\frac{\lambda}{2\pi}\right)^{-\frac{1}{2}}\sigma^{\frac{1}{2}}(g)\sigma^n(-g)\operatorname{Po}\left(n|\frac{\lambda}{2}\left(y - f\right)^2\right).
\end{align*}
```

We finally augment the part $\sigma^{\frac{1}{2}}(g)\sigma^n(-g)$ with the [Pólya-Gamma augmentation]().

```math
    \sigma^{\frac{1}{2}}(g)\sigma^n(-g) = 2^{\frac{1}{2} + n}\int_0^\infty\exp\left(\frac{1}{2}\left(\left(\frac{1}{2} - n\right)g - \omega g^2\right)\right)\operatorname{PG}\left(\omega|\frac{1}{2} + n, 0\right)d\omega.
```

This gives use the final augmented likelihood:

```math
    p(y,n,\omega|f,g,\lambda) = \left(\frac{\lambda}{2\pi}\right)^{-\frac{1}{2}}2^{\frac{1}{2} + n}\exp\left(\frac{1}{2}\left(\left(\frac{1}{2} - n\right)g - \omega g^2\right)\right)p(\omega|n)p(n|f,y,\lambda).
```

where $p(\omega|n)=\operatorname{PG}(\omega|\frac{1}{2} + n, 0)$ and $p(n|f,y,\lambda) = \operatorname{Po}\left(n|\frac{\lambda}{2}\left(y - f\right)^2\right)$.

## Conditional distributions (Sampling)

Let's start with the augmented variables $n$ and $\omega$.
The full conditional is given by:

```math
    p(\omega,n|f,g,\lambda,y) = \operatorname{PG}\left(\omega|\frac{1}{2}+n,|g|\right)\operatorname{Po}\left(n|\frac{\lambda\sigma(g)}{2}(y-f)^2\right).
```

The full conditional of $\boldsymbol{g}$ is $p(\boldsymbol{g}|\boldsymbol{\omega}, \boldsymbol{n}) = \mathcal{N}\left(\boldsymbol{g}|\boldsymbol{\mu},\Sigma_g\right)$ where

```math
\begin{align*}
    \Sigma_g =& \left(K_g^{-1} + \mathrm{Diagonal}(\boldsymbol{\omega})\right)^{-1},\\
    \boldsymbol{\mu}_g =& \Sigma_g \left(K_g^{-1}\boldsymbol{\mu}_0 + \frac{1}{2}\left(\frac{1}{2} - \boldsymbol{n}\right)\right).
\end{align*}
```

There is no closed-form solution for the full-conditional of $\boldsymbol{f}$ but we can compute the collapsed full-conditional, i.e. we take the full-conditional of $\boldsymbol{f}$ and marginalize out $\omega$ and $n$.
The conditional is given by $p(\boldsymbol{f}|\boldsymbol{g},\lambda,\boldsymbol{y}) = \mathcal{N}(\boldsymbol{f}|\boldsymbol{\mu}_f,\Sigma_f)$ where

```math
\begin{align*}
    \Sigma_f =& \left(K_f^{-1} + \mathrm{Diagonal}(\lambda \sigma(\boldsymbol{g}))\right)^{-1},\\
    \boldsymbol{\mu}_f =& \Sigma_g \left(K_g^{-1}\boldsymbol{\mu}_0 + \frac{\lambda\sigma(\boldsymbol{g})}{2}\boldsymbol{y}\right).
\end{align*}
```

We can also treat $\lambda$ probabilistically by adding a [Gamma prior](https://en.wikipedia.org/wiki/Gamma_distribution) $p(\lambda) = \mathrm{\Gamma}(\alpha, \beta)$. The (collapsed) full conditional is then given by

```math
    p(\lambda|\boldsymbol{f}, \boldsymbol{g}, \boldsymbol{y}) = \mathrm{Gamma}\left(\lambda|\alpha + \frac{N}{2}, \beta + \sum_{i=1}^N \frac{\sigma(g_i)}{2}(y_i - f_i)^2\right).
```

## Variational distributions (Variational Inference)

We define the variational distribution with a block mean-field approximation:

```math
    q(\boldsymbol{f}, \boldsymbol{g}, \boldsymbol{n}, \boldsymbol{\omega},\lambda) = \mathcal{N}(\boldsymbol{f}|\boldsymbol{m}_f,\boldsymbol{S}_f)\mathcal{N}(\boldsymbol{g}|\boldsymbol{m}_g,\boldsymbol{S}_g)\prod_{i=1}^N \operatorname{PG}\left(\omega_i|\frac{1}{2} + n_i|c_i\right)\operatorname{Po}(n_i|\gamma_i).
```

The CAVI updates are more complicated to get here.
It's not possible to obtain the optimal parameters for $q(\boldsymbol{f})$ since the full-conditional are not available in closed form.
We resort instead to a double bound approach described in a document soon to be published (see #97).

The optimal variational parameters are given by:

```math
\begin{align*}
    c^i =& \sqrt{(m^i_g)^2 + S^{ii}_g},\\
    \gamma^i =& \frac{\lambda}{2}\psi^i \sqrt{(y^i - m_f^i)^2 + S_f^{ii}} \\
    \boldsymbol{S}_g =& \left(K^{-1} + \operatorname{Diagonal}(\boldsymbol{\theta})\right)^{-1},\\
    \boldsymbol{m}_g =& \boldsymbol{S}_g\left(\frac{\frac{1}{2} - \boldsymbol{\gamma}}{2} + K_g^{-1}\mu_0\right),
\end{align*}
```

where

```math
\begin{align*}
    \psi^i = \frac{e^{-m_g^i/2}}{\sqrt{(m_g^i)^2 + S_g^{ii}}/2}
\end{align*}
```

For $\boldsymbol{f}$ the optimal parameters are:

```math
\begin{align*}
    \boldsymbol{S}_f =& \left(K^{-1} + \lambda\operatorname{Diagonal}(1 - \boldsymbol{\psi})\right)^{-1},\\
    \boldsymbol{m}_f =& \boldsymbol{S}_f\left(\lambda \mathrm{Diagonal}(1 - \boldsymbol{\psi})\boldsymbol{y} + K_g^{-1}\mu_0\right),
\end{align*}
```

As stated earlier the ELBO is made in two steps and is currently not implemented.

We get the ELBO as

```math
\begin{align*}
    \mathcal{L} =& \sum_{i=1}^N -  (\frac{1}{2} + \gamma^i_j) \log 2 + \frac{(\frac{1}{2} - \gamma^i) m^i_g}{2} - \frac{(m^i_g)^2 + S_g^{ii}}{2}\theta^i\\ 
    &- \operatorname{KL}(q(\boldsymbol{\omega},\boldsymbol{n})||p(\boldsymbol{\omega},\boldsymbol{n}|\boldsymbol{y})) - \operatorname{KL}(q(\boldsymbol{f,g})||p(\boldsymbol{f,g})),
\end{align*}
```

where

```math
\begin{align*}
    \operatorname{KL}(\prod_{j} q(\omega^i_j|n^i_j)q(\bm n^i)||\prod_j p(\omega^i_j|\bm n^i, \boldsymbol{y}^i)p(\bm n^i)) =& \operatorname{KL}(q(\bm n^i)||p(\bm n^i)) + \sum_j E_{q(\bm n^i)}\left[\operatorname{KL}(q(\omega^i_j|\bm n^i)||p(\omega^i_j|\bm n^i, \boldsymbol{y}^i)\right],\\
    E_{q(\bm \omega^i,\bm n^i)}\left[\operatorname{KL}(q(\omega^i_j|\bm n^i)||p(\omega^i_j|n^i_j, y^i)\right] =& (y^i + \gamma^i_j)\log\cosh \left(\frac{c^i_j}{2}\right) - (c^i_j)^2 \frac{\theta^i_j}{2}.
\end{align*}
```
