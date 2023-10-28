# Laplace Likelihood

The [`LaplaceLikelihood`](@ref) is defined as

```math
    p(y|f,\beta) = \frac{1}{2\beta}\exp\left(-\frac{|y-f|}{\beta}\right)
```

## The augmentation

We use the technique developed in [galy20](@cite).

Running the inverse Laplace transform on the function ``\exp(-\frac{\sqrt{x}}{\beta})`` returns the measure

```math
\frac{e^{-\frac{1}{4\beta^2\omega}}}{2\beta\sqrt{\pi}}\omega^{-\frac{3}{2}},
```

which corresponds to an [Inverse Gamma distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution) with shape ``\alpha=\frac{1}{2}`` and scale ``\beta'=(2\beta)^{-2}``.

Therefore, we can write the Laplace distribution as the following Gaussian scale-mixture:

```math
\operatorname{Laplace}(y|f,\beta) = \frac{\Gamma(\frac{1}{2})}{2\beta\sqrt{\pi}}\int_0^\infty \exp\left(-(y-f)^2\omega\right)\mathcal{IG}(\omega|\frac{1}{2},(2\beta)^{-2})d\omega,
```

where ``\mathcal{IG}`` is the inverse Gamma distribution.

## Conditional distributions (Sampling)

We are interested in the full-conditionals ``p(f|y,\omega,\beta)`` and ``p(\omega|y,f,\beta)``:

```math
\begin{align*}
    p(f|y,\omega,\sigma,\nu) =& \mathcal{N}(f|\mu,\Sigma)\\
    \Sigma =& \left(K^{-1} + \operatorname{Diagonal}(\omega^{-1})\right)^{-1}\\
    \mu =& \Sigma\left(2\omega^{-1} y + K^{-1}\mu_0\right)\\
    p(\omega_i|y_i,f_i,\sigma,\nu) =& \mathcal{IN}\left(\omega_i|\frac{1}{|2\beta(y-f)|}, 2(2\beta)^{-2}\right),
\end{align*}
```

where ``\mathcal{IN}`` is an [Inverse Gaussian distribution](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution).

## Variational distributions (Variational Inference)

We define the variational distribution with a block mean-field approximation:

```math
    q(f,\omega) = q(f)\prod_{i=1}^Nq(\omega_i) = \mathcal{N}(f|m,S)\prod_{i=1}^N \mathcal{IN}(\omega_i|\mu_i, \lambda_i).
```

The optimal variational parameters are given by:

```math
\begin{align*}
    \mu_i =& \frac{1}{2\beta\sqrt{(y_i - \mu_i)^2 + S_{ii}}},\\
    \lambda_i =& \frac{1}{2\beta^2}\\
    m =& \Sigma\left(\mu y + K^{-1}\mu_0\right),\\
    S =& \left(K^{-1} + \operatorname{Diagonal}(\mu)\right)^{-1}.
\end{align*}
```

We get the ELBO as

```math
    \mathcal{L} = N\left(\log \Gamma(\frac{1}{2}) - \log (\pi) - \log(2\beta)\right) + \sum_{i=1}^N -\left((y_i-m_i)^2 + S_{ii}\right)\theta_i - \operatorname{KL}(q(\omega)||p(\omega)) - \operatorname{KL}(q(f)||p(f)),
```

where ``\theta_i = E_{q(\omega_i)}\left[\omega_i\right] = \mu_i``

```math
\begin{align*}
    \operatorname{KL}(q(\omega_i|\mu_i,2\lambda)||p(\omega_i|\frac{1}{2},\lambda)) =& \frac{1}{2}\log 2\lambda - \frac{1}{2}\log 2\pi - \frac{1}{2}\log \lambda  + \log \Gamma(\frac{1}{2}) + \frac{\lambda}{\mu}.
\end{align*}
```
