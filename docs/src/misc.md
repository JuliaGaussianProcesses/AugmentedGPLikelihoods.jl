```@meta
CurrentModule = AugmentedGPLikelihoods
```

```@setup dist_plots
using Plots
using AugmentedGPLikelihoods.SpecialDistributions
using Distributions
default(;lw=0.0, legend=false)
to_name(d::PolyaGamma) = "PG($(d.b),$(d.c))"
function plot_hist_and_pdf(pgs)
    plts = map(pgs) do pg
        ω = rand(pg, 10000)
        plt = Plots.histogram(ω; normalize=:pdf, title=to_name(pg))
        Plots.plot!(plt, LinRange(0, maximum(ω), 1000), x->pdf(pg, x); lw=2.0)
        vline!(plt, [mean(pg)]; lw=2.0)
        vline!(plt, [mean(ω)]; lw=2.0)
        plt
    end
    return Plots.plot(plts...; layout=length(pgs))
end
```

## Additional distributions


### Polya-Gamma
```@docs
SpecialDistributions.PolyaGamma
```

```@example dist_plots
pgs = [PolyaGamma(1, 0), PolyaGamma(2, 0), PolyaGamma(1, 2.5), PolyaGamma(3.5, 4.3)]
plot_hist_and_pdf(pgs)
savefig("pg-plots.svg"); nothing # hide
```
![](pg-plots.svg)


### Negative Multinomial

```@docs
SpecialDistributions.NegativeMultinomial
```

### Polya-Gamma Poisson

```@docs
SpecialDistributions.PolyaGammaPoisson
```



## Additional likelihoods

```@docs
LaplaceLikelihood
NegBinomialLikelihood
StudentTLikelihood
```

## `NamedTuple`/`TupleVector` distribution interface

```@docs
SpecialDistributions.AbstractNTDist
SpecialDistributions.NTDist
SpecialDistributions.ntrand
SpecialDistributions.ntmean
SpecialDistributions.tvrand
SpecialDistributions.tvmean
```