@doc raw"""
`AbstractNTDist` is an abstract type for a wrapper type around measure(s)
and distributions(s).
The main idea is that instead of `rand`, `mean` and other statistical tools
wrapped objects return `NamedTuple`s or `TupleVector` when having a collection
of them.

The following API has to be implemented:
Given `π::AbstractNTDist` and `Π::AbstractVector{<:AbstractNTDist}`

## Necessary
- `ntrand(rng, π)` -> NamedTuple
- `ntmean(π) -> NamedTuple
- `MeasureBase.logdensity(π, x::NamedTuple)` -> Real
- `Distributions.kldivergence(π₀, π₁)` -> Real
## Optional
- `tvrand(rng, Π)` -> TupleVector
- `tvmean(Π)` -> TupleVector
"""
abstract type AbstractNTDist end

# We replace the `rand` approach by `ntrand`
Base.rand(rng::AbstractRNG, ::Type{T}, π::AbstractNTDist) where {T} = ntrand(rng, π)
Base.rand(rng::AbstractRNG, π::AbstractNTDist) = ntrand(rng, π)
Base.rand(rng::AbstractRNG, π::AbstractNTDist, n::Int) = TupleVector([ntrand(rng, π) for i in 1:n])

# Simple wrapper around any measure or distribution object
struct NTDist{Td} <: AbstractNTDist
    d::Td
end

dist(π::NTDist) = π.d

Statistics.mean(π::NTDist) = ntmean(dist(π))
Statistics.mean(Π::AbstractVector{<:NTDist}) = tvmean(Π)


MeasureBase.logdensity(π::NTDist, x::NamedTuple) = logpdf(dist(π), only(x))

ntrand(rng::AbstractRNG, d) = ntrand(rng, NTDist(d))
function ntrand(rng::AbstractRNG, π::NTDist)
    return (; ω=rand(rng, dist(π)))
end

function tvrand(rng::AbstractRNG, Π::AbstractVector{<:NTDist})
    return (;ω=rand.(rng, dist.(Π)))
end

ntmean(π::NTDist) = (;ω=mean(dist(π)))
function tvmean(Π::AbstractVector{<:NTDist})
    return TupleVector(;ω=mean.(dist.(Π)))
end

Distributions.kldivergence(π₀::NTDist, π₁::NTDist) = kldivergence(dist(π₀), dist(π₁))
