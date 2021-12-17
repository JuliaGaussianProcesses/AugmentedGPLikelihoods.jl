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
- `ntmean(π)` -> NamedTuple
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
function Base.rand(rng::AbstractRNG, π::AbstractNTDist, n::Int)
    return TupleVector([ntrand(rng, π) for i in 1:n])
end

# Simple wrapper around any measure or distribution object
@doc raw"""
    NTDist(d) -> NTDist{typeof(d),:ω}
    NTDist{S}(d) -> NTDist{typeof(d),S}

Wrapper around a single distribution to be compatible with the [`ntrand`](@ref), [`ntmean`](@ref) interface.
One can pass the wanted symbol via `S` while the default will be `:ω`.
"""
struct NTDist{Td,S} <: AbstractNTDist
    d::Td
end

NTDist(d) = NTDist{typeof(d),:ω}(d)
NTDist{S}(d) where {S} = NTDist{typeof(d),S}(d)

dist(π::NTDist) = π.d

Statistics.mean(π::NTDist) = ntmean(π)
Statistics.mean(Π::AbstractVector{<:NTDist}) = tvmean(Π)

MeasureBase.logdensity(π::NTDist, x::NamedTuple) = logpdf(dist(π), only(x))

ntrand(rng::AbstractRNG, d) = ntrand(rng, NTDist(d))
function ntrand(rng::AbstractRNG, π::NTDist{Td,S}) where {Td,S}
    return NamedTuple{(S,)}((rand(rng, dist(π)),))
end

function tvrand(rng::AbstractRNG, Π::AbstractVector{<:T}) where {S,Td,T<:NTDist{Td,S}}
    return TupleVector(NamedTuple{(S,)}((rand.(rng, dist.(Π)),)))
end

ntmean(d) = ntmean(NTDist(d))
ntmean(π::NTDist{Td,S}) where {Td,S} = NamedTuple{(S,)}((mean(dist(π)),))
function tvmean(Π::AbstractVector{<:T}) where {S,Td,T<:NTDist{Td,S}}
    return TupleVector(NamedTuple{(S,)}((mean.(dist.(Π)),)))
end

Distributions.kldivergence(π₀::NTDist, π₁::NTDist) = kldivergence(dist(π₀), dist(π₁))
