# File for wrapping some of the specific functions around existing distributions

## InverseGamma
function tvmean(ds::AbstractVector{<:InverseGamma})
    return (; ω=mean.(ds))
end

function tvmeaninv(ds::AbstractVector{<:InverseGamma})
    return (; ω=mean.(getfield.(ds, :invd)))
end

# return the mean of X⁻¹
function ntmeaninv(d::InverseGamma)
    return (; ω=mean(d.invd))
end

function MeasureBase.logdensity(d::InverseGamma, x::NamedTuple)
    return logpdf(d, x.ω)
end

function Base.rand(rng::AbstractRNG, ::Type{T}, d::InverseGamma) where {T}
    return ntrand(rng, d)
end

function ntrand(rng::AbstractRNG, d::InverseGamma)
    return (; ω=rand(rng, d))
end
