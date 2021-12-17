# File for wrapping some of the specific functions around existing distributions


## InverseGamma
function tvmeaninv(ds::AbstractVector{<:NTDist{<:InverseGamma}})
    return TupleVector(ntmeaninv.(ds))
end

# return the mean of X⁻¹
function ntmeaninv(d::NTDist{<:InverseGamma})
    return (; ω=mean(dist(d).invd))
end
