function second_moment(q::Normal)
    return abs2(mean(q)) + var(q)
end

function second_moment(q::Normal, y::Real)
    return abs2(mean(q) - y) + var(q)
end

# This is not exactly an approximation but corresponds anyway
# to the expectation of the function σ(f)
function approx_expected_logistic(μ::Real, c::Real)
    lower, upper = LogExpFunctions._logistic_bounds(μ)
    return μ < lower ? zero(μ) : (μ > upper ? one(μ) : exp(μ / 2) * sech(c / 2) / 2)
end

# Same thing as above for softmax
function approx_expected_logisticsoftmax(μ::AbstractVector, c::AbstractVector)
    σs = approx_expected_logistic.(μ, c)
    return σs / (logistic(0) + sum(σs))
end
