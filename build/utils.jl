function second_moment(q::Normal)
    return abs2(mean(q)) + var(q)
end

# This is not exactly an approximation but corresponds anyway
# to the expectation of the function σ(f)
function approx_expected_logistic(μ, c)
    lower, upper = LogExpFunctions._logistic_bounds(μ)
    return μ < lower ? zero(μ) : (μ > upper ? one(μ) : exp(μ / 2) * sech(c / 2) / 2)
end
