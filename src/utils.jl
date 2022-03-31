function second_moment(q::Normal)
    return abs2(mean(q)) + var(q)
end

function second_moment(q::Normal, y::Real)
    return abs2(mean(q) - y) + var(q)
end

# This is not exactly an approximation but corresponds anyway
# to the expectation of the function σ(f)
function approx_expected_logistic(μ, c)
    lower, upper = ignore_derivatives() do 
        LogExpFunctions._logistic_bounds(μ)
    end
    return μ < lower ? zero(μ) : (μ > upper ? one(μ) : exp(μ / 2) * sech(c / 2) / 2)
end
