function second_moment(q::Normal)
    return abs2(mean(q)) + var(q)
end

# This is not exactly an approximation but corresponds anyway
# to the expectation of the function σ(f)
function approx_expected_logistic(μ, c)
    halfμ = μ / 2
    halfc = c / 2
    if isfinite(exp(halfμ) / cosh(halfc))
        return exp(halfμ) / cosh(halfc) / 2
    else # When μ is big, E[sigma(f)] = sigma(μ)
        logistic(μ)
    end
end
