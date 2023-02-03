using IrrationalConstants: logtwo, log2π

function ref_logpdf(d::PolyaGamma, x::Real)
    b, c = Distributions.params(d)
    if iszero(b)
        return iszero(x) ? zero(x) : -Inf # The limit of PG when b->0
    # is the delta dirac at 0.
    else
        iszero(x) && -Inf # The limit to p(x) for x-> 0 is 0.
        ext = ref_logtilt(x, b, c) + (b - 1) * logtwo - loggamma(b) - (log2π + 3 * log(x)) / 2
        pos_val = logsumexp(ref_pdf_val_log_series(n, b, x) for n in 0:2:4001)
        neg_val = logsumexp(ref_pdf_val_log_series(n, b, x) for n in 1:2:4001)
        return ext + log(exp(pos_val) - exp(neg_val))
    end
end

function ref_pdf_val_log_series(n::Integer, b::Real, x)
    return loggamma(n + b) - loggamma(n + 1) - abs2(2n + b) / (8x) + log(2n + b) # all terms where n is present
end

function ref_logtilt(ω, b, c)
    return b * logcosh(c / 2) - abs2(c) * ω / 2
end


@testset "polyagamma" begin
    @test mean(PolyaGamma(1, 0)) == 1 / 4
    @test mean(PolyaGamma(1, 2.0)) == tanh(1.0) / 4

    for (b, c) in ((1, 0), (1, 2.0), (3, 0), (3, 2.5), (3, 3.2), (1.2, 3.2))
        p = PolyaGamma(b, c)
        xs = rand(p, 1000)
        @test all(isreal, logpdf.(p, 1000*xs))
        @test ref_logpdf.(p, xs) ≈ logpdf.(p, xs)
        @test mean(rand(p, 10000)) ≈ mean(p) atol = 1e-2
    end

    p = PolyaGamma(1, 0)
    @test Distributions.insupport(p, 0)
    @test !Distributions.insupport(p, -1)
    @test minimum(p) === 0
    @test maximum(p) == Inf
    @test Distributions.params(p) == (1, 0)
end
