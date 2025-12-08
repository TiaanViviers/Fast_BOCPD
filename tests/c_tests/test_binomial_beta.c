#include "../../fast_bocpd/_c/binomial_beta.h"
#include "../../fast_bocpd/_c/bernoulli_beta.h"
#include "test_utils.h"
#include <math.h>

// Test: Prior stats initialization
static int test_prior_stats() {
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    ASSERT_EQ(stats.n, 0);
    ASSERT_CLOSE(stats.sum_k, 0.0, 1e-10);
    
    TEST_PASS("Prior stats initialization");
    return 0;
}

// Test: Update stats with observations
static int test_update_stats() {
    BinomialBetaParams params = {
        .alpha0 = 1.0, 
        .beta0 = 1.0, 
        .N = 10, 
        .log_N_factorial = lgamma(11.0)  // Proper value (even though update_stats doesn't use it)
    };
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // Update with count data: k = 0, 3, 10, 5
    binomial_beta_update_stats(&stats, &params, 0.0);
    ASSERT_EQ(stats.n, 1);
    ASSERT_CLOSE(stats.sum_k, 0.0, 1e-10);
    
    binomial_beta_update_stats(&stats, &params, 3.0);
    ASSERT_EQ(stats.n, 2);
    ASSERT_CLOSE(stats.sum_k, 3.0, 1e-10);
    
    binomial_beta_update_stats(&stats, &params, 10.0);
    ASSERT_EQ(stats.n, 3);
    ASSERT_CLOSE(stats.sum_k, 13.0, 1e-10);
    
    binomial_beta_update_stats(&stats, &params, 5.0);
    ASSERT_EQ(stats.n, 4);
    ASSERT_CLOSE(stats.sum_k, 18.0, 1e-10);
    
    TEST_PASS("Update stats with observations");
    return 0;
}

// Test: Predictive at prior (closed-form)
static int test_predictive_at_prior() {
    BinomialBetaParams params = {
        .alpha0 = 2.0, 
        .beta0 = 3.0, 
        .N = 5,
        .log_N_factorial = lgamma(6.0)  // lgamma(N+1) = lgamma(6)
    };
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // At prior: alpha0=2, beta0=3, N=5, n=0, s=0
    // For k=2: compute Beta-Binomial manually
    // P(k) = (N choose k) * B(alpha0+k, beta0+N-k) / B(alpha0, beta0)
    
    double logp2 = binomial_beta_predictive_logpdf(&params, &stats, 2.0);
    
    // Manual computation:
    // log(5 choose 2) = lgamma(6) - lgamma(3) - lgamma(4)
    // log B(2+2, 3+5-2) = log B(4, 6) = lgamma(4) + lgamma(6) - lgamma(10)
    // log B(2, 3) = lgamma(2) + lgamma(3) - lgamma(5)
    double log_binom = lgamma(6.0) - lgamma(3.0) - lgamma(4.0);
    double log_beta_post = lgamma(4.0) + lgamma(6.0) - lgamma(10.0);
    double log_beta_prior = lgamma(2.0) + lgamma(3.0) - lgamma(5.0);
    double expected_logp = log_binom + log_beta_post - log_beta_prior;
    
    ASSERT_CLOSE(logp2, expected_logp, 1e-10);
    
    TEST_PASS("Predictive at prior (closed-form)");
    return 0;
}

// Test: Predictive after data (closed-form)
static int test_predictive_after_data() {
    BinomialBetaParams params = {
        .alpha0 = 2.0, 
        .beta0 = 3.0, 
        .N = 10,
        .log_N_factorial = lgamma(11.0)
    };
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // Add data: k=3, k=7, k=5 → n=3, sum_k=15
    binomial_beta_update_stats(&stats, &params, 3.0);
    binomial_beta_update_stats(&stats, &params, 7.0);
    binomial_beta_update_stats(&stats, &params, 5.0);
    
    // alpha_n = 2 + 15 = 17
    // beta_n = 3 + (3*10 - 15) = 3 + 15 = 18
    
    // Predict k=6
    double logp6 = binomial_beta_predictive_logpdf(&params, &stats, 6.0);
    
    // Manual:
    double log_binom = lgamma(11.0) - lgamma(7.0) - lgamma(5.0);
    double log_beta_post = lgamma(23.0) + lgamma(22.0) - lgamma(45.0);
    double log_beta_prior = lgamma(17.0) + lgamma(18.0) - lgamma(35.0);
    double expected_logp = log_binom + log_beta_post - log_beta_prior;
    
    ASSERT_CLOSE(logp6, expected_logp, 1e-10);
    
    TEST_PASS("Predictive after data (closed-form)");
    return 0;
}

// Test: N=1 reduces to Bernoulli-Beta (equivalence test)
static int test_n1_equals_bernoulli() {
    // Binomial with N=1
    BinomialBetaParams binom_params = {
        .alpha0 = 2.0,
        .beta0 = 3.0,
        .N = 1,
        .log_N_factorial = lgamma(2.0)  // lgamma(1+1)
    };
    BinomialBetaStats binom_stats;
    binomial_beta_prior_stats(&binom_stats);
    
    // Update with k=1, k=0, k=1
    binomial_beta_update_stats(&binom_stats, &binom_params, 1.0);
    binomial_beta_update_stats(&binom_stats, &binom_params, 0.0);
    binomial_beta_update_stats(&binom_stats, &binom_params, 1.0);
    
    // Bernoulli with same data
    BernoulliBetaParams bern_params = {.alpha0 = 2.0, .beta0 = 3.0};
    BernoulliBetaStats bern_stats;
    bernoulli_beta_prior_stats(&bern_stats);
    
    bernoulli_beta_update_stats(&bern_stats, &bern_params, 1.0);
    bernoulli_beta_update_stats(&bern_stats, &bern_params, 0.0);
    bernoulli_beta_update_stats(&bern_stats, &bern_params, 1.0);
    
    // Stats should match
    ASSERT_EQ(binom_stats.n, bern_stats.n);
    ASSERT_CLOSE(binom_stats.sum_k, bern_stats.sum_x, 1e-15);
    
    // Predictives should match exactly for k=0 and k=1
    double binom_logp0 = binomial_beta_predictive_logpdf(&binom_params, &binom_stats, 0.0);
    double binom_logp1 = binomial_beta_predictive_logpdf(&binom_params, &binom_stats, 1.0);
    
    double bern_logp0 = bernoulli_beta_predictive_logpdf(&bern_params, &bern_stats, 0.0);
    double bern_logp1 = bernoulli_beta_predictive_logpdf(&bern_params, &bern_stats, 1.0);
    
    ASSERT_CLOSE(binom_logp0, bern_logp0, 1e-12);
    ASSERT_CLOSE(binom_logp1, bern_logp1, 1e-12);
    
    TEST_PASS("N=1 equals Bernoulli-Beta");
    return 0;
}

// Test: Integer tolerance behavior
static int test_integer_tolerance() {
    BinomialBetaParams params = {
        .alpha0 = 1.0, 
        .beta0 = 1.0, 
        .N = 10,
        .log_N_factorial = lgamma(11.0)
    };
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // Get baseline for exact integer
    double logp5_exact = binomial_beta_predictive_logpdf(&params, &stats, 5.0);
    
    // Close to 5 (within 1e-9) should be treated as 5
    double logp5_close = binomial_beta_predictive_logpdf(&params, &stats, 5.0 + 1e-10);
    ASSERT_FALSE(logp5_close == -INFINITY);
    ASSERT_FALSE(isnan(logp5_close));
    ASSERT_CLOSE(logp5_close, logp5_exact, 1e-10);
    
    // Far from integer (> 1e-9) should be rejected
    double logp_far = binomial_beta_predictive_logpdf(&params, &stats, 5.0 + 1e-8);
    ASSERT_TRUE(logp_far == -INFINITY);
    
    // Non-integer value should be rejected
    double logp_half = binomial_beta_predictive_logpdf(&params, &stats, 5.5);
    ASSERT_TRUE(logp_half == -INFINITY);
    
    TEST_PASS("Integer tolerance behavior");
    return 0;
}

// Test: Invalid inputs return -inf
static int test_invalid_inputs() {
    BinomialBetaParams params = {
        .alpha0 = 1.0, 
        .beta0 = 1.0, 
        .N = 10,
        .log_N_factorial = lgamma(11.0)
    };
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // NaN
    double logp_nan = binomial_beta_predictive_logpdf(&params, &stats, NAN);
    ASSERT_TRUE(logp_nan == -INFINITY);
    
    // +Infinity
    double logp_inf = binomial_beta_predictive_logpdf(&params, &stats, INFINITY);
    ASSERT_TRUE(logp_inf == -INFINITY);
    
    // -Infinity
    double logp_ninf = binomial_beta_predictive_logpdf(&params, &stats, -INFINITY);
    ASSERT_TRUE(logp_ninf == -INFINITY);
    
    // Negative k
    double logp_neg = binomial_beta_predictive_logpdf(&params, &stats, -1.0);
    ASSERT_TRUE(logp_neg == -INFINITY);
    
    // k > N
    double logp_over = binomial_beta_predictive_logpdf(&params, &stats, 11.0);
    ASSERT_TRUE(logp_over == -INFINITY);
    
    // k way beyond N (overflow protection test)
    double logp_huge = binomial_beta_predictive_logpdf(&params, &stats, 1000.0);
    ASSERT_TRUE(logp_huge == -INFINITY);
    
    TEST_PASS("Invalid inputs return -inf");
    return 0;
}

// Test: Copy stats
static int test_copy_stats() {
    BinomialBetaParams params = {
        .alpha0 = 1.0, 
        .beta0 = 1.0, 
        .N = 10,
        .log_N_factorial = lgamma(11.0)
    };
    BinomialBetaStats src, dst;
    
    binomial_beta_prior_stats(&src);
    binomial_beta_update_stats(&src, &params, 3.0);
    binomial_beta_update_stats(&src, &params, 7.0);
    binomial_beta_update_stats(&src, &params, 5.0);
    
    binomial_beta_copy_stats(&dst, &src);
    
    ASSERT_EQ(dst.n, src.n);
    ASSERT_CLOSE(dst.sum_k, src.sum_k, 1e-15);
    
    TEST_PASS("Copy stats");
    return 0;
}

// Test: Stats size
static int test_stats_size() {
    size_t size = binomial_beta_stats_size();
    ASSERT_EQ(size, sizeof(BinomialBetaStats));
    
    TEST_PASS("Stats size");
    return 0;
}

// Test: Large-n numerical stability
static int test_large_n_stability() {
    BinomialBetaParams params = {
        .alpha0 = 1.0, 
        .beta0 = 1.0, 
        .N = 100,
        .log_N_factorial = lgamma(101.0)
    };
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // Add 1000 observations with k ~ N/2 (alternating pattern)
    for (int i = 0; i < 1000; i++) {
        double k = (i % 2 == 0) ? 45.0 : 55.0;
        binomial_beta_update_stats(&stats, &params, k);
    }
    
    // Predictive should be finite
    double logp50 = binomial_beta_predictive_logpdf(&params, &stats, 50.0);
    
    ASSERT_FALSE(isnan(logp50));
    ASSERT_TRUE(isfinite(logp50));
    ASSERT_TRUE(logp50 <= 0.0);  // Log probability must be <= 0
    
    TEST_PASS("Large-n numerical stability");
    return 0;
}

// Test: Parameter validation (defensive checks)
static int test_parameter_validation() {
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // Invalid alpha0
    BinomialBetaParams bad_alpha = {
        .alpha0 = 0.0, .beta0 = 1.0, .N = 10, .log_N_factorial = lgamma(11.0)
    };
    double logp1 = binomial_beta_predictive_logpdf(&bad_alpha, &stats, 5.0);
    ASSERT_TRUE(logp1 == -INFINITY);
    
    // Invalid beta0
    BinomialBetaParams bad_beta = {
        .alpha0 = 1.0, .beta0 = -1.0, .N = 10, .log_N_factorial = lgamma(11.0)
    };
    double logp2 = binomial_beta_predictive_logpdf(&bad_beta, &stats, 5.0);
    ASSERT_TRUE(logp2 == -INFINITY);
    
    // Invalid N
    BinomialBetaParams bad_N = {
        .alpha0 = 1.0, .beta0 = 1.0, .N = 0, .log_N_factorial = 0.0
    };
    double logp3 = binomial_beta_predictive_logpdf(&bad_N, &stats, 5.0);
    ASSERT_TRUE(logp3 == -INFINITY);
    
    // NaN params
    BinomialBetaParams nan_params = {
        .alpha0 = NAN, .beta0 = 1.0, .N = 10, .log_N_factorial = lgamma(11.0)
    };
    double logp4 = binomial_beta_predictive_logpdf(&nan_params, &stats, 5.0);
    ASSERT_TRUE(logp4 == -INFINITY);
    
    // NaN log_N_factorial (corrupted cache)
    BinomialBetaParams bad_cache = {
        .alpha0 = 1.0, .beta0 = 1.0, .N = 10, .log_N_factorial = NAN
    };
    double logp5 = binomial_beta_predictive_logpdf(&bad_cache, &stats, 5.0);
    ASSERT_TRUE(logp5 == -INFINITY);
    
    // Wrong but finite cache (documents caller error - will produce wrong results)
    // This is a documentation test: we don't crash, just return garbage
    BinomialBetaParams wrong_cache = {
        .alpha0 = 1.0, .beta0 = 1.0, .N = 10, 
        .log_N_factorial = 42.0  // Wrong value (should be lgamma(11) ≈ 15.1)
    };
    double logp6 = binomial_beta_predictive_logpdf(&wrong_cache, &stats, 5.0);
    // Should be finite (wrong but finite), not NaN/inf
    ASSERT_TRUE(isfinite(logp6));
    
    TEST_PASS("Parameter validation");
    return 0;
}

// Test: Corrupted stats (sum_k out of valid range)
static int test_corrupted_stats() {
    BinomialBetaParams params = {
        .alpha0 = 1.0, 
        .beta0 = 1.0, 
        .N = 10,
        .log_N_factorial = lgamma(11.0)
    };
    BinomialBetaStats stats;
    
    // sum_k > n*N (impossible)
    stats.n = 5;
    stats.sum_k = 100.0;  // Max should be 5*10=50
    double logp1 = binomial_beta_predictive_logpdf(&params, &stats, 5.0);
    ASSERT_TRUE(logp1 == -INFINITY);
    
    // sum_k < 0 (impossible)
    stats.n = 5;
    stats.sum_k = -1.0;
    double logp2 = binomial_beta_predictive_logpdf(&params, &stats, 5.0);
    ASSERT_TRUE(logp2 == -INFINITY);
    
    // sum_k = NaN
    stats.n = 5;
    stats.sum_k = NAN;
    double logp3 = binomial_beta_predictive_logpdf(&params, &stats, 5.0);
    ASSERT_TRUE(logp3 == -INFINITY);
    
    TEST_PASS("Corrupted stats handled safely");
    return 0;
}

// Test: Extreme underflow handling (very negative log probabilities)
static int test_extreme_underflow() {
    BinomialBetaParams params = {
        .alpha0 = 1.0, 
        .beta0 = 100.0,  // Strong prior toward low success rate
        .N = 100,
        .log_N_factorial = lgamma(101.0)
    };
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // Add data consistent with low success rate
    for (int i = 0; i < 50; i++) {
        binomial_beta_update_stats(&stats, &params, 5.0);  // 5% success rate
    }
    
    // Predict very high k (extreme tail event)
    double logp_extreme = binomial_beta_predictive_logpdf(&params, &stats, 95.0);
    
    // Should be finite (very negative) or -inf, but never +inf or NaN
    ASSERT_FALSE(isnan(logp_extreme));
    ASSERT_FALSE(logp_extreme > 0.0);
    ASSERT_TRUE(logp_extreme == -INFINITY || isfinite(logp_extreme));
    
    // Predict k in plausible range (should be finite)
    double logp_plausible = binomial_beta_predictive_logpdf(&params, &stats, 5.0);
    ASSERT_TRUE(isfinite(logp_plausible));
    ASSERT_TRUE(logp_plausible <= 0.0);
    
    TEST_PASS("Extreme underflow handling");
    return 0;
}

// Test: Normalization (predictive sums to 1 over k=0..N)
static int test_predictive_normalization() {
    BinomialBetaParams params = {
        .alpha0 = 2.0, 
        .beta0 = 3.0, 
        .N = 10,
        .log_N_factorial = lgamma(11.0)
    };
    BinomialBetaStats stats;
    binomial_beta_prior_stats(&stats);
    
    // Add some data
    binomial_beta_update_stats(&stats, &params, 3.0);
    binomial_beta_update_stats(&stats, &params, 7.0);
    
    // Sum probabilities over all k=0..N
    double prob_sum = 0.0;
    for (int32_t k = 0; k <= params.N; k++) {
        double logp = binomial_beta_predictive_logpdf(&params, &stats, (double)k);
        prob_sum += exp(logp);
    }
    
    // Should sum to 1 (within numerical tolerance)
    // Using 1e-9 for portability across libm implementations
    ASSERT_CLOSE(prob_sum, 1.0, 1e-9);
    
    TEST_PASS("Predictive normalization");
    return 0;
}

// Main test runner
int run_binomial_beta_tests() {
    int failed = 0;
    
    TEST_SUITE("Binomial-Beta");
    
    if (test_prior_stats() != 0) failed++;
    if (test_update_stats() != 0) failed++;
    if (test_predictive_at_prior() != 0) failed++;
    if (test_predictive_after_data() != 0) failed++;
    if (test_n1_equals_bernoulli() != 0) failed++;
    if (test_integer_tolerance() != 0) failed++;
    if (test_invalid_inputs() != 0) failed++;
    if (test_copy_stats() != 0) failed++;
    if (test_stats_size() != 0) failed++;
    if (test_large_n_stability() != 0) failed++;
    if (test_parameter_validation() != 0) failed++;
    if (test_corrupted_stats() != 0) failed++;
    if (test_extreme_underflow() != 0) failed++;
    if (test_predictive_normalization() != 0) failed++;
    
    return failed;
}
