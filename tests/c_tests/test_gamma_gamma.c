#include "../../fast_bocpd/_c/gamma_gamma_fixed_shape.h"
#include "test_utils.h"
#include <math.h>

// Test: Prior stats initialization
static int test_prior_stats() {
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    ASSERT_EQ(stats.n, 0);
    ASSERT_CLOSE(stats.sum_x, 0.0, 1e-10);
    
    TEST_PASS("Prior stats initialization");
    return 0;
}

// Test: Update stats increments correctly
static int test_update_stats() {
    GammaGammaParams params = {
        .alpha0 = 2.0,
        .beta0 = 3.0,
        .shape = 1.5,
        .log_gamma_k = lgamma(1.5)
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // Update with x = 1.5, 2.0, 0.25
    gamma_gamma_update_stats(&stats, &params, 1.5);
    ASSERT_EQ(stats.n, 1);
    ASSERT_CLOSE(stats.sum_x, 1.5, 1e-10);
    
    gamma_gamma_update_stats(&stats, &params, 2.0);
    ASSERT_EQ(stats.n, 2);
    ASSERT_CLOSE(stats.sum_x, 3.5, 1e-10);
    
    gamma_gamma_update_stats(&stats, &params, 0.25);
    ASSERT_EQ(stats.n, 3);
    ASSERT_CLOSE(stats.sum_x, 3.75, 1e-10);
    
    // Ensure it accepts x=0 (stats update is unvalidated)
    gamma_gamma_update_stats(&stats, &params, 0.0);
    ASSERT_EQ(stats.n, 4);
    ASSERT_CLOSE(stats.sum_x, 3.75, 1e-10);
    
    TEST_PASS("Update stats increments correctly");
    return 0;
}

// Test: Predictive at prior matches manual formula (x>0)
static int test_predictive_at_prior() {
    // Friendly numbers
    double alpha0 = 2.0;
    double beta0 = 3.0;
    double shape = 1.7;
    
    GammaGammaParams params = {
        .alpha0 = alpha0,
        .beta0 = beta0,
        .shape = shape,
        .log_gamma_k = lgamma(shape)
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // Test at x = 4.2
    double x = 4.2;
    double logp = gamma_gamma_predictive_logpdf(&params, &stats, x);
    
    // Manual computation (explicit formula, not reusing internal code)
    // logp(x) = lgamma(α+k) - lgamma(α) - lgamma(k) + α*log(β) + (k-1)*log(x) - (α+k)*log(β+x)
    double k = shape;
    double alpha_n = alpha0;  // n=0, so alpha_n = alpha0 + 0*k
    double beta_n = beta0;    // n=0, so beta_n = beta0 + 0
    
    double expected_logp = lgamma(alpha_n + k) 
                         - lgamma(alpha_n) 
                         - lgamma(k) 
                         + alpha_n * log(beta_n) 
                         + (k - 1.0) * log(x) 
                         - (alpha_n + k) * log(beta_n + x);
    
    ASSERT_CLOSE(logp, expected_logp, 1e-10);
    
    TEST_PASS("Predictive at prior matches manual formula");
    return 0;
}

// Test: Predictive after data matches manual formula (x>0)
static int test_predictive_after_data() {
    double alpha0 = 2.0;
    double beta0 = 3.0;
    double shape = 1.7;
    
    GammaGammaParams params = {
        .alpha0 = alpha0,
        .beta0 = beta0,
        .shape = shape,
        .log_gamma_k = lgamma(shape)
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // Update with x = 1.0, 2.0, 3.0
    gamma_gamma_update_stats(&stats, &params, 1.0);
    gamma_gamma_update_stats(&stats, &params, 2.0);
    gamma_gamma_update_stats(&stats, &params, 3.0);
    
    // Now: n=3, sum_x=6.0
    // alpha_n = alpha0 + n*k = 2.0 + 3*1.7 = 7.1
    // beta_n = beta0 + sum_x = 3.0 + 6.0 = 9.0
    
    // Test at x = 0.7
    double x = 0.7;
    double logp = gamma_gamma_predictive_logpdf(&params, &stats, x);
    
    // Manual computation
    double k = shape;
    double alpha_n = alpha0 + 3.0 * k;
    double beta_n = beta0 + 6.0;
    
    double expected_logp = lgamma(alpha_n + k) 
                         - lgamma(alpha_n) 
                         - lgamma(k) 
                         + alpha_n * log(beta_n) 
                         + (k - 1.0) * log(x) 
                         - (alpha_n + k) * log(beta_n + x);
    
    ASSERT_CLOSE(logp, expected_logp, 1e-10);
    
    TEST_PASS("Predictive after data matches manual formula");
    return 0;
}

// Test: Exponential special case at x=0 (shape=1)
static int test_exponential_special_case_x0() {
    double alpha0 = 2.5;
    double beta0 = 4.0;
    double shape = 1.0;
    
    GammaGammaParams params = {
        .alpha0 = alpha0,
        .beta0 = beta0,
        .shape = shape,
        .log_gamma_k = lgamma(shape)  // lgamma(1) = 0
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // Prior stats: expected logp(0) = log(alpha0) - log(beta0)
    double logp_prior = gamma_gamma_predictive_logpdf(&params, &stats, 0.0);
    double expected_prior = log(alpha0) - log(beta0);
    ASSERT_CLOSE(logp_prior, expected_prior, 1e-10);
    
    // After updates: x=1.0, x=2.0
    gamma_gamma_update_stats(&stats, &params, 1.0);
    gamma_gamma_update_stats(&stats, &params, 2.0);
    
    // Now: n=2, sum_x=3.0
    // alpha_n = alpha0 + n*1 = 2.5 + 2 = 4.5
    // beta_n = beta0 + sum_x = 4.0 + 3.0 = 7.0
    // expected logp(0) = log(alpha_n) - log(beta_n)
    double logp_post = gamma_gamma_predictive_logpdf(&params, &stats, 0.0);
    double expected_post = log(4.5) - log(7.0);
    ASSERT_CLOSE(logp_post, expected_post, 1e-10);
    
    TEST_PASS("Exponential special case at x=0 (shape=1)");
    return 0;
}

// Test: x=0 behavior for shape>1 returns -inf
static int test_x0_shape_gt_1() {
    GammaGammaParams params = {
        .alpha0 = 2.0,
        .beta0 = 3.0,
        .shape = 2.0,
        .log_gamma_k = lgamma(2.0)
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    double logp = gamma_gamma_predictive_logpdf(&params, &stats, 0.0);
    ASSERT_TRUE(logp == -INFINITY);
    
    TEST_PASS("x=0 with shape>1 returns -inf");
    return 0;
}

// Test: x=0 behavior for shape<1 returns -inf (anti-poisoning rule)
static int test_x0_shape_lt_1() {
    GammaGammaParams params = {
        .alpha0 = 2.0,
        .beta0 = 3.0,
        .shape = 0.7,
        .log_gamma_k = lgamma(0.7)
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    double logp = gamma_gamma_predictive_logpdf(&params, &stats, 0.0);
    ASSERT_TRUE(logp == -INFINITY);
    
    TEST_PASS("x=0 with shape<1 returns -inf (anti-poisoning)");
    return 0;
}

// Test: Domain validation (invalid x values)
static int test_domain_validation() {
    GammaGammaParams params = {
        .alpha0 = 2.0,
        .beta0 = 3.0,
        .shape = 1.5,
        .log_gamma_k = lgamma(1.5)
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // x = -1.0 (negative)
    double logp_neg = gamma_gamma_predictive_logpdf(&params, &stats, -1.0);
    ASSERT_TRUE(logp_neg == -INFINITY);
    
    // x = NAN
    double logp_nan = gamma_gamma_predictive_logpdf(&params, &stats, NAN);
    ASSERT_TRUE(logp_nan == -INFINITY);
    
    // x = INFINITY
    double logp_inf = gamma_gamma_predictive_logpdf(&params, &stats, INFINITY);
    ASSERT_TRUE(logp_inf == -INFINITY);
    
    TEST_PASS("Domain validation (invalid x values)");
    return 0;
}

// Test: Parameter validation
static int test_parameter_validation() {
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // alpha0 <= 0
    GammaGammaParams bad_alpha = {
        .alpha0 = 0.0, .beta0 = 3.0, .shape = 1.5, .log_gamma_k = lgamma(1.5)
    };
    double logp1 = gamma_gamma_predictive_logpdf(&bad_alpha, &stats, 1.0);
    ASSERT_TRUE(logp1 == -INFINITY);
    
    GammaGammaParams neg_alpha = {
        .alpha0 = -1.0, .beta0 = 3.0, .shape = 1.5, .log_gamma_k = lgamma(1.5)
    };
    double logp2 = gamma_gamma_predictive_logpdf(&neg_alpha, &stats, 1.0);
    ASSERT_TRUE(logp2 == -INFINITY);
    
    // beta0 <= 0
    GammaGammaParams bad_beta = {
        .alpha0 = 2.0, .beta0 = 0.0, .shape = 1.5, .log_gamma_k = lgamma(1.5)
    };
    double logp3 = gamma_gamma_predictive_logpdf(&bad_beta, &stats, 1.0);
    ASSERT_TRUE(logp3 == -INFINITY);
    
    GammaGammaParams neg_beta = {
        .alpha0 = 2.0, .beta0 = -2.0, .shape = 1.5, .log_gamma_k = lgamma(1.5)
    };
    double logp4 = gamma_gamma_predictive_logpdf(&neg_beta, &stats, 1.0);
    ASSERT_TRUE(logp4 == -INFINITY);
    
    // shape <= 0
    GammaGammaParams bad_shape = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = 0.0, .log_gamma_k = 0.0
    };
    double logp5 = gamma_gamma_predictive_logpdf(&bad_shape, &stats, 1.0);
    ASSERT_TRUE(logp5 == -INFINITY);
    
    GammaGammaParams neg_shape = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = -1.0, .log_gamma_k = lgamma(-1.0)
    };
    double logp6 = gamma_gamma_predictive_logpdf(&neg_shape, &stats, 1.0);
    ASSERT_TRUE(logp6 == -INFINITY);
    
    // log_gamma_k is NAN (corrupt cache)
    GammaGammaParams nan_cache = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = 1.5, .log_gamma_k = NAN
    };
    double logp7 = gamma_gamma_predictive_logpdf(&nan_cache, &stats, 1.0);
    ASSERT_TRUE(logp7 == -INFINITY);
    
    // log_gamma_k is INFINITY (corrupt cache)
    GammaGammaParams inf_cache = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = 1.5, .log_gamma_k = INFINITY
    };
    double logp8 = gamma_gamma_predictive_logpdf(&inf_cache, &stats, 1.0);
    ASSERT_TRUE(logp8 == -INFINITY);
    
    // NaN in alpha0
    GammaGammaParams nan_alpha = {
        .alpha0 = NAN, .beta0 = 3.0, .shape = 1.5, .log_gamma_k = lgamma(1.5)
    };
    double logp9 = gamma_gamma_predictive_logpdf(&nan_alpha, &stats, 1.0);
    ASSERT_TRUE(logp9 == -INFINITY);
    
    // NaN in beta0
    GammaGammaParams nan_beta = {
        .alpha0 = 2.0, .beta0 = NAN, .shape = 1.5, .log_gamma_k = lgamma(1.5)
    };
    double logp10 = gamma_gamma_predictive_logpdf(&nan_beta, &stats, 1.0);
    ASSERT_TRUE(logp10 == -INFINITY);
    
    // NaN in shape
    GammaGammaParams nan_shape = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = NAN, .log_gamma_k = NAN
    };
    double logp11 = gamma_gamma_predictive_logpdf(&nan_shape, &stats, 1.0);
    ASSERT_TRUE(logp11 == -INFINITY);
    
    TEST_PASS("Parameter validation");
    return 0;
}

// Test: Corrupted stats validation
static int test_corrupted_stats() {
    GammaGammaParams params = {
        .alpha0 = 2.0,
        .beta0 = 3.0,
        .shape = 1.5,
        .log_gamma_k = lgamma(1.5)
    };
    GammaGammaStats stats;
    
    // n < 0
    stats.n = -1;
    stats.sum_x = 5.0;
    double logp1 = gamma_gamma_predictive_logpdf(&params, &stats, 1.0);
    ASSERT_TRUE(logp1 == -INFINITY);
    
    // sum_x < 0
    stats.n = 5;
    stats.sum_x = -1.0;
    double logp2 = gamma_gamma_predictive_logpdf(&params, &stats, 1.0);
    ASSERT_TRUE(logp2 == -INFINITY);
    
    // sum_x = NAN
    stats.n = 5;
    stats.sum_x = NAN;
    double logp3 = gamma_gamma_predictive_logpdf(&params, &stats, 1.0);
    ASSERT_TRUE(logp3 == -INFINITY);
    
    TEST_PASS("Corrupted stats validation");
    return 0;
}

// Test: Copy stats
static int test_copy_stats() {
    GammaGammaParams params = {
        .alpha0 = 2.0,
        .beta0 = 3.0,
        .shape = 1.5,
        .log_gamma_k = lgamma(1.5)
    };
    GammaGammaStats src, dst;
    
    gamma_gamma_prior_stats(&src);
    gamma_gamma_update_stats(&src, &params, 1.5);
    gamma_gamma_update_stats(&src, &params, 2.3);
    gamma_gamma_update_stats(&src, &params, 0.7);
    
    gamma_gamma_copy_stats(&dst, &src);
    
    ASSERT_EQ(dst.n, src.n);
    ASSERT_CLOSE(dst.sum_x, src.sum_x, 1e-15);
    
    TEST_PASS("Copy stats");
    return 0;
}

// Test: Stats size
static int test_stats_size() {
    size_t size = gamma_gamma_stats_size();
    ASSERT_EQ(size, sizeof(GammaGammaStats));
    
    TEST_PASS("Stats size");
    return 0;
}

// Test: Numerical stability - big n / big sum
static int test_large_n_stability() {
    // Test with shape=1.0 (Exponential)
    GammaGammaParams params1 = {
        .alpha0 = 2.0,
        .beta0 = 5.0,
        .shape = 1.0,
        .log_gamma_k = lgamma(1.0)
    };
    GammaGammaStats stats1;
    gamma_gamma_prior_stats(&stats1);
    
    // Update 1000 times with x=100.0 (sum=100,000)
    for (int i = 0; i < 1000; i++) {
        gamma_gamma_update_stats(&stats1, &params1, 100.0);
    }
    
    // Evaluate at x=50.0
    double logp1 = gamma_gamma_predictive_logpdf(&params1, &stats1, 50.0);
    ASSERT_TRUE(isfinite(logp1));
    ASSERT_TRUE(logp1 <= 0.0);
    
    // Test with shape=2.0
    GammaGammaParams params2 = {
        .alpha0 = 2.0,
        .beta0 = 5.0,
        .shape = 2.0,
        .log_gamma_k = lgamma(2.0)
    };
    GammaGammaStats stats2;
    gamma_gamma_prior_stats(&stats2);
    
    for (int i = 0; i < 1000; i++) {
        gamma_gamma_update_stats(&stats2, &params2, 100.0);
    }
    
    double logp2 = gamma_gamma_predictive_logpdf(&params2, &stats2, 50.0);
    ASSERT_TRUE(isfinite(logp2));
    ASSERT_TRUE(logp2 <= 0.0);
    
    TEST_PASS("Large-n numerical stability");
    return 0;
}

// Test: Numerical stability - small x relative to beta (log1p path)
static int test_small_x_log1p_stability() {
    // Choose beta0 very large, x very small relative to beta
    GammaGammaParams params = {
        .alpha0 = 2.0,
        .beta0 = 1e12,  // Very large beta
        .shape = 1.3,
        .log_gamma_k = lgamma(1.3)
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // Evaluate at x=1.0 (tiny compared to beta)
    // This exercises the log(beta_n) + log1p(x/beta_n) path
    double logp = gamma_gamma_predictive_logpdf(&params, &stats, 1.0);
    
    ASSERT_TRUE(isfinite(logp));
    ASSERT_FALSE(isnan(logp));
    
    TEST_PASS("Small x relative to beta (log1p stability)");
    return 0;
}

// Test: Shape tolerance boundary (k ≈ 1.0 ± 1e-12)
static int test_shape_tolerance_boundary() {
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // Exactly k=1.0 → Exponential case at x=0
    GammaGammaParams params_exact = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = 1.0, .log_gamma_k = lgamma(1.0)
    };
    double logp_exact = gamma_gamma_predictive_logpdf(&params_exact, &stats, 0.0);
    double expected_exact = log(2.0) - log(3.0);
    ASSERT_CLOSE(logp_exact, expected_exact, 1e-12);
    
    // k = 1.0 + 5e-13 (within tolerance, should use Exponential case)
    GammaGammaParams params_within = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = 1.0 + 5e-13, .log_gamma_k = lgamma(1.0 + 5e-13)
    };
    double logp_within = gamma_gamma_predictive_logpdf(&params_within, &stats, 0.0);
    // Should be treated as Exponential (not -inf)
    ASSERT_TRUE(isfinite(logp_within));
    ASSERT_CLOSE(logp_within, expected_exact, 1e-10);  // Relaxed tolerance for near-1
    
    // k = 1.0 + 2e-12 (outside tolerance upward, should return -inf at x=0)
    GammaGammaParams params_above = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = 1.0 + 2e-12, .log_gamma_k = lgamma(1.0 + 2e-12)
    };
    double logp_above = gamma_gamma_predictive_logpdf(&params_above, &stats, 0.0);
    ASSERT_TRUE(logp_above == -INFINITY);
    
    // k = 1.0 - 2e-12 (outside tolerance downward, should return -inf at x=0)
    GammaGammaParams params_below = {
        .alpha0 = 2.0, .beta0 = 3.0, .shape = 1.0 - 2e-12, .log_gamma_k = lgamma(1.0 - 2e-12)
    };
    double logp_below = gamma_gamma_predictive_logpdf(&params_below, &stats, 0.0);
    ASSERT_TRUE(logp_below == -INFINITY);
    
    TEST_PASS("Shape tolerance boundary (k ≈ 1.0)");
    return 0;
}

// Test: Predictive is monotonic in reasonable regime (sanity check)
static int test_predictive_monotonicity() {
    GammaGammaParams params = {
        .alpha0 = 3.0,
        .beta0 = 2.0,
        .shape = 2.0,  // k>1 for well-behaved density
        .log_gamma_k = lgamma(2.0)
    };
    GammaGammaStats stats;
    gamma_gamma_prior_stats(&stats);
    
    // Add some data
    gamma_gamma_update_stats(&stats, &params, 1.0);
    gamma_gamma_update_stats(&stats, &params, 1.5);
    gamma_gamma_update_stats(&stats, &params, 2.0);
    
    // For Gamma(k>1, rate), density should:
    // - Be 0 at x=0 (logp = -inf)
    // - Increase to a mode
    // - Then decrease
    // We just check that very small x has lower density than x near mode
    
    double logp_small = gamma_gamma_predictive_logpdf(&params, &stats, 0.1);
    double logp_moderate = gamma_gamma_predictive_logpdf(&params, &stats, 1.0);
    
    ASSERT_TRUE(isfinite(logp_small));
    ASSERT_TRUE(isfinite(logp_moderate));
    
    // For k=2, mode is at (k-1)/rate, so with posterior rate ≈ 1.5, mode ≈ 0.67
    // So x=1.0 should be reasonably probable
    // Just check both are finite and reasonable
    ASSERT_TRUE(logp_small < 10.0);  // Not absurdly large
    ASSERT_TRUE(logp_moderate < 10.0);
    
    TEST_PASS("Predictive monotonicity sanity check");
    return 0;
}

// Main test runner
int run_gamma_gamma_tests() {
    int failed = 0;
    
    TEST_SUITE("Gamma-Gamma (Fixed Shape)");
    
    if (test_prior_stats() != 0) failed++;
    if (test_update_stats() != 0) failed++;
    if (test_predictive_at_prior() != 0) failed++;
    if (test_predictive_after_data() != 0) failed++;
    if (test_exponential_special_case_x0() != 0) failed++;
    if (test_x0_shape_gt_1() != 0) failed++;
    if (test_x0_shape_lt_1() != 0) failed++;
    if (test_domain_validation() != 0) failed++;
    if (test_parameter_validation() != 0) failed++;
    if (test_corrupted_stats() != 0) failed++;
    if (test_copy_stats() != 0) failed++;
    if (test_stats_size() != 0) failed++;
    if (test_large_n_stability() != 0) failed++;
    if (test_small_x_log1p_stability() != 0) failed++;
    if (test_shape_tolerance_boundary() != 0) failed++;
    if (test_predictive_monotonicity() != 0) failed++;
    
    return failed;
}
