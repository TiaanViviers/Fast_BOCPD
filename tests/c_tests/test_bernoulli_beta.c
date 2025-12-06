#include "../../fast_bocpd/_c/bernoulli_beta.h"
#include "test_utils.h"
#include <math.h>

// Test: Prior stats initialization
static int test_prior_stats() {
    BernoulliBetaStats stats;
    bernoulli_beta_prior_stats(&stats);
    
    ASSERT_EQ(stats.n, 0);
    ASSERT_CLOSE(stats.sum_x, 0.0, 1e-10);
    
    TEST_PASS("Prior stats initialization");
    return 0;
}

// Test: Update stats with observations
static int test_update_stats() {
    BernoulliBetaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    BernoulliBetaStats stats;
    bernoulli_beta_prior_stats(&stats);
    
    // Update with binary data: 0, 1, 1, 0
    bernoulli_beta_update_stats(&stats, &params, 0.0);
    ASSERT_EQ(stats.n, 1);
    ASSERT_CLOSE(stats.sum_x, 0.0, 1e-10);
    
    bernoulli_beta_update_stats(&stats, &params, 1.0);
    ASSERT_EQ(stats.n, 2);
    ASSERT_CLOSE(stats.sum_x, 1.0, 1e-10);
    
    bernoulli_beta_update_stats(&stats, &params, 1.0);
    ASSERT_EQ(stats.n, 3);
    ASSERT_CLOSE(stats.sum_x, 2.0, 1e-10);
    
    bernoulli_beta_update_stats(&stats, &params, 0.0);
    ASSERT_EQ(stats.n, 4);
    ASSERT_CLOSE(stats.sum_x, 2.0, 1e-10);
    
    TEST_PASS("Update stats with observations");
    return 0;
}

// Test: Predictive at prior (closed-form)
static int test_predictive_at_prior() {
    BernoulliBetaParams params = {.alpha0 = 2.0, .beta0 = 3.0};
    BernoulliBetaStats stats;
    bernoulli_beta_prior_stats(&stats);
    
    // alpha0=2, beta0=3, n=0, s=0
    // P(x=1) = 2/(2+3) = 0.4
    // P(x=0) = 3/(2+3) = 0.6
    
    double logp1 = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0);
    double logp0 = bernoulli_beta_predictive_logpdf(&params, &stats, 0.0);
    
    double expected_logp1 = log(2.0 / 5.0);
    double expected_logp0 = log(3.0 / 5.0);
    
    ASSERT_CLOSE(logp1, expected_logp1, 1e-10);
    ASSERT_CLOSE(logp0, expected_logp0, 1e-10);
    
    TEST_PASS("Predictive at prior (closed-form)");
    return 0;
}

// Test: Predictive after data (closed-form)
static int test_predictive_after_data() {
    BernoulliBetaParams params = {.alpha0 = 2.0, .beta0 = 3.0};
    BernoulliBetaStats stats;
    bernoulli_beta_prior_stats(&stats);
    
    // Add data: 1, 1, 0 → n=3, s=2
    bernoulli_beta_update_stats(&stats, &params, 1.0);
    bernoulli_beta_update_stats(&stats, &params, 1.0);
    bernoulli_beta_update_stats(&stats, &params, 0.0);
    
    // alpha_n = 2 + 2 = 4
    // beta_n = 3 + (3 - 2) = 4
    // P(x=1) = 4/8 = 0.5
    // P(x=0) = 4/8 = 0.5
    
    double logp1 = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0);
    double logp0 = bernoulli_beta_predictive_logpdf(&params, &stats, 0.0);
    
    double expected_logp = log(0.5);
    
    ASSERT_CLOSE(logp1, expected_logp, 1e-10);
    ASSERT_CLOSE(logp0, expected_logp, 1e-10);
    
    TEST_PASS("Predictive after data (closed-form)");
    return 0;
}

// Test: Binary tolerance behavior
static int test_binary_tolerance() {
    BernoulliBetaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    BernoulliBetaStats stats;
    bernoulli_beta_prior_stats(&stats);
    
    // Get baseline for exact 0 and 1
    double logp0_exact = bernoulli_beta_predictive_logpdf(&params, &stats, 0.0);
    double logp1_exact = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0);
    
    // Close to 1 (within 1e-9) should be treated as 1
    double logp1_close = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0 + 1e-10);
    ASSERT_FALSE(logp1_close == -INFINITY);
    ASSERT_FALSE(isnan(logp1_close));
    ASSERT_CLOSE(logp1_close, logp1_exact, 1e-10);
    
    // Close to 0 (within 1e-9) should be treated as 0
    double logp0_close = bernoulli_beta_predictive_logpdf(&params, &stats, -1e-10);
    ASSERT_FALSE(logp0_close == -INFINITY);
    ASSERT_FALSE(isnan(logp0_close));
    ASSERT_CLOSE(logp0_close, logp0_exact, 1e-10);
    
    // Far from integer (> 1e-9) should be rejected
    double logp_far = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0 + 1e-8);
    ASSERT_TRUE(logp_far == -INFINITY);
    
    // Non-binary value should be rejected
    double logp_half = bernoulli_beta_predictive_logpdf(&params, &stats, 0.5);
    ASSERT_TRUE(logp_half == -INFINITY);
    
    TEST_PASS("Binary tolerance behavior");
    return 0;
}

// Test: Invalid inputs return -inf
static int test_invalid_inputs() {
    BernoulliBetaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    BernoulliBetaStats stats;
    bernoulli_beta_prior_stats(&stats);
    
    // NaN
    double logp_nan = bernoulli_beta_predictive_logpdf(&params, &stats, NAN);
    ASSERT_TRUE(logp_nan == -INFINITY);
    
    // +Infinity
    double logp_inf = bernoulli_beta_predictive_logpdf(&params, &stats, INFINITY);
    ASSERT_TRUE(logp_inf == -INFINITY);
    
    // -Infinity
    double logp_ninf = bernoulli_beta_predictive_logpdf(&params, &stats, -INFINITY);
    ASSERT_TRUE(logp_ninf == -INFINITY);
    
    // Out of range: -1
    double logp_neg = bernoulli_beta_predictive_logpdf(&params, &stats, -1.0);
    ASSERT_TRUE(logp_neg == -INFINITY);
    
    // Out of range: 2
    double logp_2 = bernoulli_beta_predictive_logpdf(&params, &stats, 2.0);
    ASSERT_TRUE(logp_2 == -INFINITY);
    
    TEST_PASS("Invalid inputs return -inf");
    return 0;
}

// Test: Copy stats
static int test_copy_stats() {
    BernoulliBetaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    BernoulliBetaStats src, dst;
    
    bernoulli_beta_prior_stats(&src);
    bernoulli_beta_update_stats(&src, &params, 1.0);
    bernoulli_beta_update_stats(&src, &params, 0.0);
    bernoulli_beta_update_stats(&src, &params, 1.0);
    
    bernoulli_beta_copy_stats(&dst, &src);
    
    ASSERT_EQ(dst.n, src.n);
    ASSERT_CLOSE(dst.sum_x, src.sum_x, 1e-15);
    
    TEST_PASS("Copy stats");
    return 0;
}

// Test: Stats size
static int test_stats_size() {
    size_t size = bernoulli_beta_stats_size();
    ASSERT_EQ(size, sizeof(BernoulliBetaStats));
    
    TEST_PASS("Stats size");
    return 0;
}

// Test: Large-n numerical sanity
static int test_large_n_stability() {
    BernoulliBetaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    BernoulliBetaStats stats;
    bernoulli_beta_prior_stats(&stats);
    
    // Add 10k observations (alternating 0 and 1)
    for (int i = 0; i < 10000; i++) {
        double x = (i % 2 == 0) ? 0.0 : 1.0;
        bernoulli_beta_update_stats(&stats, &params, x);
    }
    
    // Predictive should be finite (not NaN, not +inf)
    double logp0 = bernoulli_beta_predictive_logpdf(&params, &stats, 0.0);
    double logp1 = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0);
    
    ASSERT_FALSE(isnan(logp0));
    ASSERT_FALSE(isnan(logp1));
    ASSERT_TRUE(isfinite(logp0));
    ASSERT_TRUE(isfinite(logp1));
    
    // Log probabilities should be <= 0 (since probs <= 1)
    ASSERT_TRUE(logp0 <= 0.0);
    ASSERT_TRUE(logp1 <= 0.0);
    
    TEST_PASS("Large-n numerical stability");
    return 0;
}

// Test: Parameter defensive checks (in debug mode)
static int test_parameter_validation() {
    BernoulliBetaStats stats;
    bernoulli_beta_prior_stats(&stats);
    
    // Invalid alpha0
    BernoulliBetaParams bad_alpha = {.alpha0 = 0.0, .beta0 = 1.0};
    double logp1 = bernoulli_beta_predictive_logpdf(&bad_alpha, &stats, 1.0);
    // In debug mode, should return -INFINITY; in release, undefined but safe
    
    // Invalid beta0
    BernoulliBetaParams bad_beta = {.alpha0 = 1.0, .beta0 = -1.0};
    double logp2 = bernoulli_beta_predictive_logpdf(&bad_beta, &stats, 1.0);
    
    // NaN params
    BernoulliBetaParams nan_params = {.alpha0 = NAN, .beta0 = 1.0};
    double logp3 = bernoulli_beta_predictive_logpdf(&nan_params, &stats, 1.0);
    
    // Inf params
    BernoulliBetaParams inf_params = {.alpha0 = 1.0, .beta0 = INFINITY};
    double logp4 = bernoulli_beta_predictive_logpdf(&inf_params, &stats, 1.0);
    
    // All should be safe (finite or -inf, not crash)
    ASSERT_FALSE(isnan(logp1));
    ASSERT_FALSE(isnan(logp2));
    ASSERT_FALSE(isnan(logp3));
    ASSERT_FALSE(isnan(logp4));
    
    TEST_PASS("Parameter validation (defensive checks)");
    return 0;
}

// Test: Corrupted stats (sum_x > n or < 0)
static int test_corrupted_stats() {
    BernoulliBetaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    BernoulliBetaStats stats;
    
    // sum_x > n (impossible)
    stats.n = 5;
    stats.sum_x = 10.0;
    double logp1 = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0);
    ASSERT_TRUE(logp1 == -INFINITY);  // Should reject
    
    // sum_x < 0 (impossible)
    stats.n = 5;
    stats.sum_x = -1.0;
    double logp2 = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0);
    ASSERT_TRUE(logp2 == -INFINITY);  // Should reject
    
    // sum_x = NaN
    stats.n = 5;
    stats.sum_x = NAN;
    double logp3 = bernoulli_beta_predictive_logpdf(&params, &stats, 1.0);
    ASSERT_TRUE(logp3 == -INFINITY);  // Should reject
    
    TEST_PASS("Corrupted stats handled safely");
    return 0;
}

// Main test runner
int run_bernoulli_beta_tests() {
    int failed = 0;
    
    TEST_SUITE("Bernoulli-Beta");
    
    if (test_prior_stats() != 0) failed++;
    if (test_update_stats() != 0) failed++;
    if (test_predictive_at_prior() != 0) failed++;
    if (test_predictive_after_data() != 0) failed++;
    if (test_binary_tolerance() != 0) failed++;
    if (test_invalid_inputs() != 0) failed++;
    if (test_copy_stats() != 0) failed++;
    if (test_stats_size() != 0) failed++;
    if (test_large_n_stability() != 0) failed++;
    if (test_parameter_validation() != 0) failed++;
    if (test_corrupted_stats() != 0) failed++;
    
    return failed;
}