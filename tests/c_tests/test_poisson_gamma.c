// filepath: /home/tiaan/Projects/Fast_BOCPD/tests/c_tests/test_poisson_gamma.c
#include "../../fast_bocpd/_c/poisson_gamma.h"
#include "test_utils.h"
#include <math.h>

// Test: Prior stats initialization
static int test_prior_stats() {
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    ASSERT_EQ(stats.n, 0);
    ASSERT_CLOSE(stats.sum_x, 0.0, 1e-10);
    
    TEST_PASS("Prior stats initialization");
    return 0;
}

// Test: Update stats with observations
static int test_update_stats() {
    PoissonGammaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    // Update with counts
    poisson_gamma_update_stats(&stats, &params, 0.0);
    ASSERT_EQ(stats.n, 1);
    ASSERT_CLOSE(stats.sum_x, 0.0, 1e-10);
    
    poisson_gamma_update_stats(&stats, &params, 2.0);
    ASSERT_EQ(stats.n, 2);
    ASSERT_CLOSE(stats.sum_x, 2.0, 1e-10);
    
    poisson_gamma_update_stats(&stats, &params, 3.0);
    ASSERT_EQ(stats.n, 3);
    ASSERT_CLOSE(stats.sum_x, 5.0, 1e-10);
    
    TEST_PASS("Update stats with observations");
    return 0;
}

// Test: Predictive at x=0 with prior
static int test_predictive_zero_at_prior() {
    PoissonGammaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    double logp0 = poisson_gamma_predictive_logpdf(&params, &stats, 0.0);
    
    // Should be finite
    ASSERT_FALSE(isnan(logp0));
    ASSERT_FALSE(isinf(logp0));
    
    // Manual: α=1, β=1
    // log p(0) = 1.0 * log(1/(1+1)) = log(0.5) = -log(2)
    double expected = -log(2.0);
    ASSERT_CLOSE(logp0, expected, 1e-10);
    
    TEST_PASS("Predictive at x=0 with prior");
    return 0;
}

// Test: Predictive matches manual formula
static int test_predictive_manual() {
    PoissonGammaParams params = {.alpha0 = 2.0, .beta0 = 1.5};
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    // Add some data
    poisson_gamma_update_stats(&stats, &params, 1.0);
    poisson_gamma_update_stats(&stats, &params, 3.0);
    // n=2, sum_x=4, so alpha_n=6, beta_n=3.5
    
    double alpha_n = 6.0;
    double beta_n = 3.5;
    
    // Test x=2
    int64_t x_int = 2;
    double x = (double)x_int;
    double logp = poisson_gamma_predictive_logpdf(&params, &stats, x);
    
    // Manual computation (using stable log forms)
    double log_p_succ = -log1p(1.0 / beta_n);  // log(β/(β+1))
    double log_p_fail = -log1p(beta_n);         // log(1/(β+1))
    double expected = lgamma(alpha_n + x) - lgamma(alpha_n) - lgamma(x + 1.0)
                    + alpha_n * log_p_succ + x * log_p_fail;
    
    ASSERT_CLOSE(logp, expected, 1e-10);
    
    TEST_PASS("Predictive matches manual formula");
    return 0;
}

// Test: Invalid inputs return -inf
static int test_invalid_inputs() {
    PoissonGammaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    // Negative x
    double logp_neg = poisson_gamma_predictive_logpdf(&params, &stats, -1.0);
    ASSERT_TRUE(logp_neg == -INFINITY);
    
    // Non-integer x
    double logp_frac = poisson_gamma_predictive_logpdf(&params, &stats, 1.5);
    ASSERT_TRUE(logp_frac == -INFINITY);
    
    // NaN
    double logp_nan = poisson_gamma_predictive_logpdf(&params, &stats, NAN);
    ASSERT_TRUE(logp_nan == -INFINITY);
    
    // Inf
    double logp_inf = poisson_gamma_predictive_logpdf(&params, &stats, INFINITY);
    ASSERT_TRUE(logp_inf == -INFINITY);
    
    TEST_PASS("Invalid inputs return -inf");
    return 0;
}

// Test: Numerical stability with large counts
static int test_numerical_stability() {
    PoissonGammaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    // Add large counts
    for (int i = 0; i < 10; i++) {
        poisson_gamma_update_stats(&stats, &params, 1000.0);
    }
    
    // Predictive for large x should be finite (not NaN, not +inf)
    double logp = poisson_gamma_predictive_logpdf(&params, &stats, 1200.0);
    ASSERT_FALSE(isnan(logp));
    ASSERT_TRUE(isfinite(logp) || logp == -INFINITY);
    
    TEST_PASS("Numerical stability with large counts");
    return 0;
}

// Test: Copy stats
static int test_copy_stats() {
    PoissonGammaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    PoissonGammaStats src, dst;
    
    poisson_gamma_prior_stats(&src);
    poisson_gamma_update_stats(&src, &params, 5.0);
    poisson_gamma_update_stats(&src, &params, 3.0);
    
    poisson_gamma_copy_stats(&dst, &src);
    
    ASSERT_EQ(dst.n, src.n);
    ASSERT_CLOSE(dst.sum_x, src.sum_x, 1e-15);
    
    TEST_PASS("Copy stats");
    return 0;
}

// Test: Stats size
static int test_stats_size() {
    size_t size = poisson_gamma_stats_size();
    ASSERT_EQ(size, sizeof(PoissonGammaStats));
    
    TEST_PASS("Stats size");
    return 0;
}

// Test: Predictive over multiple x values matches manual NB formula
static int test_predictive_multiple_values() {
    PoissonGammaParams params = {.alpha0 = 3.0, .beta0 = 2.0};
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    // Add some observations
    poisson_gamma_update_stats(&stats, &params, 2.0);
    poisson_gamma_update_stats(&stats, &params, 1.0);
    // n=2, sum_x=3, so alpha_n=6, beta_n=4
    
    double alpha_n = 6.0;
    double beta_n = 4.0;
    double log_p_succ = -log1p(1.0 / beta_n);
    double log_p_fail = -log1p(beta_n);
    
    // Test multiple x values
    for (int64_t x_int = 0; x_int <= 20; x_int++) {
        double x = (double)x_int;
        double logp = poisson_gamma_predictive_logpdf(&params, &stats, x);
        
        // Manual NB formula
        double expected = lgamma(alpha_n + x) - lgamma(alpha_n) - lgamma(x + 1.0)
                        + alpha_n * log_p_succ + x * log_p_fail;
        
        ASSERT_CLOSE(logp, expected, 1e-10);
    }
    
    TEST_PASS("Predictive over multiple x values");
    return 0;
}

// Test: Monotonicity check (tail decay)
static int test_monotonicity_tail_decay() {
    PoissonGammaParams params = {.alpha0 = 2.0, .beta0 = 1.0};
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    poisson_gamma_update_stats(&stats, &params, 5.0);
    poisson_gamma_update_stats(&stats, &params, 8.0);
    
    // Predictive should decay in the tail
    double logp_50 = poisson_gamma_predictive_logpdf(&params, &stats, 50.0);
    double logp_200 = poisson_gamma_predictive_logpdf(&params, &stats, 200.0);
    
    // Far-tail should be less likely
    ASSERT_TRUE(logp_200 < logp_50);
    
    TEST_PASS("Monotonicity (tail decay)");
    return 0;
}

// Test: Integer tolerance boundary
static int test_integer_tolerance_boundary() {
    PoissonGammaParams params = {.alpha0 = 1.0, .beta0 = 1.0};
    PoissonGammaStats stats;
    poisson_gamma_prior_stats(&stats);
    
    // Close to integer (within 1e-9) should be accepted
    double logp_close = poisson_gamma_predictive_logpdf(&params, &stats, 3.0 + 5e-10);
    ASSERT_FALSE(logp_close == -INFINITY);
    ASSERT_FALSE(isnan(logp_close));
    
    // Far from integer (> 1e-9) should be rejected
    double logp_far = poisson_gamma_predictive_logpdf(&params, &stats, 3.0 + 5e-8);
    ASSERT_TRUE(logp_far == -INFINITY);
    
    TEST_PASS("Integer tolerance boundary");
    return 0;
}

// Main test runner
int run_poisson_gamma_tests() {
    int failed = 0;
    
    TEST_SUITE("Poisson-Gamma");
    
    if (test_prior_stats() != 0) failed++;
    if (test_update_stats() != 0) failed++;
    if (test_predictive_zero_at_prior() != 0) failed++;
    if (test_predictive_manual() != 0) failed++;
    if (test_invalid_inputs() != 0) failed++;
    if (test_numerical_stability() != 0) failed++;
    if (test_copy_stats() != 0) failed++;
    if (test_stats_size() != 0) failed++;
    if (test_predictive_multiple_values() != 0) failed++;
    if (test_monotonicity_tail_decay() != 0) failed++;
    if (test_integer_tolerance_boundary() != 0) failed++;
    
    return failed;
}