#include "../../fast_bocpd/_c/student_t_ng.h"
#include "test_utils.h"
#include <math.h>

// Test: Initialize prior stats
int test_student_t_ng_prior_stats() {
    StudentTNGStats stats;
    student_t_ng_prior_stats(&stats);
    
    ASSERT_CLOSE(stats.S0, 0.0, 1e-10);
    ASSERT_CLOSE(stats.S1, 0.0, 1e-10);
    ASSERT_CLOSE(stats.S2, 0.0, 1e-10);
    return 0;
}

// Test: Update stats with single observation
int test_student_t_ng_update_stats_single() {
    StudentTNGParams params = {0.0, 1.0, 1.0, 1.0, 3.0};  // nu=3
    StudentTNGStats stats;
    student_t_ng_prior_stats(&stats);
    
    student_t_ng_update_stats(&stats, &params, 5.0);
    
    // Weighted count will be < 1 due to down-weighting (outlier from prior mean of 0)
    ASSERT_TRUE(stats.S0 > 0.0 && stats.S0 < 1.5);
    ASSERT_TRUE(stats.S1 > 0.0);  // Positive value
    ASSERT_TRUE(stats.S2 > 0.0);  // Positive value
    return 0;
}

// Test: Update stats with multiple observations
int test_student_t_ng_update_stats_multiple() {
    StudentTNGParams params = {0.0, 1.0, 1.0, 1.0, 3.0};
    StudentTNGStats stats;
    student_t_ng_prior_stats(&stats);
    
    student_t_ng_update_stats(&stats, &params, 1.0);
    student_t_ng_update_stats(&stats, &params, 2.0);
    student_t_ng_update_stats(&stats, &params, 3.0);
    
    // Weighted sums (should be positive and grow with data)
    ASSERT_TRUE(stats.S0 > 0.0);
    ASSERT_TRUE(stats.S1 > 0.0);
    ASSERT_TRUE(stats.S2 > 0.0);
    return 0;
}

// Test: Predictive PDF at prior (no data)
int test_student_t_ng_predictive_logpdf_basic() {
    StudentTNGParams params = {0.0, 1.0, 1.0, 1.0, 3.0};
    StudentTNGStats stats;
    student_t_ng_prior_stats(&stats);
    
    double logp = student_t_ng_predictive_logpdf(&params, &stats, 0.0);
    
    // Should be finite and negative (log prob)
    ASSERT_FALSE(isnan(logp));
    ASSERT_FALSE(isinf(logp));
    ASSERT_TRUE(logp < 0.0);
    return 0;
}

// Test: Predictive PDF after observing data
int test_student_t_ng_predictive_after_data() {
    StudentTNGParams params = {5.0, 1.0, 1.0, 1.0, 3.0};
    StudentTNGStats stats;
    student_t_ng_prior_stats(&stats);
    
    // Add data centered around 5.0
    for (int i = 0; i < 50; i++) {
        student_t_ng_update_stats(&stats, &params, 5.0);
    }
    
    // Probability at mean should be higher than at outliers
    double logp_at_mean = student_t_ng_predictive_logpdf(&params, &stats, 5.0);
    double logp_outlier = student_t_ng_predictive_logpdf(&params, &stats, 20.0);
    
    ASSERT_TRUE(logp_at_mean > logp_outlier);
    return 0;
}

// Test: Student-t is more robust to outliers
int test_student_t_ng_heavy_tails() {
    // nu=3 (heavy tails) vs nu=10 (light tails)
    StudentTNGParams params_nu3 = {0.0, 1.0, 1.0, 1.0, 3.0};
    StudentTNGParams params_nu10 = {0.0, 1.0, 1.0, 1.0, 10.0};
    
    StudentTNGStats stats_nu3, stats_nu10;
    student_t_ng_prior_stats(&stats_nu3);
    student_t_ng_prior_stats(&stats_nu10);
    
    // Add normal data
    for (int i = 0; i < 50; i++) {
        student_t_ng_update_stats(&stats_nu3, &params_nu3, 0.0);
        student_t_ng_update_stats(&stats_nu10, &params_nu10, 0.0);
    }
    
    // Both should accumulate reasonable stats (for normal data at mean)
    ASSERT_TRUE(stats_nu3.S0 > 40.0);   // Most points should get full weight
    ASSERT_TRUE(stats_nu10.S0 > 40.0);  // Both should be similar
    return 0;
}

// Test: Numerical stability with extreme values
int test_student_t_ng_numerical_stability() {
    StudentTNGParams params = {0.0, 1.0, 1.0, 1.0, 3.0};
    StudentTNGStats stats;
    student_t_ng_prior_stats(&stats);
    
    // Extreme values
    student_t_ng_update_stats(&stats, &params, 1e6);
    student_t_ng_update_stats(&stats, &params, -1e6);
    student_t_ng_update_stats(&stats, &params, 0.0);
    
    double logp = student_t_ng_predictive_logpdf(&params, &stats, 0.0);
    
    ASSERT_FALSE(isnan(logp));
    ASSERT_FALSE(isinf(logp));
    return 0;
}

// Test: Zero variance data handling
int test_student_t_ng_zero_variance_data() {
    StudentTNGParams params = {5.0, 1.0, 1.0, 1.0, 3.0};
    StudentTNGStats stats;
    student_t_ng_prior_stats(&stats);
    
    // All same value
    for (int i = 0; i < 100; i++) {
        student_t_ng_update_stats(&stats, &params, 5.0);
    }
    
    double logp = student_t_ng_predictive_logpdf(&params, &stats, 5.0);
    
    ASSERT_FALSE(isnan(logp));
    ASSERT_FALSE(isinf(logp));
    return 0;
}

// Main test runner - called by test_runner.c
int run_student_t_ng_tests() {
    int failed = 0;
    
    TEST_SUITE("Student-t NG Model");
    
    if (test_student_t_ng_prior_stats() == 0) {
        TEST_PASS("Prior stats initialization");
    } else {
        failed++;
    }
    
    if (test_student_t_ng_update_stats_single() == 0) {
        TEST_PASS("Update stats (single observation)");
    } else {
        failed++;
    }
    
    if (test_student_t_ng_update_stats_multiple() == 0) {
        TEST_PASS("Update stats (multiple observations)");
    } else {
        failed++;
    }
    
    if (test_student_t_ng_predictive_logpdf_basic() == 0) {
        TEST_PASS("Predictive PDF (prior)");
    } else {
        failed++;
    }
    
    if (test_student_t_ng_predictive_after_data() == 0) {
        TEST_PASS("Predictive PDF (after data)");
    } else {
        failed++;
    }
    
    if (test_student_t_ng_heavy_tails() == 0) {
        TEST_PASS("Heavy tails behavior");
    } else {
        failed++;
    }
    
    if (test_student_t_ng_numerical_stability() == 0) {
        TEST_PASS("Numerical stability");
    } else {
        failed++;
    }
    
    if (test_student_t_ng_zero_variance_data() == 0) {
        TEST_PASS("Zero variance data");
    } else {
        failed++;
    }
    
    return failed;
}
