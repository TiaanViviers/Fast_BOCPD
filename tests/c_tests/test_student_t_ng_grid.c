// filepath: /home/tiaan/Projects/Fast_BOCPD/tests/c_tests/test_student_t_ng_grid.c
#include "../../fast_bocpd/_c/student_t_ng_grid.h"
#include "../../fast_bocpd/_c/student_t_ng.h"
#include "test_utils.h"
#include <math.h>
#include <stdlib.h>

// Test: K=1 grid should equal fixed ν
int test_grid_k1_equals_fixed() {
    // Grid with K=1, nu=3.0
    double nu_grid[1] = {3.0};
    double nu_prior[1] = {1.0};
    StudentTNGGridParams grid_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = 1, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    // Fixed ν=3.0
    StudentTNGParams fixed_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0, .nu = 3.0
    };
    
    // Allocate stats
    size_t grid_size = student_t_ng_grid_stats_size(1);
    void* grid_stats = malloc(grid_size);
    ASSERT_TRUE(grid_stats != NULL);
    StudentTNGStats fixed_stats;
    
    // Initialize both
    student_t_ng_grid_prior_stats(grid_stats, &grid_params);
    student_t_ng_prior_stats(&fixed_stats);
    
    // Test predictive at prior (should be identical)
    double grid_logp = student_t_ng_grid_predictive_logpdf(grid_stats, &grid_params, 0.5);
    double fixed_logp = student_t_ng_predictive_logpdf(&fixed_params, &fixed_stats, 0.5);
    ASSERT_CLOSE(grid_logp, fixed_logp, 1e-10);
    
    // Update both with same data
    student_t_ng_grid_update_stats(grid_stats, &grid_params, 1.5);
    student_t_ng_update_stats(&fixed_stats, &fixed_params, 1.5);
    
    // Predictive should still match
    grid_logp = student_t_ng_grid_predictive_logpdf(grid_stats, &grid_params, 2.0);
    fixed_logp = student_t_ng_predictive_logpdf(&fixed_params, &fixed_stats, 2.0);
    ASSERT_CLOSE(grid_logp, fixed_logp, 1e-10);
    
    free(grid_stats);
    return 0;
}

// Test: Grid weights normalize (logsumexp = 0)
int test_grid_weights_normalize() {
    double nu_grid[3] = {2.0, 3.0, 5.0};
    double nu_prior[3] = {0.2, 0.5, 0.3};  // Already normalized
    StudentTNGGridParams params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = 3, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    size_t stats_size = student_t_ng_grid_stats_size(3);
    void* stats = malloc(stats_size);
    ASSERT_TRUE(stats != NULL);
    
    student_t_ng_grid_prior_stats(stats, &params);
    
    // Check initial log_pi sums to 0 (log-space normalization)
    const double* log_pi = grid_blob_log_pi_const(stats);
    
    // Use stable log-sum-exp
    double m = log_pi[0];
    for (int k = 1; k < 3; k++) {
        m = (log_pi[k] > m) ? log_pi[k] : m;
    }
    double sum = 0.0;
    for (int k = 0; k < 3; k++) {
        if (log_pi[k] > -INFINITY) {
            sum += exp(log_pi[k] - m);
        }
    }
    double logsum = m + log(sum);
    ASSERT_CLOSE(logsum, 0.0, 1e-12);
    
    // Update with observation
    student_t_ng_grid_update_stats(stats, &params, 1.0);
    
    // Weights should still normalize
    log_pi = grid_blob_log_pi_const(stats);
    m = log_pi[0];
    for (int k = 1; k < 3; k++) {
        m = (log_pi[k] > m) ? log_pi[k] : m;
    }
    sum = 0.0;
    for (int k = 0; k < 3; k++) {
        if (log_pi[k] > -INFINITY) {
            ASSERT_FALSE(isnan(log_pi[k]));
            sum += exp(log_pi[k] - m);
        }
    }
    logsum = m + log(sum);
    ASSERT_CLOSE(logsum, 0.0, 1e-12);
    
    free(stats);
    return 0;
}

// Test: K=65 heap allocation path
int test_grid_large_k() {
    // K=65 triggers heap allocation in predictive/update (stack buffer is 64)
    int K = 65;
    double* nu_grid = (double*)malloc(K * sizeof(double));
    double* nu_prior = (double*)malloc(K * sizeof(double));
    ASSERT_TRUE(nu_grid != NULL && nu_prior != NULL);
    
    // Fill with reasonable values
    for (int k = 0; k < K; k++) {
        nu_grid[k] = 2.0 + k * 0.5;  // 2.0, 2.5, 3.0, ...
        nu_prior[k] = 1.0 / K;       // Uniform
    }
    
    StudentTNGGridParams params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = K, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    size_t stats_size = student_t_ng_grid_stats_size(K);
    void* stats = malloc(stats_size);
    ASSERT_TRUE(stats != NULL);
    
    student_t_ng_grid_prior_stats(stats, &params);
    
    // Predictive should work
    double logp = student_t_ng_grid_predictive_logpdf(stats, &params, 0.5);
    ASSERT_FALSE(isnan(logp));
    ASSERT_FALSE(isinf(logp));
    
    // Update should work
    student_t_ng_grid_update_stats(stats, &params, 1.0);
    
    // Weights should normalize (use log-sum-exp)
    const double* log_pi = grid_blob_log_pi_const(stats);
    double m = log_pi[0];
    for (int k = 1; k < K; k++) {
        m = (log_pi[k] > m) ? log_pi[k] : m;
    }
    double sum = 0.0;
    for (int k = 0; k < K; k++) {
        if (log_pi[k] > -INFINITY) {
            sum += exp(log_pi[k] - m);
        }
    }
    double logsum = m + log(sum);
    ASSERT_CLOSE(logsum, 0.0, 1e-12);
    
    free(stats);
    free(nu_grid);
    free(nu_prior);
    return 0;
}

// Test: Zero prior stays -inf
int test_grid_zero_prior() {
    double nu_grid[3] = {2.0, 3.0, 5.0};
    double nu_prior[3] = {0.0, 0.5, 0.5};  // First component has zero weight
    StudentTNGGridParams params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = 3, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    size_t stats_size = student_t_ng_grid_stats_size(3);
    void* stats = malloc(stats_size);
    ASSERT_TRUE(stats != NULL);
    
    student_t_ng_grid_prior_stats(stats, &params);
    
    // First component should be -inf
    const double* log_pi = grid_blob_log_pi_const(stats);
    ASSERT_TRUE(log_pi[0] == -INFINITY);
    ASSERT_FALSE(isinf(log_pi[1]));
    ASSERT_FALSE(isinf(log_pi[2]));
    
    // After update, should still be -inf
    student_t_ng_grid_update_stats(stats, &params, 1.0);
    log_pi = grid_blob_log_pi_const(stats);
    ASSERT_TRUE(log_pi[0] == -INFINITY);
    
    // Remaining weights should still normalize (log-sum-exp)
    double m = (log_pi[1] > log_pi[2]) ? log_pi[1] : log_pi[2];
    double sum = exp(log_pi[1] - m) + exp(log_pi[2] - m);
    double logsum = m + log(sum);
    ASSERT_CLOSE(logsum, 0.0, 1e-12);
    
    free(stats);
    return 0;
}

// Test: Grid adapts weights based on data
int test_grid_weight_adaptation() {
    // nu=2 (heavy tails) vs nu=50 (light tails) - extreme contrast
    double nu_grid[2] = {2.0, 50.0};
    double nu_prior[2] = {0.5, 0.5};  // Start uniform
    StudentTNGGridParams params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = 2, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    size_t stats_size = student_t_ng_grid_stats_size(2);
    void* stats = malloc(stats_size);
    ASSERT_TRUE(stats != NULL);
    
    student_t_ng_grid_prior_stats(stats, &params);
    
    // Feed heavy-tailed data: frequent large outliers
    for (int i = 0; i < 60; i++) {
        double x = (i % 3 == 0) ? 15.0 : 0.0;  // 1 in 3 is huge outlier
        student_t_ng_grid_update_stats(stats, &params, x);
    }
    
    // Heavy-tail component should dominate
    const double* log_pi = grid_blob_log_pi_const(stats);
    double pi_heavy = exp(log_pi[0]);  // nu=2
    double pi_light = exp(log_pi[1]);  // nu=50
    
    ASSERT_TRUE(pi_heavy > 0.8);  // Should strongly favor heavy tails
    ASSERT_CLOSE(pi_heavy + pi_light, 1.0, 1e-10);  // Still normalized
    
    free(stats);
    return 0;
}

// Test: Numerical stability with extreme values
int test_grid_numerical_stability() {
    double nu_grid[3] = {2.0, 3.0, 5.0};
    double nu_prior[3] = {0.33, 0.33, 0.34};
    StudentTNGGridParams params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = 3, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    size_t stats_size = student_t_ng_grid_stats_size(3);
    void* stats = malloc(stats_size);
    ASSERT_TRUE(stats != NULL);
    
    student_t_ng_grid_prior_stats(stats, &params);
    
    // Extreme values
    student_t_ng_grid_update_stats(stats, &params, 1e6);
    student_t_ng_grid_update_stats(stats, &params, -1e6);
    student_t_ng_grid_update_stats(stats, &params, 0.0);
    
    // Should not produce NaN
    double logp = student_t_ng_grid_predictive_logpdf(stats, &params, 0.0);
    ASSERT_FALSE(isnan(logp));
    
    // Weights should still be valid
    const double* log_pi = grid_blob_log_pi_const(stats);
    for (int k = 0; k < 3; k++) {
        ASSERT_FALSE(isnan(log_pi[k]));
    }
    
    free(stats);
    return 0;
}

// Test: Stats blob alignment
int test_grid_stats_alignment() {
    int K = 5;
    size_t stats_size = student_t_ng_grid_stats_size(K);
    
    // Verify size is reasonable (not absurdly large due to alignment bugs)
    size_t min_size = sizeof(int32_t) + K * sizeof(double) + K * sizeof(StudentTNGStats);
    ASSERT_TRUE(stats_size >= min_size);
    ASSERT_TRUE(stats_size < min_size * 2);  // Shouldn't double due to alignment
    
    // Verify alignment matches expected
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    size_t expected_align = _Alignof(StudentTNGStats);
#else
    size_t expected_align = 16;
#endif
    
    // Stats base offset should respect alignment
    size_t base_offset = grid_stats_base_offset(K);
    ASSERT_TRUE(base_offset % expected_align == 0);
    
    return 0;
}

// Test: Copy stats preserves everything
int test_grid_copy_stats() {
    double nu_grid[2] = {3.0, 5.0};
    double nu_prior[2] = {0.4, 0.6};
    StudentTNGGridParams params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = 2, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    size_t stats_size = student_t_ng_grid_stats_size(2);
    void* src = malloc(stats_size);
    void* dst = malloc(stats_size);
    ASSERT_TRUE(src != NULL && dst != NULL);
    
    // Initialize and update source
    student_t_ng_grid_prior_stats(src, &params);
    student_t_ng_grid_update_stats(src, &params, 1.5);
    
    // Copy
    student_t_ng_grid_copy_stats(dst, src, &params);
    
    // Verify log_pi matches
    const double* src_log_pi = grid_blob_log_pi_const(src);
    const double* dst_log_pi = grid_blob_log_pi_const(dst);
    for (int k = 0; k < 2; k++) {
        ASSERT_CLOSE(dst_log_pi[k], src_log_pi[k], 1e-15);
    }
    
    // Verify component stats match
    for (int k = 0; k < 2; k++) {
        const StudentTNGStats* src_stats = grid_blob_comp_stats_const(src, k, 2);
        const StudentTNGStats* dst_stats = grid_blob_comp_stats_const(dst, k, 2);
        ASSERT_CLOSE(dst_stats->S0, src_stats->S0, 1e-15);
        ASSERT_CLOSE(dst_stats->S1, src_stats->S1, 1e-15);
        ASSERT_CLOSE(dst_stats->S2, src_stats->S2, 1e-15);
    }
    
    free(src);
    free(dst);
    return 0;
}

// Test: Predictive equals manual mixture (CRITICAL CORRECTNESS TEST)
int test_grid_predictive_equals_manual_mixture() {
    double nu_grid[3] = {2.0, 3.0, 5.0};
    double nu_prior[3] = {0.25, 0.5, 0.25};
    StudentTNGGridParams params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = 3, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    size_t stats_size = student_t_ng_grid_stats_size(3);
    void* grid_stats = malloc(stats_size);
    ASSERT_TRUE(grid_stats != NULL);
    
    student_t_ng_grid_prior_stats(grid_stats, &params);
    
    // Update with some data
    student_t_ng_grid_update_stats(grid_stats, &params, 0.5);
    student_t_ng_grid_update_stats(grid_stats, &params, 1.0);
    
    // Compute grid predictive
    double x_test = 0.75;
    double grid_logp = student_t_ng_grid_predictive_logpdf(grid_stats, &params, x_test);
    
    // Compute manual mixture: log( sum_k pi_k * p_k(x) )
    const double* log_pi = grid_blob_log_pi_const(grid_stats);
    
    // Use log-sum-exp for stability
    double log_terms[3];
    for (int k = 0; k < 3; k++) {
        // Skip -inf priors
        if (log_pi[k] == -INFINITY) {
            log_terms[k] = -INFINITY;
            continue;
        }
        
        const StudentTNGStats* comp_stats = grid_blob_comp_stats_const(grid_stats, k, 3);
        StudentTNGParams fixed_params = {
            .mu0 = params.mu0, .kappa0 = params.kappa0,
            .alpha0 = params.alpha0, .beta0 = params.beta0,
            .nu = nu_grid[k]
        };
        double logp_k = student_t_ng_predictive_logpdf(&fixed_params, comp_stats, x_test);
        log_terms[k] = log_pi[k] + logp_k;
    }
    
    // Logsumexp
    double m = -INFINITY;
    for (int k = 0; k < 3; k++) {
        if (log_terms[k] > m) m = log_terms[k];
    }
    double sum = 0.0;
    for (int k = 0; k < 3; k++) {
        if (log_terms[k] > -INFINITY) {
            sum += exp(log_terms[k] - m);
        }
    }
    double manual_logp = m + log(sum);
    
    // They should match
    ASSERT_CLOSE(grid_logp, manual_logp, 1e-10);
    
    free(grid_stats);
    return 0;
}

// Test: Weight update follows Bayes rule (CRITICAL CORRECTNESS TEST)
int test_grid_weight_update_matches_bayes_rule() {
    double nu_grid[2] = {3.0, 5.0};
    double nu_prior[2] = {0.4, 0.6};
    StudentTNGGridParams params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0,
        .K = 2, .nu_grid = nu_grid, .nu_prior = nu_prior
    };
    
    size_t stats_size = student_t_ng_grid_stats_size(2);
    void* stats = malloc(stats_size);
    ASSERT_TRUE(stats != NULL);
    
    student_t_ng_grid_prior_stats(stats, &params);
    
    // Store old log_pi
    const double* log_pi_old = grid_blob_log_pi_const(stats);
    double old_weights[2] = {log_pi_old[0], log_pi_old[1]};
    
    // Compute predictive for each component BEFORE update
    double x = 1.5;
    double logp_k[2];
    for (int k = 0; k < 2; k++) {
        const StudentTNGStats* comp_stats = grid_blob_comp_stats_const(stats, k, 2);
        StudentTNGParams fixed_params = {
            .mu0 = params.mu0, .kappa0 = params.kappa0,
            .alpha0 = params.alpha0, .beta0 = params.beta0,
            .nu = nu_grid[k]
        };
        logp_k[k] = student_t_ng_predictive_logpdf(&fixed_params, comp_stats, x);
    }
    
    // Update
    student_t_ng_grid_update_stats(stats, &params, x);
    
    // Get new log_pi
    const double* log_pi_new = grid_blob_log_pi_const(stats);
    
    // Check normalization
    double m_new = (log_pi_new[0] > log_pi_new[1]) ? log_pi_new[0] : log_pi_new[1];
    double sum_new = exp(log_pi_new[0] - m_new) + exp(log_pi_new[1] - m_new);
    double logsum_new = m_new + log(sum_new);
    ASSERT_CLOSE(logsum_new, 0.0, 1e-12);
    
    // Check Bayes rule: log_pi_new[k] ∝ log_pi_old[k] + logp_k[k]
    // Verify by checking relative differences (cancels normalizer)
    double diff_expected = (old_weights[0] + logp_k[0]) - (old_weights[1] + logp_k[1]);
    double diff_actual = log_pi_new[0] - log_pi_new[1];
    
    ASSERT_CLOSE(diff_actual, diff_expected, 1e-10);
    
    free(stats);
    return 0;
}

// Test: Invalid K=0 should be handled safely
int test_grid_invalid_k0() {
    // K=0 should result in safe behavior (likely stats_size returns 0 or init fails)
    size_t stats_size = student_t_ng_grid_stats_size(0);
    
    // Either stats_size is 0 (no allocation needed), or minimal size
    // The key is it shouldn't crash or return garbage
    ASSERT_TRUE(stats_size == 0 || stats_size < 1000);  // Sanity check
    
    return 0;
}

// Main test runner
int run_student_t_ng_grid_tests() {
    int failed = 0;
    
    TEST_SUITE("Student-t NG Grid");
    
    if (test_grid_k1_equals_fixed() == 0) {
        TEST_PASS("K=1 grid equals fixed ν");
    } else {
        failed++;
    }
    
    if (test_grid_weights_normalize() == 0) {
        TEST_PASS("Grid weights normalize");
    } else {
        failed++;
    }
    
    if (test_grid_large_k() == 0) {
        TEST_PASS("K=65 heap allocation path");
    } else {
        failed++;
    }
    
    if (test_grid_zero_prior() == 0) {
        TEST_PASS("Zero prior stays -inf");
    } else {
        failed++;
    }
    
    if (test_grid_weight_adaptation() == 0) {
        TEST_PASS("Grid adapts weights to data");
    } else {
        failed++;
    }
    
    if (test_grid_numerical_stability() == 0) {
        TEST_PASS("Numerical stability with extremes");
    } else {
        failed++;
    }
    
    if (test_grid_stats_alignment() == 0) {
        TEST_PASS("Stats blob alignment");
    } else {
        failed++;
    }
    
    if (test_grid_copy_stats() == 0) {
        TEST_PASS("Copy stats preserves data");
    } else {
        failed++;
    }
    
    if (test_grid_predictive_equals_manual_mixture() == 0) {
        TEST_PASS("Predictive equals manual mixture");
    } else {
        failed++;
    }
    
    if (test_grid_weight_update_matches_bayes_rule() == 0) {
        TEST_PASS("Weight update matches Bayes rule");
    } else {
        failed++;
    }
    
    if (test_grid_invalid_k0() == 0) {
        TEST_PASS("Invalid K=0 handled safely");
    } else {
        failed++;
    }
    
    return failed;
}
