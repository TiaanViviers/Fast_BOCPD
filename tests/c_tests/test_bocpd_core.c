#include "test_utils.h"
#include "../../fast_bocpd/_c/bocpd_core.h"
#include "../../fast_bocpd/_c/gaussian_nig.h"
#include "../../fast_bocpd/_c/student_t_ng.h"
#include "../../fast_bocpd/_c/hazard.h"

int test_bocpd_init() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    
    int ret = bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
                         HAZARD_CONSTANT, &hazard_params, 50);
    
    ASSERT_EQ(ret, 0);
    ASSERT_EQ(state.max_run_length, 50);
    ASSERT_TRUE(state.log_joint != NULL);
    ASSERT_TRUE(state.stats != NULL);
    
    bocpd_free(&state);
    TEST_PASS("BOCPD initialization");
    return 0;
}

int test_bocpd_invalid_params() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    
    // Invalid max_run_length
    int ret = bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
                         HAZARD_CONSTANT, &hazard_params, 0);
    ASSERT_EQ(ret, -1);
    
    TEST_PASS("Rejects invalid parameters");
    return 0;
}

int test_bocpd_reset() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    // Process some data
    double cp_prob;
    bocpd_update(&state, 1.0, &cp_prob);
    bocpd_update(&state, 2.0, &cp_prob);
    
    // Reset
    bocpd_reset(&state);
    
    // Check initial state
    ASSERT_CLOSE(state.log_joint[0], 0.0, 1e-10);
    ASSERT_TRUE(state.log_joint[1] == -INFINITY);
    
    bocpd_free(&state);
    TEST_PASS("Reset to initial state");
    return 0;
}

int test_bocpd_single_update() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    double cp_prob;
    double* posterior_r = bocpd_update(&state, 0.5, &cp_prob);
    
    ASSERT_TRUE(posterior_r != NULL);
    ASSERT_TRUE(cp_prob >= 0.0 && cp_prob <= 1.0);
    
    // Check posterior sums to 1
    double sum = 0.0;
    for (int r = 0; r <= state.max_run_length; r++) {
        sum += posterior_r[r];
    }
    ASSERT_CLOSE(sum, 1.0, 1e-6);
    
    bocpd_free(&state);
    TEST_PASS("Single update produces valid posterior");
    return 0;
}

int test_bocpd_changepoint_detection() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    // Feed constant data
    double cp_prob_const = 0.0;
    for (int i = 0; i < 10; i++) {
        bocpd_update(&state, 0.1, &cp_prob_const);
    }
    
    // Feed a very different observation
    double cp_prob_change;
    bocpd_update(&state, 10.0, &cp_prob_change);
    
    // CP probability should increase
    ASSERT_TRUE(cp_prob_change > cp_prob_const);
    
    bocpd_free(&state);
    TEST_PASS("Detects changepoint");
    return 0;
}

int test_bocpd_batch_update() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    double data[] = {0.1, 0.2, 0.15, 5.0, 5.1, 4.9};
    double cp_probs[6];
    int n_data = 6;
    
    int ret = bocpd_batch_update(&state, data, n_data, cp_probs);
    ASSERT_EQ(ret, 0);
    
    // All probabilities should be valid
    for (int i = 0; i < n_data; i++) {
        ASSERT_TRUE(cp_probs[i] >= 0.0 && cp_probs[i] <= 1.0);
    }
    
    bocpd_free(&state);
    TEST_PASS("Batch update processes multiple observations");
    return 0;
}

int test_bocpd_known_values() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    // Known test case (values from corrected implementation with proper r=0 stats update)
    double test_data[] = {0.1, 0.2, 0.15, 5.0, 5.1, 4.9};
    double expected_probs[] = {0.010000, 0.006854, 0.005487, 0.207100, 0.002784, 0.001547};
    
    for (int i = 0; i < 6; i++) {
        double cp_prob;
        bocpd_update(&state, test_data[i], &cp_prob);
        ASSERT_CLOSE(cp_prob, expected_probs[i], 1e-4);
    }
    
    bocpd_free(&state);
    TEST_PASS("Matches known reference values");
    return 0;
}

int test_get_map_run_length() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    // Initially should be at r=0
    int32_t map_r = bocpd_get_map_run_length(&state);
    ASSERT_EQ(map_r, 0);
    
    // After constant observations, should increase
    double cp_prob;
    for (int i = 0; i < 10; i++) {
        bocpd_update(&state, 0.1, &cp_prob);
    }
    
    map_r = bocpd_get_map_run_length(&state);
    ASSERT_TRUE(map_r > 5);  // Should have grown
    
    bocpd_free(&state);
    TEST_PASS("MAP run length tracking");
    return 0;
}

int test_get_posterior() {
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    double cp_prob;
    bocpd_update(&state, 0.5, &cp_prob);
    
    double posterior[51];
    int ret = bocpd_get_posterior(&state, posterior);
    ASSERT_EQ(ret, 0);
    
    // Posterior should sum to 1
    double sum = 0.0;
    for (int r = 0; r <= 50; r++) {
        sum += posterior[r];
        ASSERT_TRUE(posterior[r] >= 0.0);  // All non-negative
    }
    ASSERT_CLOSE(sum, 1.0, 1e-6);
    
    bocpd_free(&state);
    TEST_PASS("Get posterior distribution");
    return 0;
}

int test_stats_alignment() {
    // Test that stats_size is properly aligned for both models
    BOCPDState state_gaussian, state_student;
    
    GaussianNIGParams gauss_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    StudentTNGParams student_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0, .nu = 3.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    
    bocpd_init(&state_gaussian, OBS_MODEL_GAUSSIAN_NIG, &gauss_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    bocpd_init(&state_student, OBS_MODEL_STUDENT_T_NG, &student_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    // Determine expected alignment (C11 vs C99 fallback)
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    size_t expected_align = _Alignof(max_align_t);
#else
    size_t expected_align = 16;
#endif
    
    // Verify stats_size is properly aligned
    ASSERT_TRUE(state_gaussian.stats_size % expected_align == 0);
    ASSERT_TRUE(state_student.stats_size % expected_align == 0);
    
    // Verify stats_size >= actual struct size
    ASSERT_TRUE(state_gaussian.stats_size >= sizeof(GaussianNIGStats));
    ASSERT_TRUE(state_student.stats_size >= sizeof(StudentTNGStats));
    
    bocpd_free(&state_gaussian);
    bocpd_free(&state_student);
    TEST_PASS("Stats size is properly aligned");
    return 0;
}

int test_stats_indexing_sentinel() {
    // Sentinel test: verify no memory corruption with stride calculation
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    
    // Use max_run_length=10 for manageable test
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 10);
    
    // Perform several updates with varying data
    double cp_prob;
    bocpd_update(&state, 0.5, &cp_prob);
    bocpd_update(&state, 1.2, &cp_prob);
    bocpd_update(&state, -0.3, &cp_prob);
    
    // Verify stats are correctly indexed and stride is correct
    const GaussianNIGStats* stats_0 = (const GaussianNIGStats*)cstats_at(state.stats, 0, state.stats_size);
    const GaussianNIGStats* stats_1 = (const GaussianNIGStats*)cstats_at(state.stats, 1, state.stats_size);
    const GaussianNIGStats* stats_2 = (const GaussianNIGStats*)cstats_at(state.stats, 2, state.stats_size);
    
    // Verify stats have been properly updated (non-zero counts)
    ASSERT_TRUE(stats_0->n > 0);
    ASSERT_TRUE(stats_1->n > 0);
    ASSERT_TRUE(stats_2->n > 0);
    
    // Verify stride calculation: difference between adjacent stats pointers
    size_t ptr_diff_01 = (uint8_t*)stats_1 - (uint8_t*)stats_0;
    size_t ptr_diff_12 = (uint8_t*)stats_2 - (uint8_t*)stats_1;
    ASSERT_EQ(ptr_diff_01, state.stats_size);
    ASSERT_EQ(ptr_diff_12, state.stats_size);
    
    // Verify different run-lengths accumulate different data
    // r=0 is always a fresh start (just the last observation)
    // r=1 includes the last 2 observations
    // r=2 includes all 3 observations
    ASSERT_TRUE(stats_0->n == 1);  // Fresh regime
    ASSERT_TRUE(stats_1->n == 2);  // Continued for 2 steps
    ASSERT_TRUE(stats_2->n == 3);  // Continued for 3 steps
    
    bocpd_free(&state);
    TEST_PASS("Stats indexing sentinel test (no corruption)");
    return 0;
}

int test_vtable_dispatch_correctness() {
    // Verify vtable dispatch produces correct results for both models
    BOCPDState state_gaussian, state_student;
    
    GaussianNIGParams gauss_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    StudentTNGParams student_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0, .nu = 10.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    
    bocpd_init(&state_gaussian, OBS_MODEL_GAUSSIAN_NIG, &gauss_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    bocpd_init(&state_student, OBS_MODEL_STUDENT_T_NG, &student_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    // Feed same data to both
    double test_data[] = {0.1, 0.2, 0.15, 0.12};
    double cp_prob_g, cp_prob_s;
    
    for (int i = 0; i < 4; i++) {
        bocpd_update(&state_gaussian, test_data[i], &cp_prob_g);
        bocpd_update(&state_student, test_data[i], &cp_prob_s);
    }
    
    // Both should produce valid probabilities
    ASSERT_TRUE(cp_prob_g >= 0.0 && cp_prob_g <= 1.0);
    ASSERT_TRUE(cp_prob_s >= 0.0 && cp_prob_s <= 1.0);
    
    // Posteriors should sum to 1
    double sum_g = 0.0, sum_s = 0.0;
    for (int r = 0; r <= 50; r++) {
        sum_g += state_gaussian.posterior_r[r];
        sum_s += state_student.posterior_r[r];
    }
    ASSERT_CLOSE(sum_g, 1.0, 1e-6);
    ASSERT_CLOSE(sum_s, 1.0, 1e-6);
    
    bocpd_free(&state_gaussian);
    bocpd_free(&state_student);
    TEST_PASS("VTable dispatch correctness");
    return 0;
}

int test_large_max_run_length() {
    // Stress test with large max_run_length
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 500.0);
    
    // Large max_run_length tests memory allocation
    int ret = bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
                         HAZARD_CONSTANT, &hazard_params, 1000);
    ASSERT_EQ(ret, 0);
    
    // Perform several updates
    double cp_prob;
    for (int i = 0; i < 50; i++) {
        double* posterior = bocpd_update(&state, 0.1, &cp_prob);
        ASSERT_TRUE(posterior != NULL);
    }
    
    // Verify posterior still sums to 1
    double sum = 0.0;
    for (int r = 0; r <= 1000; r++) {
        sum += state.posterior_r[r];
    }
    ASSERT_CLOSE(sum, 1.0, 1e-6);
    
    bocpd_free(&state);
    TEST_PASS("Large max_run_length stress test");
    return 0;
}

int test_r0_stats_updated_invariant() {
    // CRITICAL: Test that r=0 stats are properly updated each timestep
    // This catches the exact bug we fixed in Phase 1
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    // Update with x1
    double cp_prob;
    bocpd_update(&state, 1.5, &cp_prob);
    
    // Check stats[0] reflects exactly 1 observation
    const GaussianNIGStats* stats_0 = (const GaussianNIGStats*)cstats_at(state.stats, 0, state.stats_size);
    ASSERT_EQ(stats_0->n, 1);
    ASSERT_CLOSE(stats_0->sum_x, 1.5, 1e-9);
    ASSERT_CLOSE(stats_0->sum_x2, 1.5 * 1.5, 1e-9);
    
    // Update with x2
    bocpd_update(&state, -2.3, &cp_prob);
    
    // Check stats[0] is FRESH for x2, not accumulating x1+x2
    stats_0 = (const GaussianNIGStats*)cstats_at(state.stats, 0, state.stats_size);
    ASSERT_EQ(stats_0->n, 1);  // Still 1, not 2!
    ASSERT_CLOSE(stats_0->sum_x, -2.3, 1e-9);  // Just x2, not x1+x2
    ASSERT_CLOSE(stats_0->sum_x2, (-2.3) * (-2.3), 1e-9);
    
    bocpd_free(&state);
    TEST_PASS("r=0 stats updated invariant (Phase 1 bug)");
    return 0;
}

int test_extreme_values() {
    // Test numerical robustness with extreme observations
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_params, 50);
    
    double extreme_values[] = {1e6, -1e6, 1e-6, -1e-6};
    
    for (int i = 0; i < 4; i++) {
        double cp_prob;
        double* posterior = bocpd_update(&state, extreme_values[i], &cp_prob);
        
        ASSERT_TRUE(posterior != NULL);
        ASSERT_TRUE(!isnan(cp_prob) && !isinf(cp_prob));
        
        // Check posterior sums to 1 and has no NaNs
        double sum = 0.0;
        for (int r = 0; r <= state.max_run_length; r++) {
            ASSERT_TRUE(!isnan(posterior[r]));
            sum += posterior[r];
        }
        ASSERT_CLOSE(sum, 1.0, 1e-6);
    }
    
    bocpd_free(&state);
    TEST_PASS("Extreme values numerical robustness");
    return 0;
}

int test_hazard_extremes() {
    // Test with extreme hazard parameters
    BOCPDState state_low, state_high;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    
    // Very high hazard (frequent changepoints)
    ConstantHazardParams hazard_high;
    constant_hazard_init(&hazard_high, 2.0);
    bocpd_init(&state_high, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_high, 50);
    
    // Very low hazard (rare changepoints)
    ConstantHazardParams hazard_low;
    constant_hazard_init(&hazard_low, 1e6);
    bocpd_init(&state_low, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
               HAZARD_CONSTANT, &hazard_low, 50);
    
    // Feed same data to both
    double cp_prob_high, cp_prob_low;
    bocpd_update(&state_high, 0.5, &cp_prob_high);
    bocpd_update(&state_low, 0.5, &cp_prob_low);
    
    // High hazard should have meaningfully higher CP probability (ratio-based, platform-independent)
    ASSERT_TRUE(cp_prob_high > cp_prob_low);
    ASSERT_TRUE(cp_prob_high > 5.0 * cp_prob_low);  // At least 5x higher
    ASSERT_TRUE(cp_prob_high >= 0.0 && cp_prob_high <= 1.0);  // Valid probability
    ASSERT_TRUE(cp_prob_low >= 0.0 && cp_prob_low <= 1.0);    // Valid probability
    
    bocpd_free(&state_high);
    bocpd_free(&state_low);
    TEST_PASS("Hazard extremes");
    return 0;
}

int test_boundary_max_run_length_1() {
    // Boundary test: max_run_length=1
    BOCPDState state;
    GaussianNIGParams obs_params = {
        .mu0 = 0.0, .kappa0 = 1.0, .alpha0 = 1.0, .beta0 = 1.0
    };
    ConstantHazardParams hazard_params;
    constant_hazard_init(&hazard_params, 100.0);
    
    int ret = bocpd_init(&state, OBS_MODEL_GAUSSIAN_NIG, &obs_params,
                         HAZARD_CONSTANT, &hazard_params, 1);
    ASSERT_EQ(ret, 0);
    
    // Do 3 updates
    double cp_prob;
    for (int i = 0; i < 3; i++) {
        double* posterior = bocpd_update(&state, 0.1, &cp_prob);
        ASSERT_TRUE(posterior != NULL);
        
        // Posterior length is 2 (r=0, r=1)
        double sum = 0.0;
        for (int r = 0; r <= 1; r++) {
            sum += posterior[r];
        }
        ASSERT_CLOSE(sum, 1.0, 1e-6);
    }
    
    bocpd_free(&state);
    TEST_PASS("Boundary: max_run_length=1");
    return 0;
}

/* Main test suite runner */
int run_bocpd_core_tests() {
    TEST_SUITE("BOCPD Core");
    
    if (test_bocpd_init() != 0) return 1;
    if (test_bocpd_invalid_params() != 0) return 1;
    if (test_bocpd_reset() != 0) return 1;
    if (test_bocpd_single_update() != 0) return 1;
    if (test_bocpd_changepoint_detection() != 0) return 1;
    if (test_bocpd_batch_update() != 0) return 1;
    if (test_bocpd_known_values() != 0) return 1;
    if (test_get_map_run_length() != 0) return 1;
    if (test_get_posterior() != 0) return 1;
    
    // Phase 1 refactor validation tests
    if (test_stats_alignment() != 0) return 1;
    if (test_stats_indexing_sentinel() != 0) return 1;
    if (test_vtable_dispatch_correctness() != 0) return 1;
    if (test_large_max_run_length() != 0) return 1;
    
    // Critical correctness & robustness tests
    if (test_r0_stats_updated_invariant() != 0) return 1;
    if (test_extreme_values() != 0) return 1;
    if (test_hazard_extremes() != 0) return 1;
    if (test_boundary_max_run_length_1() != 0) return 1;
    
    return 0;
}
