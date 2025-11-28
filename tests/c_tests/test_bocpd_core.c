#include "test_utils.h"
#include "../../fast_bocpd/_c/bocpd_core.h"
#include "../../fast_bocpd/_c/gaussian_nig.h"
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
    
    // Known test case
    double test_data[] = {0.1, 0.2, 0.15, 5.0, 5.1, 4.9};
    double expected_probs[] = {0.010000, 0.006854, 0.005487, 0.188768, 0.003517, 0.001671};
    
    for (int i = 0; i < 6; i++) {
        double cp_prob;
        bocpd_update(&state, test_data[i], &cp_prob);
        ASSERT_CLOSE(cp_prob, expected_probs[i], 1e-4);
    }
    
    bocpd_free(&state);
    TEST_PASS("Matches known reference values");
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
    
    return 0;
}
