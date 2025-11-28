#include "test_utils.h"
#include "../../fast_bocpd/_c/hazard.h"
#include <math.h>

int test_constant_hazard_init() {
    ConstantHazardParams params;
    
    // Valid initialization
    int ret = constant_hazard_init(&params, 100.0);
    ASSERT_EQ(ret, 0);
    ASSERT_CLOSE(params.lambda, 100.0, 1e-10);
    
    TEST_PASS("Constant hazard initialization");
    return 0;
}

int test_constant_hazard_invalid_lambda() {
    ConstantHazardParams params;
    
    // Negative lambda should fail
    ASSERT_EQ(constant_hazard_init(&params, -1.0), -1);
    
    // Zero lambda should fail
    ASSERT_EQ(constant_hazard_init(&params, 0.0), -1);
    
    TEST_PASS("Rejects invalid lambda values");
    return 0;
}

int test_constant_hazard_precomputed_values() {
    ConstantHazardParams params;
    constant_hazard_init(&params, 100.0);
    
    double expected_H = 1.0 / 100.0;
    double expected_log_H = log(expected_H);
    double expected_log_1mH = log(1.0 - expected_H);
    
    ASSERT_CLOSE(params.log_H, expected_log_H, 1e-10);
    ASSERT_CLOSE(params.log_1mH, expected_log_1mH, 1e-10);
    
    TEST_PASS("Precomputed log values are correct");
    return 0;
}

int test_constant_hazard_transitions() {
    ConstantHazardParams params;
    constant_hazard_init(&params, 100.0);
    
    // For constant hazard, transitions are independent of r_prev
    double log_cp_0 = constant_hazard_log_transition_cp(&params, 0);
    double log_cp_50 = constant_hazard_log_transition_cp(&params, 50);
    double log_cp_100 = constant_hazard_log_transition_cp(&params, 100);
    
    ASSERT_CLOSE(log_cp_0, log_cp_50, 1e-10);
    ASSERT_CLOSE(log_cp_0, log_cp_100, 1e-10);
    
    double log_cont_0 = constant_hazard_log_transition_cont(&params, 0);
    double log_cont_50 = constant_hazard_log_transition_cont(&params, 50);
    
    ASSERT_CLOSE(log_cont_0, log_cont_50, 1e-10);
    
    TEST_PASS("Transition probabilities are constant");
    return 0;
}

int test_constant_hazard_probability_sum() {
    ConstantHazardParams params;
    constant_hazard_init(&params, 100.0);
    
    // P(cp) + P(cont) should equal 1
    double log_cp = constant_hazard_log_transition_cp(&params, 0);
    double log_cont = constant_hazard_log_transition_cont(&params, 0);
    
    double p_cp = exp(log_cp);
    double p_cont = exp(log_cont);
    
    ASSERT_CLOSE(p_cp + p_cont, 1.0, 1e-10);
    
    TEST_PASS("Probabilities sum to 1");
    return 0;
}

int test_constant_hazard_different_lambdas() {
    ConstantHazardParams params1, params2;
    constant_hazard_init(&params1, 50.0);
    constant_hazard_init(&params2, 200.0);
    
    double log_cp1 = constant_hazard_log_transition_cp(&params1, 0);
    double log_cp2 = constant_hazard_log_transition_cp(&params2, 0);
    
    // Smaller lambda = higher hazard = higher CP probability
    ASSERT_TRUE(log_cp1 > log_cp2);
    
    TEST_PASS("Different lambdas produce different probabilities");
    return 0;
}

/* Main test suite runner */
int run_hazard_tests() {
    TEST_SUITE("Hazard Functions");
    
    if (test_constant_hazard_init() != 0) return 1;
    if (test_constant_hazard_invalid_lambda() != 0) return 1;
    if (test_constant_hazard_precomputed_values() != 0) return 1;
    if (test_constant_hazard_transitions() != 0) return 1;
    if (test_constant_hazard_probability_sum() != 0) return 1;
    if (test_constant_hazard_different_lambdas() != 0) return 1;
    
    return 0;
}
