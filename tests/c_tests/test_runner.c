#include <stdio.h>
#include "test_utils.h"

/* Declare test suite runners */
int run_gaussian_nig_tests();
int run_student_t_ng_tests();
int run_student_t_ng_grid_tests();
int run_poisson_gamma_tests();
int run_bernoulli_beta_tests();
int run_binomial_beta_tests();
int run_gamma_gamma_tests();
int run_hazard_tests();
int run_bocpd_core_tests();

/* Test counters */
int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

int main() {
    printf("\n");
    printf("========================================\n");
    printf("  Fast BOCPD C Unit Tests\n");
    printf("========================================\n");
    
    /* Run all test suites */
    int failed = 0;
    failed += run_gaussian_nig_tests();
    failed += run_student_t_ng_tests();
    failed += run_student_t_ng_grid_tests();
    failed += run_poisson_gamma_tests();
    failed += run_bernoulli_beta_tests();
    failed += run_binomial_beta_tests();
    failed += run_gamma_gamma_tests();
    failed += run_hazard_tests();
    failed += run_bocpd_core_tests();
    
    /* Print result */
    printf("\n========================================\n");
    if (failed == 0) {
        printf(COLOR_GREEN "✓ All tests passed!\n" COLOR_RESET);
        printf("========================================\n");
        return 0;
    } else {
        printf(COLOR_RED "✗ Some tests failed\n" COLOR_RESET);
        printf("========================================\n");
        return 1;
    }
}
