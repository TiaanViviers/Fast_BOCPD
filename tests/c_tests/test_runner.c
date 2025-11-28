#include <stdio.h>
#include "test_utils.h"

/* Declare test suite runners */
int run_gaussian_nig_tests();
int run_hazard_tests();
int run_bocpd_core_tests();

int main() {
    printf("\n");
    printf("========================================\n");
    printf("  Fast BOCPD C Unit Tests\n");
    printf("========================================\n");
    
    int failed = 0;
    
    /* Run all test suites */
    if (run_gaussian_nig_tests() != 0) {
        printf(COLOR_RED "\n✗ GaussianNIG tests failed\n" COLOR_RESET);
        failed = 1;
    }
    
    if (run_hazard_tests() != 0) {
        printf(COLOR_RED "\n✗ Hazard tests failed\n" COLOR_RESET);
        failed = 1;
    }
    
    if (run_bocpd_core_tests() != 0) {
        printf(COLOR_RED "\n✗ BOCPD Core tests failed\n" COLOR_RESET);
        failed = 1;
    }
    
    /* Print summary */
    printf("\n========================================\n");
    if (failed) {
        printf(COLOR_RED "  ✗ TESTS FAILED\n" COLOR_RESET);
        printf("========================================\n\n");
        return 1;
    } else {
        printf(COLOR_GREEN "  ✓ ALL TESTS PASSED\n" COLOR_RESET);
        printf("========================================\n\n");
        return 0;
    }
}
