#include "test_utils.h"
#include "../../fast_bocpd/_c/gaussian_nig.h"

int test_prior_stats() {
    GaussianNIGStats stats;
    gaussian_nig_prior_stats(&stats);
    
    ASSERT_EQ(stats.n, 0);
    ASSERT_CLOSE(stats.sum_x, 0.0, 1e-10);
    ASSERT_CLOSE(stats.sum_x2, 0.0, 1e-10);
    
    TEST_PASS("Prior stats initialization");
    return 0;
}

int test_update_stats() {
    GaussianNIGStats stats;
    gaussian_nig_prior_stats(&stats);
    
    // Add one observation
    gaussian_nig_update_stats(&stats, 1.5);
    ASSERT_EQ(stats.n, 1);
    ASSERT_CLOSE(stats.sum_x, 1.5, 1e-10);
    ASSERT_CLOSE(stats.sum_x2, 2.25, 1e-10);
    
    // Add another observation
    gaussian_nig_update_stats(&stats, 2.5);
    ASSERT_EQ(stats.n, 2);
    ASSERT_CLOSE(stats.sum_x, 4.0, 1e-10);
    ASSERT_CLOSE(stats.sum_x2, 8.5, 1e-10);
    
    TEST_PASS("Stats update with observations");
    return 0;
}

int test_predictive_logpdf_prior() {
    GaussianNIGParams params = {
        .mu0 = 0.0,
        .kappa0 = 1.0,
        .alpha0 = 1.0,
        .beta0 = 1.0
    };
    
    GaussianNIGStats prior_stats;
    gaussian_nig_prior_stats(&prior_stats);
    
    // Test at mean
    double logpdf = gaussian_nig_predictive_logpdf(&params, &prior_stats, 0.0);
    ASSERT_TRUE(isfinite(logpdf));
    
    // Predictive should be symmetric around mu0
    double logpdf_pos = gaussian_nig_predictive_logpdf(&params, &prior_stats, 1.0);
    double logpdf_neg = gaussian_nig_predictive_logpdf(&params, &prior_stats, -1.0);
    ASSERT_CLOSE(logpdf_pos, logpdf_neg, 1e-10);
    
    TEST_PASS("Predictive logpdf with prior");
    return 0;
}

int test_predictive_logpdf_posterior() {
    GaussianNIGParams params = {
        .mu0 = 0.0,
        .kappa0 = 1.0,
        .alpha0 = 1.0,
        .beta0 = 1.0
    };
    
    // Create stats with some observations around mean=5
    GaussianNIGStats stats;
    gaussian_nig_prior_stats(&stats);
    gaussian_nig_update_stats(&stats, 5.0);
    gaussian_nig_update_stats(&stats, 5.1);
    gaussian_nig_update_stats(&stats, 4.9);
    
    // Should have higher probability near 5.0 than far away
    double logpdf_near = gaussian_nig_predictive_logpdf(&params, &stats, 5.0);
    double logpdf_far = gaussian_nig_predictive_logpdf(&params, &stats, 10.0);
    ASSERT_TRUE(logpdf_near > logpdf_far);
    
    TEST_PASS("Predictive logpdf with posterior");
    return 0;
}

int test_numerical_stability() {
    GaussianNIGParams params = {
        .mu0 = 0.0,
        .kappa0 = 1.0,
        .alpha0 = 1.0,
        .beta0 = 1.0
    };
    
    GaussianNIGStats stats;
    gaussian_nig_prior_stats(&stats);
    
    // Test with extreme values
    double logpdf1 = gaussian_nig_predictive_logpdf(&params, &stats, 100.0);
    double logpdf2 = gaussian_nig_predictive_logpdf(&params, &stats, -100.0);
    
    ASSERT_TRUE(isfinite(logpdf1));
    ASSERT_TRUE(isfinite(logpdf2));
    
    TEST_PASS("Numerical stability with extreme values");
    return 0;
}

/* Main test suite runner */
int run_gaussian_nig_tests() {
    TEST_SUITE("GaussianNIG");
    
    if (test_prior_stats() != 0) return 1;
    if (test_update_stats() != 0) return 1;
    if (test_predictive_logpdf_prior() != 0) return 1;
    if (test_predictive_logpdf_posterior() != 0) return 1;
    if (test_numerical_stability() != 0) return 1;
    
    return 0;
}
