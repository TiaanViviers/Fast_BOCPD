#ifndef GAUSSIAN_NIG_H
#define GAUSSIAN_NIG_H

#include <stdint.h>

/**
 * GaussianNIG hyperparameters (prior)
 */
typedef struct {
    double mu0;
    double kappa0;
    double alpha0;
    double beta0;
} GaussianNIGParams;

/**
 * GaussianNIG sufficient statistics for a run
 */
typedef struct {
    int32_t n;
    double sum_x;
    double sum_x2;
} GaussianNIGStats;

/**
 * Initialize prior statistics (empty run)
 */
void gaussian_nig_prior_stats(GaussianNIGStats* stats);

/**
 * Update sufficient statistics with new observation
 * 
 * @param stats     Current statistics (will be modified in-place)
 * @param x         New observation
 */
void gaussian_nig_update_stats(GaussianNIGStats* stats, double x);

/**
 * Compute predictive log probability density
 * 
 * @param params    Model hyperparameters
 * @param stats     Current sufficient statistics
 * @param x         New observation to predict
 * @return          Log predictive density
 */
double gaussian_nig_predictive_logpdf(
    const GaussianNIGParams* params,
    const GaussianNIGStats* stats,
    double x
);

#endif // GAUSSIAN_NIG_H
