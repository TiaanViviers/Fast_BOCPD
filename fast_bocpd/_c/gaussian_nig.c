#include <math.h>

#include "gaussian_nig.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================================
// == Validation Helpers =======================================================
// =============================================================================

static int gaussian_nig_validate_params(const GaussianNIGParams* params)
{
    if (!params) {
        return -1;
    }
    if (!isfinite(params->kappa0) || params->kappa0 <= 0.0) {
        return -1;
    }
    if (!isfinite(params->alpha0) || params->alpha0 <= 0.0) {
        return -1;
    }
    if (!isfinite(params->beta0) || params->beta0 <= 0.0) {
        return -1;
    }
    if (!isfinite(params->mu0)) {
        return -1;
    }
    return 0;
}

static int gaussian_nig_validate_stats(const GaussianNIGStats* stats)
{
    if (!stats) {
        return -1;
    }
    if (stats->n < 0) {
        return -1;
    }
    if (!isfinite(stats->sum_x) || !isfinite(stats->sum_x2)) {
        return -1;
    }
    if (stats->sum_x2 < 0.0) {
        return -1;
    }
    return 0;
}

// =============================================================================
// == Public API ===============================================================
// =============================================================================

void gaussian_nig_prior_stats(GaussianNIGStats* stats)
{
    stats->n = 0;
    stats->sum_x = 0.0;
    stats->sum_x2 = 0.0;
}

void gaussian_nig_update_stats(GaussianNIGStats* stats, double x)
{
    if (!stats) {
        return;
    }
    stats->n += 1;
    stats->sum_x += x;
    stats->sum_x2 += x * x;
}

static void compute_posterior_hyperparams(
    const GaussianNIGParams* params,
    const GaussianNIGStats* stats,
    double* mu_n,
    double* kappa_n,
    double* alpha_n,
    double* beta_n)
{
    if (stats->n == 0) {
        // No data yet: posterior == prior
        *mu_n = params->mu0;
        *kappa_n = params->kappa0;
        *alpha_n = params->alpha0;
        *beta_n = params->beta0;
        return;
    }

    double n = (double)stats->n;
    double x_bar = stats->sum_x / n;
    // Sum of squared deviations from the mean
    double S = stats->sum_x2 - n * x_bar * x_bar;

    *kappa_n = params->kappa0 + n;
    *mu_n = (params->kappa0 * params->mu0 + n * x_bar) / (*kappa_n);
    *alpha_n = params->alpha0 + 0.5 * n;

    double diff = x_bar - params->mu0;
    *beta_n = params->beta0 + 0.5 * (
        S + (params->kappa0 * n / (*kappa_n)) * diff * diff
    );
}

double gaussian_nig_predictive_logpdf(
    const GaussianNIGParams* params,
    const GaussianNIGStats* stats,
    double x)
{
    if (gaussian_nig_validate_params(params) != 0) {
        return -INFINITY;
    }
    if (gaussian_nig_validate_stats(stats) != 0) {
        return -INFINITY;
    }
    if (!isfinite(x)) {
        return -INFINITY;
    }

    // Compute posterior hyperparameters
    double mu_n, kappa_n, alpha_n, beta_n;
    compute_posterior_hyperparams(params, stats, &mu_n, &kappa_n, &alpha_n, &beta_n);

    // Student-t parameters
    double nu = 2.0 * alpha_n;  // degrees of freedom
    double scale2 = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n);

    if (!isfinite(nu) || nu <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(scale2) || scale2 <= 0.0) {
        return -INFINITY;
    }

    double scale = sqrt(scale2);

    double z = (x - mu_n) / scale;

    // Student-t log pdf
    // log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(νπ) - log(scale) - (ν+1)/2 * log(1 + z²/ν)
    double log_norm = lgamma((nu + 1.0) / 2.0)
                    - lgamma(nu / 2.0)
                    - 0.5 * log(nu * M_PI)
                    - log(scale);
    
    double log_kernel = -(nu + 1.0) / 2.0 * log1p((z * z) / nu);

    return log_norm + log_kernel;
}
