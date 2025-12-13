#include <math.h>
#include <string.h>

#include "gamma_gamma_fixed_shape.h"

#define GAMMA_K_TOL 1e-12

// =============================================================================
// == Validation Helpers =======================================================
// =============================================================================

static int validate_params(const GammaGammaParams* params)
{
    if (!params) {
        return -1;
    }
    if (!isfinite(params->alpha0) || params->alpha0 <= 0.0) {
        return -1;
    }
    if (!isfinite(params->beta0) || params->beta0 <= 0.0) {
        return -1;
    }
    if (!isfinite(params->shape) || params->shape <= 0.0) {
        return -1;
    }
    if (!isfinite(params->log_gamma_k)) {
        return -1;
    }
    return 0;
}

static int validate_stats(const GammaGammaStats* stats)
{
    if (!stats) {
        return -1;
    }
    if (stats->n < 0) {
        return -1;
    }
    if (!isfinite(stats->sum_x) || stats->sum_x < 0.0) {
        return -1;
    }
    return 0;
}

static double predictive_at_zero(const GammaGammaParams* params,
                                 const GammaGammaStats* stats)
{
    double k = params->shape;

    if (k > 1.0 + GAMMA_K_TOL) {
        return -INFINITY;
    }
    if (k < 1.0 - GAMMA_K_TOL) {
        return -INFINITY;
    }

    double alpha_n = params->alpha0 + (double)stats->n * k;
    double beta_n = params->beta0 + stats->sum_x;

    if (!isfinite(alpha_n) || alpha_n <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(beta_n) || beta_n <= 0.0) {
        return -INFINITY;
    }

    return log(alpha_n) - log(beta_n);
}

// =============================================================================
// == Public API ===============================================================
// =============================================================================

size_t gamma_gamma_stats_size(void)
{
    return sizeof(GammaGammaStats);
}

void gamma_gamma_prior_stats(GammaGammaStats* stats)
{
    stats->n = 0;
    stats->sum_x = 0.0;
}

void gamma_gamma_update_stats(GammaGammaStats* stats,
                              const GammaGammaParams* params,
                              double x)
{
    (void)params;
    stats->n++;
    stats->sum_x += x;
}

double gamma_gamma_predictive_logpdf(const GammaGammaParams* params,
                                     const GammaGammaStats* stats,
                                     double x)
{
    if (validate_params(params) != 0) {
        return -INFINITY;
    }
    if (validate_stats(stats) != 0) {
        return -INFINITY;
    }

    if (!isfinite(x)) {
        return -INFINITY;
    }
    if (x < 0.0) {
        return -INFINITY;
    }

    if (x == 0.0) {
        return predictive_at_zero(params, stats);
    }

    double k = params->shape;
    double alpha_n = params->alpha0 + (double)stats->n * k;
    double beta_n = params->beta0 + stats->sum_x;
    
    if (!isfinite(alpha_n) || alpha_n <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(beta_n) || beta_n <= 0.0) {
        return -INFINITY;
    }
    
    // Compute predictive log density (numerically stable):
    // log p(x) = lgamma(α_n + k) - lgamma(α_n) - lgamma(k)
    //          + α_n·log(β_n) + (k-1)·log(x)
    //          - (α_n + k)·log(β_n + x)
    double log_gamma_alpha_n_plus_k = lgamma(alpha_n + k);
    double log_gamma_alpha_n = lgamma(alpha_n);
    double log_gamma_k = params->log_gamma_k;
    
    double term1 = log_gamma_alpha_n_plus_k - log_gamma_alpha_n - log_gamma_k;
    double term2 = alpha_n * log(beta_n);
    double term3 = (k - 1.0) * log(x);
    
    double log_beta_n_plus_x = log(beta_n) + log1p(x / beta_n);
    double term4 = -(alpha_n + k) * log_beta_n_plus_x;
    
    double logp = term1 + term2 + term3 + term4;
    
    if (!isfinite(logp)) {
        return -INFINITY;
    }
    
    return logp;
}

void gamma_gamma_copy_stats(void* dst, const void* src)
{
    memcpy(dst, src, sizeof(GammaGammaStats));
}
