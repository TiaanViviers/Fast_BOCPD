#include <math.h>
#include <string.h>

#include "poisson_gamma.h"

#define POISSON_INT_TOL 1e-9

// =============================================================================
// == Validation Helpers =======================================================
// =============================================================================

static int validate_params(const PoissonGammaParams* params)
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
    return 0;
}

static int validate_stats(const PoissonGammaStats* stats)
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

// Normalize/validate observation count using tolerance.
static int to_valid_count(double x, int64_t* out)
{
    if (!isfinite(x) || x < 0.0) {
        return -1;
    }
    
    double rounded = nearbyint(x);
    if (fabs(x - rounded) > POISSON_INT_TOL) {
        return -1;
    }
    
    if (rounded > (double)INT64_MAX) {
        return -1;
    }
    
    *out = (int64_t)rounded;
    return 0;
}

// =============================================================================
// == Public API ===============================================================
// =============================================================================

size_t poisson_gamma_stats_size(void)
{
    return sizeof(PoissonGammaStats);
}

void poisson_gamma_prior_stats(PoissonGammaStats* stats)
{
    stats->n = 0;
    stats->sum_x = 0.0;
}

void poisson_gamma_update_stats(
    PoissonGammaStats* stats,
    const PoissonGammaParams* params,
    double x)
{
    (void)params;  // Unused for this model
    if (!stats) {
        return;
    }
    stats->n++;
    stats->sum_x += x;
}

double poisson_gamma_predictive_logpdf(
    const PoissonGammaParams* params,
    const PoissonGammaStats* stats,
    double x)
{
    if (validate_params(params) != 0) {
        return -INFINITY;
    }
    if (validate_stats(stats) != 0) {
        return -INFINITY;
    }

    int64_t count = 0;
    if (to_valid_count(x, &count) != 0) {
        return -INFINITY;
    }
    
    // Posterior parameters
    double alpha_n = params->alpha0 + stats->sum_x;
    double beta_n = params->beta0 + (double)stats->n;
    
    // Guard against numerical issues in posterior
    if (!isfinite(alpha_n) || !isfinite(beta_n) || 
        !(alpha_n > 0.0) || !(beta_n > 0.0)) {
        return -INFINITY;
    }
    
    // Predictive: Negative Binomial
    // log p(x) = lgamma(α_n + x) - lgamma(α_n) - lgamma(x + 1)
    //          + α_n * log(β_n / (β_n + 1))
    //          + x * log(1 / (β_n + 1))
    
    // Stable log computations (avoid cancellation)
    double log_p_success = -log1p(1.0 / beta_n);  // log(β_n / (β_n + 1))
    double log_p_fail = -log1p(beta_n);           // log(1 / (β_n + 1))
    
    double logpdf = lgamma(alpha_n + (double)count)
                  - lgamma(alpha_n)
                  - lgamma((double)count + 1.0)
                  + alpha_n * log_p_success
                  + (double)count * log_p_fail;
    
    return logpdf;
}

void poisson_gamma_copy_stats(void* dst, const void* src)
{
    memcpy(dst, src, sizeof(PoissonGammaStats));
}
