// filepath: /home/tiaan/Projects/Fast_BOCPD/fast_bocpd/_c/poisson_gamma.c
#include "poisson_gamma.h"
#include <math.h>
#include <string.h>

size_t poisson_gamma_stats_size(void) {
    return sizeof(PoissonGammaStats);
}

void poisson_gamma_prior_stats(PoissonGammaStats* stats) {
    stats->n = 0;
    stats->sum_x = 0.0;
}

void poisson_gamma_update_stats(
    PoissonGammaStats* stats,
    const PoissonGammaParams* params,
    double x
) {
    (void)params;  // Unused for this model
    stats->n++;
    stats->sum_x += x;
}

double poisson_gamma_predictive_logpdf(
    const PoissonGammaParams* params,
    const PoissonGammaStats* stats,
    double x
) {
    // Guard against invalid inputs
    if (!isfinite(x) || x < 0.0) {
        return -INFINITY;
    }
    
    // Guard against bad hyperparameters (cheap insurance)
    if (!isfinite(params->alpha0) || !isfinite(params->beta0) ||
        !(params->alpha0 > 0.0) || !(params->beta0 > 0.0)) {
        return -INFINITY;
    }
    
    // Check integer-ness (tolerance 1e-9)
    double xr = nearbyint(x);
    if (fabs(x - xr) > 1e-9) {
        return -INFINITY;
    }
    
    // Clamp out-of-range integers
    if (xr > (double)INT64_MAX) {
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
    double log_p_fail = -log1p(beta_n);            // log(1 / (β_n + 1))
    
    double logpdf = lgamma(alpha_n + xr) 
                  - lgamma(alpha_n)
                  - lgamma(xr + 1.0)  // No int overflow
                  + alpha_n * log_p_success
                  + xr * log_p_fail;
    
    return logpdf;
}

void poisson_gamma_copy_stats(void* dst, const void* src) {
    memcpy(dst, src, sizeof(PoissonGammaStats));
}