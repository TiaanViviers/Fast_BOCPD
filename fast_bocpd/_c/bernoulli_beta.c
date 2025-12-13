#include <math.h>
#include <string.h>

#include "bernoulli_beta.h"

// =============================================================================
// == Public API ===============================================================
// =============================================================================

size_t bernoulli_beta_stats_size(void)
{
    return sizeof(BernoulliBetaStats);
}

void bernoulli_beta_prior_stats(BernoulliBetaStats* stats)
{
    stats->n = 0;
    stats->sum_x = 0.0;
}

void bernoulli_beta_update_stats(
    BernoulliBetaStats* stats,
    const BernoulliBetaParams* params,
    double x
)
{
    (void)params;  // Unused
    stats->n++;
    stats->sum_x += x;
}


double bernoulli_beta_predictive_logpdf(
    const BernoulliBetaParams* params,
    const BernoulliBetaStats* stats,
    double x
)
{
    if (!params || !stats) {
        return -INFINITY;
    }

#ifdef BOCPD_DEBUG_CHECKS
    // Validate parameters (only in debug builds - checked in bocpd_init)
    if (!isfinite(params->alpha0) || !isfinite(params->beta0) ||
        params->alpha0 <= 0.0 || params->beta0 <= 0.0) {
        return -INFINITY;
    }
#endif
    
    // Validate observation
    if (!isfinite(x)) {
        return -INFINITY;
    }
    
    // Snap to nearest integer and check binary-ness
    double xr = nearbyint(x);
    if (fabs(x - xr) > 1e-9) {
        return -INFINITY;  // Not close enough to an integer
    }
    if (!(xr == 0.0 || xr == 1.0)) {
        return -INFINITY;  // Not binary (0 or 1)
    }
    
    // Validate stats (guards against strict=False corruption)
    double s = stats->sum_x;
    if (!isfinite(s) || s < 0.0 || s > (double)stats->n) {
        return -INFINITY;
    }
    
    // Compute posterior parameters
    double alpha_n = params->alpha0 + s;
    double beta_n = params->beta0 + ((double)stats->n - s);
    
    // Guard against numerical issues
    if (!(alpha_n > 0.0) || !(beta_n > 0.0)) {
        return -INFINITY;
    }
    
    // Compute log probability
    // P(x=1 | data) = α_n / (α_n + β_n)
    // P(x=0 | data) = β_n / (α_n + β_n)
    double total = alpha_n + beta_n;
    double log_total = log(total);
    
    return (xr == 1.0) ? (log(alpha_n) - log_total)
                       : (log(beta_n) - log_total);
}


void bernoulli_beta_copy_stats(BernoulliBetaStats* dst,
                               const BernoulliBetaStats* src)
{
    memcpy(dst, src, sizeof(BernoulliBetaStats));
}
