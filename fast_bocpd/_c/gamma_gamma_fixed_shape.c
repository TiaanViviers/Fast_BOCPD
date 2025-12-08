/**
 * Gamma-Gamma model implementation (fixed shape, unknown rate)
 */

#include "gamma_gamma_fixed_shape.h"
#include <math.h>
#include <string.h>

size_t gamma_gamma_stats_size(void) {
    return sizeof(GammaGammaStats);
}

void gamma_gamma_prior_stats(GammaGammaStats* stats) {
    stats->n = 0;
    stats->sum_x = 0.0;
}

void gamma_gamma_update_stats(GammaGammaStats* stats,
                              const GammaGammaParams* params,
                              double x) {
    (void)params;  // Unused; kept for API consistency
    stats->n++;
    stats->sum_x += x;
}

double gamma_gamma_predictive_logpdf(const GammaGammaParams* params,
                                     const GammaGammaStats* stats,
                                     double x) {
    // Defensive: validate parameters
    if (!params || !stats) {
        return -INFINITY;
    }
    
    // Validate params are finite and positive
    if (!isfinite(params->alpha0) || params->alpha0 <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(params->beta0) || params->beta0 <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(params->shape) || params->shape <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(params->log_gamma_k)) {
        return -INFINITY;  // Corrupted cache
    }
    
    // Validate stats
    if (stats->n < 0) {
        return -INFINITY;
    }
    if (!isfinite(stats->sum_x) || stats->sum_x < 0.0) {
        return -INFINITY;
    }
    
    // Validate observation x
    if (!isfinite(x)) {
        return -INFINITY;
    }
    if (x < 0.0) {
        return -INFINITY;  // Domain violation
    }
    
    // Handle x = 0 edge case
    if (x == 0.0) {
        if (params->shape > 1.0) {
            // Gamma(k > 1) has zero density at origin
            return -INFINITY;
        }
        // k == 1 (Exponential): special case to avoid 0 * log(0) NaN
        // Predictive at x=0 for Exponential:
        //   p(0) = α_n / β_n  (after marginalizing λ)
        // So log p(0) = log(α_n) - log(β_n)
        double k = params->shape;
        double alpha_n = params->alpha0 + (double)stats->n * k;
        double beta_n = params->beta0 + stats->sum_x;
        
        // Sanity check posterior params
        if (!isfinite(alpha_n) || alpha_n <= 0.0) {
            return -INFINITY;
        }
        if (!isfinite(beta_n) || beta_n <= 0.0) {
            return -INFINITY;
        }
        
        return log(alpha_n) - log(beta_n);
    }
    
    // General case: x > 0
    // Compute posterior parameters
    double k = params->shape;
    double alpha_n = params->alpha0 + (double)stats->n * k;
    double beta_n = params->beta0 + stats->sum_x;
    
    // Validate posterior params (should be positive if params valid)
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
    double log_gamma_k = params->log_gamma_k;  // Use cached value
    
    double term1 = log_gamma_alpha_n_plus_k - log_gamma_alpha_n - log_gamma_k;
    double term2 = alpha_n * log(beta_n);
    double term3 = (k - 1.0) * log(x);
    double term4 = -(alpha_n + k) * log(beta_n + x);
    
    double logp = term1 + term2 + term3 + term4;
    
    // Final sanity check (should be <= 0 for valid PDF)
    if (!isfinite(logp)) {
        return -INFINITY;
    }
    
    return logp;
}

void gamma_gamma_copy_stats(void* dst, const void* src) {
    memcpy(dst, src, sizeof(GammaGammaStats));
}
