/**
 * Gamma-Gamma model (fixed shape, unknown rate)
 * 
 * Conjugate Bayesian model for continuous, non-negative observations.
 * 
 * Model specification:
 *   Likelihood: x ~ Gamma(shape=k, rate=λ)  where k is FIXED (known constant)
 *   Prior:      λ ~ Gamma(α₀, rate=β₀)     where λ is UNKNOWN
 *   Posterior:  λ | x ~ Gamma(α_n, rate=β_n)
 *   Predictive: Beta-prime / Lomax-like (closed form)
 * 
 * Key assumptions:
 *   - Shape parameter k is FIXED (not learned)
 *   - Rate parameter λ has Gamma prior (conjugate)
 *   - Parameterization uses RATE (not scale): higher rate → smaller values
 *   - Special case: k=1 reduces to Exponential(λ) likelihood
 * 
 * Sufficient statistics:
 *   - n: number of observations
 *   - sum_x: sum of observations
 * 
 * Posterior update (conjugate):
 *   α_n = α₀ + n·k
 *   β_n = β₀ + Σxᵢ
 * 
 * Use cases:
 *   - Positive continuous data (durations, amounts, inter-arrival times)
 *   - Waiting times / survival data (k=1 for Exponential)
 *   - Right-skewed distributions
 * 
 * Implementation notes:
 *   - Recommend shape >= 1 to avoid infinite density at x=0
 *   - For shape < 1, x=0 returns -inf (prevents BOCPD poisoning)
 *   - Cached lgamma(k) for efficiency
 *   - Numerically stable log-space computation with log1p
 */

#ifndef GAMMA_GAMMA_FIXED_SHAPE_H
#define GAMMA_GAMMA_FIXED_SHAPE_H

#include <stddef.h>
#include <stdint.h>

/**
 * Parameters for Gamma-Gamma model.
 * 
 * Prior: λ ~ Gamma(α₀, rate=β₀)
 * Likelihood: x ~ Gamma(k, rate=λ) with k FIXED
 */
typedef struct {
    double alpha0;        // Prior shape parameter on λ (must be > 0)
    double beta0;         // Prior rate parameter on λ (must be > 0, NOT scale!)
    double shape;         // Fixed shape k of likelihood (recommend >= 1)
    double log_gamma_k;   // Cached lgamma(shape) for efficiency
} GammaGammaParams;

/**
 * Sufficient statistics for Gamma-Gamma model.
 * 
 * Tracks count and sum of observations for O(1) updates.
 */
typedef struct {
    int32_t n;      // Number of observations in this run
    double sum_x;   // Sum of observations
} GammaGammaStats;

/**
 * Get size of statistics structure.
 * 
 * Returns:
 *   Size in bytes of GammaGammaStats
 */
size_t gamma_gamma_stats_size(void);

/**
 * Initialize statistics to prior (empty run).
 * 
 * Args:
 *   stats: Statistics structure to initialize
 */
void gamma_gamma_prior_stats(GammaGammaStats* stats);

/**
 * Update sufficient statistics with new observation.
 * 
 * This is a lightweight operation (no validation):
 *   - Increments count: n += 1
 *   - Updates sum: sum_x += x
 * 
 * Validation happens in predictive_logpdf (Python strict mode).
 * 
 * Args:
 *   stats: Statistics to update (modified in-place)
 *   params: Model parameters (unused, for API consistency)
 *   x: New observation
 */
void gamma_gamma_update_stats(GammaGammaStats* stats,
                              const GammaGammaParams* params,
                              double x);

/**
 * Compute predictive log probability density.
 * 
 * Marginalizes over λ to get p(x | α_n, β_n, k) using closed-form
 * Beta-prime-like distribution.
 * 
 * Formula (log space, numerically stable):
 *   log p(x) = lgamma(α_n + k) - lgamma(α_n) - lgamma(k)
 *            + α_n·log(β_n) + (k-1)·log(x)
 *            - (α_n + k)·log(β_n + x)
 * 
 * where α_n = α₀ + n·k, β_n = β₀ + sum_x
 * 
 * Edge cases:
 *   - x < 0: returns -∞ (invalid domain)
 *   - x = 0 and k > 1: returns -∞ (zero density at origin)
 *   - x = 0 and k = 1: special case (Exponential at origin)
 *   - x = 0 and k < 1: returns -∞ (avoids +∞ BOCPD poisoning)
 *   - Invalid params/stats: returns -∞ (defensive)
 * 
 * Args:
 *   params: Model parameters (validated defensively)
 *   stats: Current sufficient statistics (validated defensively)
 *   x: Query point
 * 
 * Returns:
 *   Log probability density, or -∞ if invalid
 */
double gamma_gamma_predictive_logpdf(const GammaGammaParams* params,
                                     const GammaGammaStats* stats,
                                     double x);

/**
 * Copy statistics structure.
 * 
 * Args:
 *   dst: Destination
 *   src: Source
 */
void gamma_gamma_copy_stats(void* dst, const void* src);

#endif  // GAMMA_GAMMA_FIXED_SHAPE_H
