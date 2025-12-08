#ifndef BINOMIAL_BETA_H
#define BINOMIAL_BETA_H

#include <stddef.h>
#include <stdint.h>

/**
 * Binomial-Beta conjugate model for count data with fixed number of trials.
 * 
 * Model:
 *   k | p ~ Binomial(N, p)
 *   p ~ Beta(alpha0, beta0)
 * 
 * Predictive: Beta-Binomial distribution
 * 
 * Use cases:
 *   - Conversion rates (k successes in N trials per period)
 *   - A/B testing with fixed sample sizes
 *   - Binary outcomes aggregated over N attempts
 * 
 * Special case: N=1 reduces to Bernoulli-Beta
 */

typedef struct {
    double alpha0;           // Beta prior parameter (shape, successes); must be > 0
    double beta0;            // Beta prior parameter (shape, failures); must be > 0
    int32_t N;              // Fixed number of trials per observation; must be >= 1
    double log_N_factorial; // Cached lgamma(N+1); set by bocpd_init, not by user
} BinomialBetaParams;

typedef struct {
    int32_t n;     // Number of observations (timesteps); >= 0
    double sum_k;  // Total successes across all timesteps; 0 <= sum_k <= n*N
} BinomialBetaStats;

/**
 * Get size of statistics structure.
 */
size_t binomial_beta_stats_size(void);

/**
 * Initialize statistics to prior (no observations).
 */
void binomial_beta_prior_stats(BinomialBetaStats* stats);

/**
 * Update sufficient statistics with new observation.
 * 
 * @param stats Statistics to update (modified in-place)
 * @param params Model parameters (unused, for API consistency)
 * @param k Number of successes (assumed valid: 0 <= k <= N)
 */
void binomial_beta_update_stats(
    BinomialBetaStats* stats,
    const BinomialBetaParams* params,
    double k
);

/**
 * Compute log predictive probability (Beta-Binomial).
 * 
 * Guards against invalid inputs (returns -INFINITY):
 *   - Non-finite or non-positive alpha0/beta0
 *   - Invalid N (< 1)
 *   - Non-finite, negative, or > N values of k
 *   - Non-integer k (tolerance 1e-9)
 *   - Invalid posterior parameters
 * 
 * @param params Model parameters
 * @param stats Current sufficient statistics
 * @param k Number of successes to predict (0 <= k <= N)
 * @return Log probability, or -INFINITY if invalid
 */
double binomial_beta_predictive_logpdf(
    const BinomialBetaParams* params,
    const BinomialBetaStats* stats,
    double k
);

/**
 * Copy statistics (for BOCPD internal use).
 */
void binomial_beta_copy_stats(void* dst, const void* src);

#endif /* BINOMIAL_BETA_H */