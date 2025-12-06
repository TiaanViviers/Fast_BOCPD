#ifndef BERNOULLI_BETA_H
#define BERNOULLI_BETA_H

#include <stddef.h>
#include <stdint.h>

/**
 * Bernoulli-Beta conjugate model for binary data (0/1).
 * 
 * Data: x ∈ {0, 1}
 * Likelihood: x | p ~ Bernoulli(p)
 * Prior: p ~ Beta(α₀, β₀)
 * Posterior: p | data ~ Beta(α₀ + s, β₀ + (n - s))
 * Predictive: P(x=1 | data) = αₙ / (αₙ + βₙ)
 * 
 * Sufficient statistics: (n, s) where s = sum of x's (successes)
 */

/**
 * Model parameters (prior hyperparameters)
 */
typedef struct {
    double alpha0;  // Prior successes (must be > 0)
    double beta0;   // Prior failures (must be > 0)
} BernoulliBetaParams;

/**
 * Sufficient statistics for Bernoulli-Beta
 */
typedef struct {
    int32_t n;      // Number of observations
    double sum_x;   // Number of successes (sum of x's)
} BernoulliBetaStats;

/**
 * Get size of stats structure
 */
size_t bernoulli_beta_stats_size(void);

/**
 * Initialize stats to prior (no observations)
 */
void bernoulli_beta_prior_stats(BernoulliBetaStats* stats);

/**
 * Update sufficient statistics with new observation
 * 
 * @param stats     Current statistics
 * @param params    Model parameters (unused but kept for API consistency)
 * @param x         New observation (should be 0 or 1, validated in predictive)
 */
void bernoulli_beta_update_stats(
    BernoulliBetaStats* stats,
    const BernoulliBetaParams* params,
    double x
);

/**
 * Compute predictive log probability P(x | data)
 * 
 * Predictive distribution:
 *   P(x=1 | data) = αₙ / (αₙ + βₙ)
 *   P(x=0 | data) = βₙ / (αₙ + βₙ)
 * 
 * where αₙ = α₀ + s, βₙ = β₀ + (n - s)
 * 
 * @param params    Model parameters
 * @param stats     Current statistics
 * @param x         Observation to evaluate
 * @return          Log probability, or -INFINITY if invalid
 */
double bernoulli_beta_predictive_logpdf(
    const BernoulliBetaParams* params,
    const BernoulliBetaStats* stats,
    double x
);

/**
 * Copy statistics (for BOCPD continuation branch)
 */
void bernoulli_beta_copy_stats(
    BernoulliBetaStats* dst,
    const BernoulliBetaStats* src
);

#endif // BERNOULLI_BETA_H