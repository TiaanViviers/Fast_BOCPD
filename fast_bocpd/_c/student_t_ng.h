#ifndef STUDENT_T_NG_H
#define STUDENT_T_NG_H

#include <stdint.h>

/**
 * Fixed-ν Student-t observation model using Gaussian scale-mixture.
 * 
 * Model:
 *   x_i | μ, σ², λ_i ~ N(μ, σ²/λ_i)
 *   λ_i ~ Gamma(ν/2, ν/2)
 *   => x_i ~ StudentT(ν, μ, σ²)
 * 
 * Key property: ν explicitly controls tail heaviness
 *   - ν → ∞: Gaussian (no heavy tails)
 *   - ν = 4-5: Financial data (moderate heavy tails)
 *   - ν = 3: Heavier tails (recommended default)
 *   - ν = 1: Cauchy (very heavy tails)
 * 
 * Inference: Online EM/VB with weighted sufficient statistics
 *   - Maintains weighted counts: S0, S1, S2
 *   - Weight w = (ν+1)/(ν+δ²) down-weights outliers
 *   - Updates location/scale using weighted stats
 * 
 * NOTE: This is an approximation (reweighted Normal-Gamma), not exact
 * conjugate Student-t Bayes. Outlier down-weighting is heuristic and
 * depends on how fast weighted stats adapt.
 * 
 * Prior: Normal-Gamma on (μ, precision)
 *   μ | τ ~ N(μ₀, (κ₀τ)⁻¹)
 *   τ ~ Gamma(α₀, β₀)
 */
typedef struct {
    double mu0;      // Prior mean
    double kappa0;   // Prior precision scaling (must be > 0)
    double alpha0;   // Prior shape parameter (must be > 0)
    double beta0;    // Prior rate parameter (must be > 0)
    double nu;       // Degrees of freedom (FIXED, must be > 0)
} StudentTNGParams;

/**
 * Weighted sufficient statistics for robust updates
 */
typedef struct {
    double S0;       // Weighted count: Σ w_i
    double S1;       // Weighted sum: Σ w_i x_i
    double S2;       // Weighted sum of squares: Σ w_i x_i²
} StudentTNGStats;

void student_t_ng_prior_stats(StudentTNGStats* stats);
void student_t_ng_update_stats(StudentTNGStats* stats, const StudentTNGParams* params, double x);
double student_t_ng_predictive_logpdf(const StudentTNGParams* params, const StudentTNGStats* stats, double x);

#endif // STUDENT_T_NG_H
