#include <math.h>

#include "student_t_ng.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define S0_EPS 1e-10

// =============================================================================
// == Validation Helpers =======================================================
// =============================================================================

static int validate_params(const StudentTNGParams* params)
{
    if (params->kappa0 <= 0.0) return -1;
    if (params->alpha0 <= 0.0) return -1;
    if (params->beta0 <= 0.0) return -1;
    if (params->nu <= 0.0) return -1;
    if (!isfinite(params->kappa0)) return -1;
    if (!isfinite(params->alpha0)) return -1;
    if (!isfinite(params->beta0)) return -1;
    if (!isfinite(params->nu)) return -1;
    return 0;
}

static int validate_stats(const StudentTNGStats* stats)
{
    if (!stats) {
        return -1;
    }
    if (!isfinite(stats->S0) || !isfinite(stats->S1) || !isfinite(stats->S2)) {
        return -1;
    }
    if (stats->S0 < 0.0) {
        return -1;
    }
    return 0;
}

// =============================================================================
// == Public API ===============================================================
// =============================================================================

void student_t_ng_prior_stats(StudentTNGStats* stats)
{
    stats->S0 = 0.0;
    stats->S1 = 0.0;
    stats->S2 = 0.0;
}

static void compute_posterior_hyperparams(
    const StudentTNGParams* params,
    const StudentTNGStats* stats,
    double* mu_n,
    double* kappa_n,
    double* alpha_n,
    double* beta_n)
{
    if (stats->S0 < S0_EPS) {
        // No data yet: posterior = prior
        *mu_n = params->mu0;
        *kappa_n = params->kappa0;
        *alpha_n = params->alpha0;
        *beta_n = params->beta0;
        return;
    }

    // Weighted mean: μ̂ = S1/S0
    double mu_hat = stats->S1 / stats->S0;
    
    // Weighted sum of squared deviations
    double S = stats->S2 - (stats->S1 * stats->S1) / stats->S0;
    if (S < 0.0) S = 0.0;

    // Normal-Gamma conjugate update with weighted stats
    *kappa_n = params->kappa0 + stats->S0;
    *mu_n = (params->kappa0 * params->mu0 + stats->S1) / (*kappa_n);
    *alpha_n = params->alpha0 + 0.5 * stats->S0;

    double diff = mu_hat - params->mu0;
    *beta_n = params->beta0 + 0.5 * (
        S + (params->kappa0 * stats->S0 / (*kappa_n)) * diff * diff
    );
    
    // Ensure beta_n is positive
    if (*beta_n <= 0.0) *beta_n = params->beta0;
}

void student_t_ng_update_stats(
    StudentTNGStats* stats,
    const StudentTNGParams* params,
    double x
) 
{
    if (!stats) {
        return;
    }
    // Validate parameters
    if (validate_params(params) != 0) {
        return;  // Invalid params dont update
    }
    
    double mu_n, kappa_n, alpha_n, beta_n;
    compute_posterior_hyperparams(params, stats, &mu_n, &kappa_n, &alpha_n, &beta_n);
    
    double sigma2 = (alpha_n > 0.0) ? (beta_n / alpha_n) : 1.0;
    double residual = x - mu_n;
    double delta2 = (residual * residual) / (sigma2 + 1e-10);
    
    double nu = params->nu;
    double weight = (nu + 1.0) / (nu + delta2);
    
    // Ensure weight is finite and positive (prevents NaN propagation)
    if (!isfinite(weight) || weight <= 0.0) {
        weight = 0.0;  // Skip this observation if weight is invalid
    }
    
    stats->S0 += weight;
    stats->S1 += weight * x;
    stats->S2 += weight * x * x;
}

double student_t_ng_predictive_logpdf(
    const StudentTNGParams* params,
    const StudentTNGStats* stats,
    double x
) 
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
    
    // Validate parameters - return -inf for invalid params instead of silently fixing
    double mu_n, kappa_n, alpha_n, beta_n;
    compute_posterior_hyperparams(params, stats, &mu_n, &kappa_n, &alpha_n, &beta_n);

    // Use fixed nu for degrees of freedom
    double nu_pred = params->nu;
    
    // Validate computed hyperparameters - return -inf instead of clamping
    if (nu_pred < 1.0 || alpha_n <= 0.0 || beta_n <= 0.0 || kappa_n <= 0.0) {
        return -INFINITY;
    }
    
    // Scale formula for Student-t with Normal-Gamma prior
    // scale² = (β/α) * (κ+1)/κ
    double scale2 = (beta_n / alpha_n) * ((kappa_n + 1.0) / kappa_n);
    
    // Validate scale - return -inf for invalid scale
    if (!isfinite(scale2) || scale2 <= 0.0) {
        return -INFINITY;
    }
    
    double scale = sqrt(scale2);
    double z = (x - mu_n) / scale;

    // Student-t log PDF with fixed nu
    double log_norm = lgamma((nu_pred + 1.0) / 2.0)
                    - lgamma(nu_pred / 2.0)
                    - 0.5 * log(nu_pred * M_PI)
                    - log(scale);
    
    // Use log1p for numerical stability with large z
    double log_kernel = -(nu_pred + 1.0) / 2.0 * log1p((z * z) / nu_pred);

    return log_norm + log_kernel;
}
