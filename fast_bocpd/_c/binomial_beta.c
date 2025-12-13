#include <math.h>
#include <stdint.h>
#include <string.h>

#include "binomial_beta.h"

#define BINOM_INT_TOL 1e-9

// =============================================================================
// == Validation Helpers =======================================================
// =============================================================================

static int validate_params(const BinomialBetaParams* params)
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
    if (params->N < 1) {
        return -1;
    }
    if (!isfinite(params->log_N_factorial)) {
        return -1;
    }
    return 0;
}


static int validate_stats(const BinomialBetaStats* stats,
                          const BinomialBetaParams* params)
{
    if (!stats) {
        return -1;
    }

    double max_sum_k = (double)stats->n * (double)params->N;
    if (!isfinite(stats->sum_k) || stats->sum_k < 0.0 || stats->sum_k > max_sum_k) {
        return -1;
    }
    return 0;
}


static int to_valid_successes(double k, int32_t N, int32_t* out_k)
{
    if (!isfinite(k) || k < 0.0) {
        return -1;
    }

    if (k > (double)N + 1.0) {
        return -1;
    }

    double rounded = nearbyint(k);
    if (fabs(k - rounded) > BINOM_INT_TOL) {
        return -1;
    }

    int64_t as_int = (int64_t)rounded;
    if (as_int < 0 || as_int > (int64_t)N) {
        return -1;
    }

    *out_k = (int32_t)as_int;
    return 0;
}

// =============================================================================
// == Public API ===============================================================
// =============================================================================

size_t binomial_beta_stats_size(void)
{
    return sizeof(BinomialBetaStats);
}


void binomial_beta_prior_stats(BinomialBetaStats* stats)
{
    stats->n = 0;
    stats->sum_k = 0.0;
}


void binomial_beta_update_stats(BinomialBetaStats* stats,
                                const BinomialBetaParams* params,
                                double k)
{
    (void)params;
    stats->n += 1;
    stats->sum_k += k;
}


double binomial_beta_predictive_logpdf(
    const BinomialBetaParams* params,
    const BinomialBetaStats* stats,
    double k
)
{
    if (validate_params(params) != 0) {
        return -INFINITY;
    }
    if (validate_stats(stats, params) != 0) {
        return -INFINITY;
    }

    int32_t k_int = 0;
    if (to_valid_successes(k, params->N, &k_int) != 0) {
        return -INFINITY;
    }
    
    double alpha_n = params->alpha0 + stats->sum_k;
    double beta_n = params->beta0 + ((double)stats->n * (double)params->N - stats->sum_k);
    
    if (!isfinite(alpha_n) || alpha_n <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(beta_n) || beta_n <= 0.0) {
        return -INFINITY;
    }
    
    // Compute Beta-Binomial log predictive
    // log p(k) = log(N choose k) + log B(alpha_n + k, beta_n + N - k) - log B(alpha_n, beta_n)
    
    // Binomial coefficient: log(N choose k) = lgamma(N+1) - lgamma(k+1) - lgamma(N-k+1)
    // Use cached log_N_factorial for efficiency
    double log_binom_coef = params->log_N_factorial
                          - lgamma((double)k_int + 1.0)
                          - lgamma((double)(params->N - k_int) + 1.0);
    
    // log B(a, b) = lgamma(a) + lgamma(b) - lgamma(a + b)
    double alpha_post = alpha_n + (double)k_int;
    double beta_post = beta_n + (double)(params->N - k_int);
    
    double log_beta_post = lgamma(alpha_post) + lgamma(beta_post) - lgamma(alpha_post + beta_post);
    double log_beta_prior = lgamma(alpha_n) + lgamma(beta_n) - lgamma(alpha_n + beta_n);
    
    double logpdf = log_binom_coef + log_beta_post - log_beta_prior;
    
    if (isnan(logpdf) || logpdf == INFINITY) {
        return -INFINITY;
    }
    
    return logpdf;
}


void binomial_beta_copy_stats(void* dst, const void* src)
{
    memcpy(dst, src, sizeof(BinomialBetaStats));
}
