#include "binomial_beta.h"
#include <stdint.h>
#include <math.h>
#include <string.h>

size_t binomial_beta_stats_size(void) {
    return sizeof(BinomialBetaStats);
}

void binomial_beta_prior_stats(BinomialBetaStats* stats) {
    stats->n = 0;
    stats->sum_k = 0.0;
}

void binomial_beta_update_stats(
    BinomialBetaStats* stats,
    const BinomialBetaParams* params,
    double k
) {
    (void)params;  // Unused, for API consistency
    stats->n += 1;
    stats->sum_k += k;
}

double binomial_beta_predictive_logpdf(
    const BinomialBetaParams* params,
    const BinomialBetaStats* stats,
    double k
) {
    // NULL pointer guards
    if (!params || !stats) {
        return -INFINITY;
    }
    
    // 1. Validate parameters (fail fast)
    if (!isfinite(params->alpha0) || params->alpha0 <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(params->beta0) || params->beta0 <= 0.0) {
        return -INFINITY;
    }
    if (params->N < 1) {
        return -INFINITY;
    }
    
    // Sanity check cached factorial (should be set by bocpd_init)
    if (!isfinite(params->log_N_factorial)) {
        return -INFINITY;
    }
    
    // 2. Validate observation k
    if (!isfinite(k) || k < 0.0) {
        return -INFINITY;
    }
    
    // Protect against overflow: reject k that's way beyond N
    if (k > (double)params->N + 1.0) {
        return -INFINITY;
    }
    
    // 3. Check integer tolerance (use int64_t to avoid overflow on cast)
    int64_t k_int64 = (int64_t)nearbyint(k);
    if (fabs(k - (double)k_int64) > 1e-9) {
        return -INFINITY;  // Not an integer
    }
    
    // 4. Range check: k must be <= N
    if (k_int64 > (int64_t)params->N) {
        return -INFINITY;
    }
    
    // Safe to cast to int32_t now (we know k_int64 <= N and N is int32_t)
    int32_t k_int = (int32_t)k_int64;
    
    // 5. Validate stats early (before using them in computations)
    double max_sum_k = (double)stats->n * (double)params->N;
    if (!isfinite(stats->sum_k) || stats->sum_k < 0.0 || stats->sum_k > max_sum_k) {
        return -INFINITY;
    }
    
    // 6. Compute posterior parameters
    double alpha_n = params->alpha0 + stats->sum_k;
    double beta_n = params->beta0 + ((double)stats->n * (double)params->N - stats->sum_k);
    
    // Validate posterior parameters
    if (!isfinite(alpha_n) || alpha_n <= 0.0) {
        return -INFINITY;
    }
    if (!isfinite(beta_n) || beta_n <= 0.0) {
        return -INFINITY;
    }
    
    // 7. Compute Beta-Binomial log predictive
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
    
    // Final sanity check: prevent NaN or +infinity (allow -infinity as valid prob ~ 0)
    if (isnan(logpdf) || logpdf == INFINITY) {
        return -INFINITY;
    }
    
    return logpdf;
}

void binomial_beta_copy_stats(void* dst, const void* src) {
    memcpy(dst, src, sizeof(BinomialBetaStats));
}