#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "student_t_ng_grid.h"

#define GRID_STACK_K 64

// =============================================================================
// == Internal Helpers =========================================================
// =============================================================================

static double logsumexp_k(const double* log_arr, int32_t K)
{
    // Find maximum
    double max_val = -INFINITY;
    for (int32_t k = 0; k < K; k++) {
        if (log_arr[k] > max_val) {
            max_val = log_arr[k];
        }
    }
    
    if (max_val == -INFINITY) {
        return -INFINITY;
    }
    
    // Sum exp(log_arr[k] - max_val)
    double sum = 0.0;
    for (int32_t k = 0; k < K; k++) {
        sum += exp(log_arr[k] - max_val);
    }
    
    return max_val + log(sum);
}

static inline size_t align_up(size_t offset, size_t alignment)
{
    if (alignment == 0) {
        return offset;
    }
    return ((offset + alignment - 1) / alignment) * alignment;
}

// =============================================================================
// == Public API ===============================================================
// =============================================================================

/**
 * Calculate stats blob size for K components
 */
size_t student_t_ng_grid_stats_size(int32_t K)
{
    size_t size = 0;
    
    // int32_t K
    size += sizeof(int32_t);
    
    // Padding to align to sizeof(double)
    size = align_up(size, sizeof(double));
    
    // double log_pi[K]
    size += K * sizeof(double);
    
    // Padding to align to StudentTNGStats
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    size = align_up(size, _Alignof(StudentTNGStats));
#else
    size = align_up(size, 16);  // Conservative fallback
#endif
    
    // StudentTNGStats stats[K]
    size += K * sizeof(StudentTNGStats);
    
    return size;
}

/**
 * Initialize stats blob with prior
 */
void student_t_ng_grid_prior_stats(void* blob, const StudentTNGGridParams* params)
{
    if (!blob || !params) {
        return;
    }

    int32_t K = params->K;
    
    // Store K (for debugging/validation)
    *grid_blob_K_ptr(blob) = K;
    
    // Initialize log_pi with log of normalized prior
    // params->nu_prior is already normalized (done in bocpd_init)
    double* log_pi = grid_blob_log_pi(blob);
    
    for (int32_t k = 0; k < K; k++) {
        // Handle zero weights gracefully
        if (params->nu_prior[k] <= 0.0) {
            log_pi[k] = -INFINITY;
        } else {
            log_pi[k] = log(params->nu_prior[k]);
        }
    }
    
    // Initialize each component's stats with prior
    for (int32_t k = 0; k < K; k++) {
        StudentTNGStats* stats_k = grid_blob_comp_stats(blob, k, K);
        student_t_ng_prior_stats(stats_k);
    }
    
#ifdef BOCPD_DEBUG_CHECKS
    // Debug: verify blob K matches params K
    int32_t blob_K = *grid_blob_K_cptr(blob);
    if (blob_K != K) {
        // Critical error: blob corruption
        *grid_blob_K_ptr(blob) = K;  // Force correct value
    }
#endif
}

/**
 * Compute mixture predictive log-likelihood
 */
double student_t_ng_grid_predictive_logpdf(
    const void* blob,
    const StudentTNGGridParams* params,
    double x
)
{
    if (!blob || !params || !isfinite(x)) {
        return -INFINITY;
    }

    int32_t K = params->K;  // Use params->K as source of truth
    const double* log_pi = grid_blob_log_pi_const(blob);
    
    // Use stack buffer for small K, heap for large K
    double log_terms_stack[GRID_STACK_K];
    double* log_terms = (K <= GRID_STACK_K) ? log_terms_stack : (double*)malloc((size_t)K * sizeof(double));
    if (!log_terms) {
        return -INFINITY;  // Allocation failed
    }
    
    // For each component: log_term[k] = log_pi[k] + log p(x | stats_k, ν_k)
    for (int32_t k = 0; k < K; k++) {
        const StudentTNGStats* stats_k = grid_blob_comp_stats_const(blob, k, K);
        
        // Create StudentTNGParams for this component with ν_k
        StudentTNGParams params_k = {
            .mu0 = params->mu0,
            .kappa0 = params->kappa0,
            .alpha0 = params->alpha0,
            .beta0 = params->beta0,
            .nu = params->nu_grid[k]
        };
        
        double logp_k = student_t_ng_predictive_logpdf(&params_k, stats_k, x);
        log_terms[k] = log_pi[k] + logp_k;
    }
    
    // Mixture log-likelihood: logsumexp(log_terms)
    double result = logsumexp_k(log_terms, K);
    
    if (K > GRID_STACK_K) {
        free(log_terms);
    }
    return result;
}

/**
 * Update stats blob with new observation
 */
void student_t_ng_grid_update_stats(
    void* blob,
    const StudentTNGGridParams* params,
    double x
)
{
    if (!blob || !params || !isfinite(x)) {
        return;
    }

    int32_t K = params->K;  // Use params->K as source of truth
    double* log_pi = grid_blob_log_pi(blob);
    
    // Use stack buffer for small K, heap for large K
    double logp_k_stack[GRID_STACK_K];
    double* logp_k = (K <= GRID_STACK_K) ? logp_k_stack : (double*)malloc((size_t)K * sizeof(double));
    if (!logp_k) {
        return;  // Allocation failed, leave stats unchanged
    }
    
    // Step 1: Compute logp_k for each component (using PRE-x stats)
    for (int32_t k = 0; k < K; k++) {
        const StudentTNGStats* stats_k = grid_blob_comp_stats_const(blob, k, K);
        
        StudentTNGParams params_k = {
            .mu0 = params->mu0,
            .kappa0 = params->kappa0,
            .alpha0 = params->alpha0,
            .beta0 = params->beta0,
            .nu = params->nu_grid[k]
        };
        
        logp_k[k] = student_t_ng_predictive_logpdf(&params_k, stats_k, x);
    }
    
    // Step 2: Update log_pi using Bayes rule
    // log_pi_new[k] = log_pi_old[k] + logp_k[k]
    for (int32_t k = 0; k < K; k++) {
        log_pi[k] += logp_k[k];
    }
    
    // Step 3: Normalize log_pi
    double log_Z = logsumexp_k(log_pi, K);
    
    if (log_Z == -INFINITY) {
        // All components became -INFINITY (numerical collapse or zero priors)
        // Fallback: reset to prior to avoid NaN propagation
        for (int32_t k = 0; k < K; k++) {
            if (params->nu_prior[k] <= 0.0) {
                log_pi[k] = -INFINITY;
            } else {
                log_pi[k] = log(params->nu_prior[k]);
            }
        }
    } else {
        // Normal case: normalize
        for (int32_t k = 0; k < K; k++) {
            log_pi[k] -= log_Z;
        }
    }
    
    // Step 4: Update each component's sufficient statistics with x
    for (int32_t k = 0; k < K; k++) {
        StudentTNGStats* stats_k = grid_blob_comp_stats(blob, k, K);
        
        StudentTNGParams params_k = {
            .mu0 = params->mu0,
            .kappa0 = params->kappa0,
            .alpha0 = params->alpha0,
            .beta0 = params->beta0,
            .nu = params->nu_grid[k]
        };
        
        student_t_ng_update_stats(stats_k, &params_k, x);
    }
    
    if (K > GRID_STACK_K) {
        free(logp_k);
    }
}

/**
 * Deep copy stats blob
 */
void student_t_ng_grid_copy_stats(
    void* dst,
    const void* src,
    const StudentTNGGridParams* params
)
{
    if (!dst || !src || !params) {
        return;
    }
    // Simply copy the entire blob
    size_t blob_size = student_t_ng_grid_stats_size(params->K);
    memcpy(dst, src, blob_size);
}

/**
 * Validate grid parameters
 */
int student_t_ng_grid_validate_params(const StudentTNGGridParams* params)
{
    if (params->K < 1) {
        return -1;
    }
    
    if (!params->nu_grid || !params->nu_prior) {
        return -1;
    }
    
    double prior_sum = 0.0;
    
    for (int32_t k = 0; k < params->K; k++) {
        // Check ν_k is finite and > 0
        if (!isfinite(params->nu_grid[k]) || params->nu_grid[k] <= 0.0) {
            return -1;
        }
        
        // Check prior weight is finite and >= 0
        if (!isfinite(params->nu_prior[k]) || params->nu_prior[k] < 0.0) {
            return -1;
        }
        
        prior_sum += params->nu_prior[k];
    }
    
    // Check prior weights sum to something positive
    if (prior_sum <= 0.0) {
        return -1;
    }
    
    return 0;
}

/**
 * Normalize prior weights
 */
int student_t_ng_grid_normalize_prior(
    const double* nu_prior,
    int32_t K,
    double* normalized
)
{
    double sum = 0.0;
    
    for (int32_t k = 0; k < K; k++) {
        if (!isfinite(nu_prior[k]) || nu_prior[k] < 0.0) {
            return -1;
        }
        sum += nu_prior[k];
    }
    
    if (sum <= 0.0) {
        return -1;
    }
    
    for (int32_t k = 0; k < K; k++) {
        normalized[k] = nu_prior[k] / sum;
    }
    
    return 0;
}
