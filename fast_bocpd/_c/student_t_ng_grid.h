#ifndef STUDENT_T_NG_GRID_H
#define STUDENT_T_NG_GRID_H

#include <stdint.h>
#include <stddef.h>
#include "student_t_ng.h"

/**
 * ============================================================================
 * Student-t NG Grid Model - ν integrated out via discrete grid
 * ============================================================================
 * 
 * This model maintains a posterior distribution over ν (degrees of freedom)
 * for each run-length using a discrete grid of K values.
 * 
 * The predictive likelihood is a mixture:
 *   p(x | r) = Σ_k π_r[k] * p(x | stats_k, ν_k)
 * 
 * Where:
 *   - ν_grid[k] = k-th degrees of freedom value
 *   - π_r[k] = posterior probability of ν_k given data since last changepoint
 *   - stats_k = sufficient statistics for component k
 * 
 * MEMORY LAYOUT (byte blob):
 * ─────────────────────────────────────────────────────────────────────────
 * [0..3]:       int32_t K           (number of grid components)
 * [4..7]:       padding             (align to sizeof(double))
 * [8..8+K*8):   double log_pi[K]   (log posterior weights over ν)
 * [...]:        padding             (align to StudentTNGStats)
 * [...]:        StudentTNGStats[K]  (K component sufficient statistics)
 * ─────────────────────────────────────────────────────────────────────────
 */

/**
 * Grid model parameters
 * 
 * nu_grid and nu_prior are BORROWED pointers (Python-owned).
 * They must remain valid for the lifetime of the BOCPD object.
 */
typedef struct {
    double mu0;       // Prior mean location
    double kappa0;    // Prior precision scaling
    double alpha0;    // Prior shape (Gamma for precision)
    double beta0;     // Prior rate (Gamma for precision)
    
    int32_t K;                    // Number of ν grid values
    const double* nu_grid;        // [K] ν values (borrowed, Python-owned)
    const double* nu_prior;       // [K] prior weights (borrowed, Python-owned)
} StudentTNGGridParams;

/**
 * ============================================================================
 * Blob Layout Accessors
 * ============================================================================
 */

/**
 * Helper: Align offset to alignment boundary (same as in .c)
 */
static inline size_t grid_align_up(size_t offset, size_t alignment) {
    if (alignment == 0) {
        return offset;
    }
    return ((offset + alignment - 1) / alignment) * alignment;
}

/**
 * Calculate base offset where StudentTNGStats array begins
 * Must match the alignment logic in student_t_ng_grid_stats_size()
 */
static inline size_t grid_stats_base_offset(int32_t K) {
    size_t off = sizeof(int32_t);
    off = grid_align_up(off, sizeof(double));
    off += (size_t)K * sizeof(double);
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    off = grid_align_up(off, _Alignof(StudentTNGStats));
#else
    off = grid_align_up(off, 16);
#endif
    return off;
}

/**
 * Get pointer to K (number of components) from stats blob
 */
static inline int32_t* grid_blob_K_ptr(void* blob) {
    return (int32_t*)blob;
}

static inline const int32_t* grid_blob_K_cptr(const void* blob) {
    return (const int32_t*)blob;
}

/**
 * Get pointer to log_pi array from stats blob
 * Returns: pointer to K doubles (log posterior weights)
 */
static inline double* grid_blob_log_pi(void* blob) {
    return (double*)((uint8_t*)blob + grid_align_up(sizeof(int32_t), sizeof(double)));
}

static inline const double* grid_blob_log_pi_const(const void* blob) {
    return (const double*)((const uint8_t*)blob + grid_align_up(sizeof(int32_t), sizeof(double)));
}

/**
 * Get pointer to k-th component's StudentTNGStats
 * 
 * @param blob  Stats blob pointer
 * @param k     Component index (0 <= k < K)
 * @param K     Number of components
 * @return Pointer to StudentTNGStats for component k
 */
static inline StudentTNGStats* grid_blob_comp_stats(void* blob, int32_t k, int32_t K) {
    return (StudentTNGStats*)((uint8_t*)blob + grid_stats_base_offset(K) + k * sizeof(StudentTNGStats));
}

static inline const StudentTNGStats* grid_blob_comp_stats_const(const void* blob, int32_t k, int32_t K) {
    return (const StudentTNGStats*)((const uint8_t*)blob + grid_stats_base_offset(K) + k * sizeof(StudentTNGStats));
}

/**
 * ============================================================================
 * Core Functions (vtable interface)
 * ============================================================================
 */

/**
 * Calculate stats blob size for K components
 * 
 * Layout:
 *   - 4 bytes: K
 *   - 4 bytes: padding
 *   - K*8 bytes: log_pi
 *   - padding to align StudentTNGStats
 *   - K * sizeof(StudentTNGStats)
 * 
 * @param K  Number of grid components
 * @return Total byte size (unaligned; caller must round up to STATS_ALIGNMENT)
 */
size_t student_t_ng_grid_stats_size(int32_t K);

/**
 * Initialize stats blob with prior (fresh regime)
 * 
 * For each component k:
 *   - Set stats_k = prior StudentTNGStats
 *   - Set log_pi[k] = log(nu_prior[k])
 * 
 * @param blob    Stats blob to initialize
 * @param params  Grid parameters (must be StudentTNGGridParams*)
 */
void student_t_ng_grid_prior_stats(void* blob, const StudentTNGGridParams* params);

/**
 * Compute mixture predictive log-likelihood
 * 
 * Returns: log[ Σ_k exp(log_pi[k]) * p(x | stats_k, ν_k) ]
 * 
 * Uses logsumexp for numerical stability.
 * 
 * @param blob    Stats blob (pre-x, contains old stats)
 * @param params  Grid parameters
 * @param x       Observation
 * @return Mixture predictive log-likelihood
 */
double student_t_ng_grid_predictive_logpdf(
    const void* blob,
    const StudentTNGGridParams* params,
    double x
);

/**
 * Update stats blob with new observation x
 * 
 * Steps:
 *   1. For each k: compute logp_k = p(x | stats_k, ν_k)
 *   2. Update log_pi: log_pi_new[k] = log_pi_old[k] + logp_k
 *   3. Normalize: log_pi_new -= logsumexp(log_pi_new)
 *   4. For each k: update stats_k with x using ν_k
 * 
 * @param blob    Stats blob to update (modified in-place)
 * @param params  Grid parameters
 * @param x       New observation
 */
void student_t_ng_grid_update_stats(
    void* blob,
    const StudentTNGGridParams* params,
    double x
);

/**
 * Deep copy stats blob from src to dst
 * 
 * Copies entire blob: K, log_pi[K], and stats[K]
 * 
 * @param dst     Destination blob
 * @param src     Source blob
 * @param params  Grid parameters (used to determine K)
 */
void student_t_ng_grid_copy_stats(
    void* dst,
    const void* src,
    const StudentTNGGridParams* params
);

/**
 * ============================================================================
 * Helper Functions
 * ============================================================================
 */

/**
 * Validate grid parameters
 * 
 * Checks:
 *   - K >= 1
 *   - All ν_k > 0 and finite
 *   - All prior weights >= 0 and finite
 *   - Prior weights sum > 0
 * 
 * @param params  Parameters to validate
 * @return 0 if valid, -1 if invalid
 */
int student_t_ng_grid_validate_params(const StudentTNGGridParams* params);

/**
 * Normalize prior weights and store in output buffer
 * 
 * @param nu_prior     Input prior weights (K values)
 * @param K            Number of components
 * @param normalized   Output buffer (K doubles, caller-allocated)
 * @return 0 on success, -1 if sum is zero or invalid
 */
int student_t_ng_grid_normalize_prior(
    const double* nu_prior,
    int32_t K,
    double* normalized
);

#endif // STUDENT_T_NG_GRID_H
