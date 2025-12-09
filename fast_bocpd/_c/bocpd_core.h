#ifndef BOCPD_CORE_H
#define BOCPD_CORE_H

#include <stdint.h>
#include <stddef.h>
#include "gaussian_nig.h"
#include "student_t_ng.h"
#include "student_t_ng_grid.h"
#include "poisson_gamma.h"
#include "bernoulli_beta.h"
#include "binomial_beta.h"
#include "gamma_gamma_fixed_shape.h"
#include "hazard.h"

/**
 * Observation model virtual table interface.
 * 
 * All observation models must implement these functions to integrate with BOCPD.
 */
typedef struct {
    size_t (*stats_size)(const void* params);
    void (*prior_stats)(void* stats, const void* params);
    void (*update_stats)(void* stats, const void* params, double x);
    double (*predictive_logpdf)(const void* stats, const void* params, double x);
    void (*copy_stats)(void* dst, const void* src, const void* params);
} ObsModelVTable;

/**
 * Helper: Round up size to alignment boundary (safe for any alignment >= 1)
 */
static inline size_t round_up_align(size_t n, size_t align) {
    if (align == 0) return n;
    size_t rem = n % align;
    return rem ? (n + (align - rem)) : n;
}

/**
 * Helper: Get mutable pointer to stats blob for run-length r
 * Use when writing/modifying stats (e.g., update_stats, prior_stats)
 */
static inline void* stats_at(uint8_t* base, size_t r, size_t stride) {
    return (void*)(base + r * stride);
}

/**
 * Helper: Get const pointer to stats blob for run-length r
 * Use when reading stats without modification (e.g., predictive_logpdf)
 * Maintains const-correctness and prevents accidental modification
 */
static inline const void* cstats_at(const uint8_t* base, size_t r, size_t stride) {
    return (const void*)(base + r * stride);
}

/**
 * Observation model types
 */
typedef enum {
    OBS_MODEL_GAUSSIAN_NIG = 0,
    OBS_MODEL_STUDENT_T_NG = 1,
    OBS_MODEL_STUDENT_T_NG_GRID = 2,
    OBS_MODEL_POISSON_GAMMA = 3,
    OBS_MODEL_BERNOULLI_BETA = 4,
    OBS_MODEL_BINOMIAL_BETA = 5,
    OBS_MODEL_GAMMA_GAMMA = 6
} ObsModelType;

/**
 * Hazard function types
 */
typedef enum {
    HAZARD_CONSTANT,
} HazardType;

/**
 * Union for observation model parameters
 */
typedef union {
    GaussianNIGParams gaussian_nig;
    StudentTNGParams student_t_ng;
    StudentTNGGridParams student_t_ng_grid;
    PoissonGammaParams poisson_gamma;
    BernoulliBetaParams bernoulli_beta;
    BinomialBetaParams binomial_beta;
    GammaGammaParams gamma_gamma;
} ObsModelParams;

/**
 * Union for hazard parameters
 */
typedef union {
    ConstantHazardParams constant;
} HazardParams;

/**
 * BOCPD state for online processing
 */
typedef struct {
    int32_t max_run_length;
    
    // Model type identifiers
    ObsModelType obs_model_type;
    HazardType hazard_type;
    
    // Model parameters
    ObsModelParams obs_params;
    HazardParams hazard_params;
    
    // Observation model vtable
    ObsModelVTable obs_vtable;
    size_t stats_size;  // Size of one stats blob in bytes
    
    // Pointer to active params member (for proper aliasing)
    const void* obs_params_ptr;
    
    // State arrays (using byte buffers for variable-size stats)
    double* log_joint;              // log P(r_t = r, x_1:t) [size: max_run_length + 1]
    uint8_t* stats;                 // Sufficient statistics [size: (max_run_length + 1) * stats_size]
    
    // Working arrays for update
    double* new_log_joint;          // [size: max_run_length + 1]
    uint8_t* new_stats;             // [size: (max_run_length + 1) * stats_size]
    double* posterior_r;            // Output buffer [size: max_run_length + 1]
    
    // Grid Student-t ownership (NULL for non-grid models)
    double* owned_nu_grid;          // Deep copy of nu_grid
    double* owned_nu_prior;         // Normalized copy of nu_prior
} BOCPDState;

/**
 * Initialize BOCPD state
 * 
 * @param state             BOCPD state structure (already allocated)
 * @param obs_model_type    Type of observation model
 * @param obs_params        Observation model parameters (cast to appropriate type)
 * @param hazard_type       Type of hazard function
 * @param hazard_params     Hazard function parameters (cast to appropriate type)
 * @param max_run_length    Maximum run length to track
 * @return                  0 on success, -1 on error
 */
int bocpd_init(
    BOCPDState* state,
    ObsModelType obs_model_type,
    const void* obs_params,
    HazardType hazard_type,
    const void* hazard_params,
    int32_t max_run_length
);

/**
 * Free BOCPD state memory
 * 
 * @param state    BOCPD state structure
 */
void bocpd_free(BOCPDState* state);

/**
 * Reset BOCPD to prior (as if no data has been seen)
 * 
 * @param state    BOCPD state structure
 */
void bocpd_reset(BOCPDState* state);

/**
 * Process one new observation
 * 
 * @param state         BOCPD state structure
 * @param x             New observation
 * @param cp_prob_out   Output: probability of changepoint (can be NULL)
 * @return              Pointer to posterior_r array, or NULL on error
 */
double* bocpd_update(BOCPDState* state, double x, double* cp_prob_out);

/**
 * Batch processing: update with multiple observations
 * 
 * @param state         BOCPD state structure
 * @param x_array       Array of observations
 * @param n_obs         Number of observations
 * @param cp_probs_out  Output array for changepoint probabilities (size n_obs)
 * @return              0 on success, -1 on error
 */
int bocpd_batch_update(
    BOCPDState* state,
    const double* x_array,
    int32_t n_obs,
    double* cp_probs_out
);

/**
 * Get maximum a posteriori (MAP) run length at current time
 * 
 * @param state    BOCPD state structure
 * @return         Most likely run length, or -1 on error
 */
int32_t bocpd_get_map_run_length(const BOCPDState* state);

/**
 * Get current posterior distribution over run lengths
 * 
 * @param state         BOCPD state structure
 * @param posterior_out Output array (size max_run_length + 1)
 * @return              0 on success, -1 on error
 */
int bocpd_get_posterior(const BOCPDState* state, double* posterior_out);

#endif // BOCPD_CORE_H
