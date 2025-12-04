#ifndef BOCPD_CORE_H
#define BOCPD_CORE_H

#include <stdint.h>
#include "gaussian_nig.h"
#include "student_t_ng.h"
#include "hazard.h"

/**
 * Observation model types
 */
typedef enum {
    OBS_MODEL_GAUSSIAN_NIG,
    OBS_MODEL_STUDENT_T_NG,
    // Future: OBS_MODEL_POISSON_GAMMA, OBS_MODEL_BERNOULLI_BETA, etc.
} ObsModelType;

/**
 * Hazard function types
 */
typedef enum {
    HAZARD_CONSTANT,
    // Future: HAZARD_GEOMETRIC, etc.
} HazardType;

/**
 * Union for observation model parameters
 */
typedef union {
    GaussianNIGParams gaussian_nig;
    StudentTNGParams student_t_ng;
    // Future models will be added here
} ObsModelParams;

/**
 * Union for observation model statistics
 */
typedef union {
    GaussianNIGStats gaussian_nig;
    StudentTNGStats student_t_ng;
    // Future model stats will be added here
} ObsModelStats;

/**
 * Union for hazard parameters
 */
typedef union {
    ConstantHazardParams constant;
    // Future hazard types will be added here
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
    
    // State arrays (size: max_run_length + 1)
    double* log_joint;              // log P(r_t = r, x_1:t)
    ObsModelStats* stats;           // Sufficient statistics for each run length
    
    // Working arrays for update
    double* new_log_joint;
    ObsModelStats* new_stats;
    double* posterior_r;            // Output buffer for posterior distribution
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
