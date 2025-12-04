#include "bocpd_core.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Numerically stable log-sum-exp for a pair of values
 */
static double logsumexp_pair(double a, double b)
{
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    
    double m = (a > b) ? a : b;
    return m + log(exp(a - m) + exp(b - m));
}

/**
 * Numerically stable log-sum-exp for an array
 */
static double logsumexp_array(const double* arr, int32_t n)
{
    // Find maximum
    double m = -INFINITY;
    for (int32_t i = 0; i < n; i++) {
        if (arr[i] > m) {
            m = arr[i];
        }
    }
    
    if (m == -INFINITY) {
        return -INFINITY;
    }
    
    // Sum exp(arr[i] - m)
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) {
        sum += exp(arr[i] - m);
    }
    
    return m + log(sum);
}

int bocpd_init(BOCPDState* state, ObsModelType obs_model_type, 
    const void* obs_params, HazardType hazard_type, const void* hazard_params,
    int32_t max_run_length) 
{
    if (max_run_length <= 0) {
        return -1;
    }
    
    state->max_run_length = max_run_length;
    state->obs_model_type = obs_model_type;
    state->hazard_type = hazard_type;
    
    // Copy model parameters based on type
    switch (obs_model_type) {
        case OBS_MODEL_GAUSSIAN_NIG:
            state->obs_params.gaussian_nig = *(const GaussianNIGParams*)obs_params;
            break;
        case OBS_MODEL_STUDENT_T_NG:
            state->obs_params.student_t_ng = *(const StudentTNGParams*)obs_params;
            break;
        default:
            return -1;  // Unknown model type
    }
    
    // Initialize hazard based on type
    switch (hazard_type) {
        case HAZARD_CONSTANT: {
            const ConstantHazardParams* chp = (const ConstantHazardParams*)hazard_params;
            state->hazard_params.constant = *chp;
            break;
        }
        default:
            return -1;  // Unknown hazard type
    }
    
    // Allocate arrays (size: max_run_length + 1)
    int32_t size = max_run_length + 1;
    
    state->log_joint = (double*)malloc(size * sizeof(double));
    state->new_log_joint = (double*)malloc(size * sizeof(double));
    state->posterior_r = (double*)malloc(size * sizeof(double));
    state->stats = (ObsModelStats*)malloc(size * sizeof(ObsModelStats));
    state->new_stats = (ObsModelStats*)malloc(size * sizeof(ObsModelStats));
    
    if (!state->log_joint || !state->new_log_joint || !state->posterior_r ||
        !state->stats || !state->new_stats) {
        bocpd_free(state);
        return -1;
    }
    
    // Initialize to prior
    bocpd_reset(state);
    
    return 0;
}

void bocpd_free(BOCPDState* state) 
{
    if (state->log_joint) free(state->log_joint);
    if (state->new_log_joint) free(state->new_log_joint);
    if (state->posterior_r) free(state->posterior_r);
    if (state->stats) free(state->stats);
    if (state->new_stats) free(state->new_stats);
    
    state->log_joint = NULL;
    state->new_log_joint = NULL;
    state->posterior_r = NULL;
    state->stats = NULL;
    state->new_stats = NULL;
}

void bocpd_reset(BOCPDState* state) 
{
    int32_t size = state->max_run_length + 1;
    
    // Initialize log_joint to -inf
    for (int32_t i = 0; i < size; i++) {
        state->log_joint[i] = -INFINITY;
    }
    state->log_joint[0] = 0.0;  // log P(r_0=0, no data) = 0
    
    // Initialize stats[0] to prior based on model type
    switch (state->obs_model_type) {
        case OBS_MODEL_GAUSSIAN_NIG:
            gaussian_nig_prior_stats(&state->stats[0].gaussian_nig);
            break;
        case OBS_MODEL_STUDENT_T_NG:
            student_t_ng_prior_stats(&state->stats[0].student_t_ng);
            break;
        default:
            // Unknown model - shouldn't happen if bocpd_init succeeded
            break;
    }
}

double* bocpd_update(BOCPDState* state, double x, double* cp_prob_out) 
{
    int32_t R = state->max_run_length;
    
    // Initialize new arrays
    for (int32_t i = 0; i <= R; i++) {
        state->new_log_joint[i] = -INFINITY;
    }
    
    // Prior stats for changepoint branch (model-specific)
    ObsModelStats prior_stats;
    switch (state->obs_model_type) {
        case OBS_MODEL_GAUSSIAN_NIG:
            gaussian_nig_prior_stats(&prior_stats.gaussian_nig);
            break;
        case OBS_MODEL_STUDENT_T_NG:
            student_t_ng_prior_stats(&prior_stats.student_t_ng);
            break;
        default:
            return NULL;  // Unknown model type
    }
    state->new_stats[0] = prior_stats;
    
    // Loop over all previous run lengths
    for (int32_t r_prev = 0; r_prev <= R; r_prev++) {
        double lj_prev = state->log_joint[r_prev];
        
        if (lj_prev == -INFINITY) {
            continue;
        }
        
        ObsModelStats* stats_prev = &state->stats[r_prev];
        
        // Predictive log likelihood (model-specific)
        double log_pred, log_pred_cp;
        switch (state->obs_model_type) {
            case OBS_MODEL_GAUSSIAN_NIG:
                log_pred = gaussian_nig_predictive_logpdf(
                    &state->obs_params.gaussian_nig, &stats_prev->gaussian_nig, x
                );
                log_pred_cp = gaussian_nig_predictive_logpdf(
                    &state->obs_params.gaussian_nig, &prior_stats.gaussian_nig, x
                );
                break;
            case OBS_MODEL_STUDENT_T_NG:
                log_pred = student_t_ng_predictive_logpdf(
                    &state->obs_params.student_t_ng, &stats_prev->student_t_ng, x
                );
                log_pred_cp = student_t_ng_predictive_logpdf(
                    &state->obs_params.student_t_ng, &prior_stats.student_t_ng, x
                );
                break;
            default:
                return NULL;  // Unknown model type
        }
        
        // Hazard transitions (hazard-specific)
        double log_trans_cp, log_trans_cont;
        switch (state->hazard_type) {
            case HAZARD_CONSTANT:
                log_trans_cp = constant_hazard_log_transition_cp(&state->hazard_params.constant, r_prev);
                log_trans_cont = constant_hazard_log_transition_cont(&state->hazard_params.constant, r_prev);
                break;
            default:
                return NULL;  // Unknown hazard type
        }
        
        // Changepoint branch: r_t = 0
        double logp_cp = lj_prev + log_pred_cp + log_trans_cp;
        state->new_log_joint[0] = logsumexp_pair(state->new_log_joint[0], logp_cp);
        
        // Continuation branch: r_t = r_prev + 1
        int32_t r_cont = r_prev + 1;
        if (r_cont <= R) {
            double logp_cont = lj_prev + log_pred + log_trans_cont;
            state->new_log_joint[r_cont] = logsumexp_pair(state->new_log_joint[r_cont], logp_cont);
            
            // Update stats for continuation (model-specific)
            state->new_stats[r_cont] = *stats_prev;
            switch (state->obs_model_type) {
                case OBS_MODEL_GAUSSIAN_NIG:
                    gaussian_nig_update_stats(&state->new_stats[r_cont].gaussian_nig, x);
                    break;
                case OBS_MODEL_STUDENT_T_NG:
                    student_t_ng_update_stats(
                        &state->new_stats[r_cont].student_t_ng,
                        &state->obs_params.student_t_ng,
                        x
                    );
                    break;
                default:
                    return NULL;  // Unknown model type
            }
        }
    }
    
    // Normalize to get posterior over run length
    double log_Z = logsumexp_array(state->new_log_joint, R + 1);
    
    if (log_Z == -INFINITY) {
        // All probabilities are zero - something went wrong
        for (int32_t i = 0; i <= R; i++) {
            state->posterior_r[i] = 0.0;
        }
    } else {
        for (int32_t i = 0; i <= R; i++) {
            state->posterior_r[i] = exp(state->new_log_joint[i] - log_Z);
        }
    }
    
    // Update internal state
    double* tmp_log = state->log_joint;
    state->log_joint = state->new_log_joint;
    state->new_log_joint = tmp_log;
    
    ObsModelStats* tmp_stats = state->stats;
    state->stats = state->new_stats;
    state->new_stats = tmp_stats;
    
    // Output changepoint probability if requested
    if (cp_prob_out) {
        *cp_prob_out = state->posterior_r[0];
    }
    
    return state->posterior_r;
}

int bocpd_batch_update(
    BOCPDState* state,
    const double* x_array,
    int32_t n_obs,
    double* cp_probs_out
) {
    for (int32_t i = 0; i < n_obs; i++) {
        double cp_prob;
        if (bocpd_update(state, x_array[i], &cp_prob) == NULL) {
            return -1;
        }
        if (cp_probs_out) {
            cp_probs_out[i] = cp_prob;
        }
    }
    return 0;
}

int32_t bocpd_get_map_run_length(const BOCPDState* state) {
    if (!state || !state->posterior_r) {
        return -1;
    }
    
    int32_t max_r = 0;
    double max_prob = state->posterior_r[0];
    
    for (int32_t r = 1; r <= state->max_run_length; r++) {
        if (state->posterior_r[r] > max_prob) {
            max_prob = state->posterior_r[r];
            max_r = r;
        }
    }
    
    return max_r;
}

int bocpd_get_posterior(const BOCPDState* state, double* posterior_out) {
    if (!state || !state->posterior_r || !posterior_out) {
        return -1;
    }
    
    for (int32_t r = 0; r <= state->max_run_length; r++) {
        posterior_out[r] = state->posterior_r[r];
    }
    
    return 0;
}
