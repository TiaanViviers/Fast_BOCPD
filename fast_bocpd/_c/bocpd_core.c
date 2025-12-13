#include "bocpd_core.h"
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #include <stdalign.h>
#endif

/*
 * Alignment for stats buffers (use max_align_t when available).
 * C11 has _Alignof and max_align_t, C99 fallback uses 16-byte alignment.
 */
#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define STATS_ALIGNMENT _Alignof(max_align_t)
#else
    #define STATS_ALIGNMENT 16  // Practical fallback: works on most platforms
#endif

// =============================================================================
// == Observation Model VTable Implementations ================================
// =============================================================================

/* Gaussian NIG vtable functions */
static size_t gaussian_nig_stats_size_fn(const void* params) {
    (void)params;  // unused
    return sizeof(GaussianNIGStats);
}

static void gaussian_nig_prior_stats_fn(void* stats, const void* params) {
    (void)params;  // unused
    gaussian_nig_prior_stats((GaussianNIGStats*)stats);
}

static void gaussian_nig_update_stats_fn(void* stats, const void* params, double x) {
    (void)params;  // unused
    gaussian_nig_update_stats((GaussianNIGStats*)stats, x);
}

static double gaussian_nig_predictive_logpdf_fn(const void* stats, const void* params, double x) {
    return gaussian_nig_predictive_logpdf(
        (const GaussianNIGParams*)params,
        (const GaussianNIGStats*)stats,
        x
    );
}

static void gaussian_nig_copy_stats_fn(void* dst, const void* src, const void* params) {
    (void)params;  // unused
    memcpy(dst, src, sizeof(GaussianNIGStats));
}

/* Student-t NG vtable functions */
static size_t student_t_ng_stats_size_fn(const void* params) {
    (void)params;  // unused
    return sizeof(StudentTNGStats);
}

static void student_t_ng_prior_stats_fn(void* stats, const void* params) {
    (void)params;  // unused
    student_t_ng_prior_stats((StudentTNGStats*)stats);
}

static void student_t_ng_update_stats_fn(void* stats, const void* params, double x) {
    student_t_ng_update_stats(
        (StudentTNGStats*)stats,
        (const StudentTNGParams*)params,
        x
    );
}

static double student_t_ng_predictive_logpdf_fn(const void* stats, const void* params, double x) {
    return student_t_ng_predictive_logpdf(
        (const StudentTNGParams*)params,
        (const StudentTNGStats*)stats,
        x
    );
}

static void student_t_ng_copy_stats_fn(void* dst, const void* src, const void* params) {
    (void)params;  // unused
    memcpy(dst, src, sizeof(StudentTNGStats));
}

/* Student-t NG Grid vtable functions */
static size_t student_t_ng_grid_stats_size_fn(const void* params) {
    const StudentTNGGridParams* p = (const StudentTNGGridParams*)params;
    return student_t_ng_grid_stats_size(p->K);
}

static void student_t_ng_grid_prior_stats_fn(void* stats, const void* params) {
    student_t_ng_grid_prior_stats(stats, (const StudentTNGGridParams*)params);
}

static void student_t_ng_grid_update_stats_fn(void* stats, const void* params, double x) {
    student_t_ng_grid_update_stats(stats, (const StudentTNGGridParams*)params, x);
}

static double student_t_ng_grid_predictive_logpdf_fn(const void* stats, const void* params, double x) {
    return student_t_ng_grid_predictive_logpdf(stats, (const StudentTNGGridParams*)params, x);
}

static void student_t_ng_grid_copy_stats_fn(void* dst, const void* src, const void* params) {
    student_t_ng_grid_copy_stats(dst, src, (const StudentTNGGridParams*)params);
}

/* Poisson-Gamma vtable functions */
static size_t poisson_gamma_stats_size_fn(const void* params) {
    (void)params;  // unused
    return sizeof(PoissonGammaStats);
}

static void poisson_gamma_prior_stats_fn(void* stats, const void* params) {
    (void)params;  // unused
    poisson_gamma_prior_stats((PoissonGammaStats*)stats);
}

static void poisson_gamma_update_stats_fn(void* stats, const void* params, double x) {
    poisson_gamma_update_stats(
        (PoissonGammaStats*)stats,
        (const PoissonGammaParams*)params,
        x
    );
}

static double poisson_gamma_predictive_logpdf_fn(const void* stats, const void* params, double x) {
    return poisson_gamma_predictive_logpdf(
        (const PoissonGammaParams*)params,
        (const PoissonGammaStats*)stats,
        x
    );
}

static void poisson_gamma_copy_stats_fn(void* dst, const void* src, const void* params) {
    (void)params;  // unused
    poisson_gamma_copy_stats(dst, src);
}

/* Bernoulli-Beta vtable functions */
static size_t bernoulli_beta_stats_size_fn(const void* params) {
    (void)params;  // unused
    return sizeof(BernoulliBetaStats);
}

static void bernoulli_beta_prior_stats_fn(void* stats, const void* params) {
    (void)params;  // unused
    bernoulli_beta_prior_stats((BernoulliBetaStats*)stats);
}

static void bernoulli_beta_update_stats_fn(void* stats, const void* params, double x) {
    bernoulli_beta_update_stats(
        (BernoulliBetaStats*)stats,
        (const BernoulliBetaParams*)params,
        x
    );
}

static double bernoulli_beta_predictive_logpdf_fn(const void* stats, const void* params, double x) {
    return bernoulli_beta_predictive_logpdf(
        (const BernoulliBetaParams*)params,
        (const BernoulliBetaStats*)stats,
        x
    );
}

static void bernoulli_beta_copy_stats_fn(void* dst, const void* src, const void* params) {
    (void)params;  // unused
    bernoulli_beta_copy_stats(dst, src);
}

/* Binomial-Beta vtable functions */
static size_t binomial_beta_stats_size_fn(const void* params) {
    (void)params;  // unused
    return sizeof(BinomialBetaStats);
}

static void binomial_beta_prior_stats_fn(void* stats, const void* params) {
    (void)params;  // unused
    binomial_beta_prior_stats((BinomialBetaStats*)stats);
}

static void binomial_beta_update_stats_fn(void* stats, const void* params, double x) {
    binomial_beta_update_stats(
        (BinomialBetaStats*)stats,
        (const BinomialBetaParams*)params,
        x
    );
}

static double binomial_beta_predictive_logpdf_fn(const void* stats, const void* params, double x) {
    return binomial_beta_predictive_logpdf(
        (const BinomialBetaParams*)params,
        (const BinomialBetaStats*)stats,
        x
    );
}

static void binomial_beta_copy_stats_fn(void* dst, const void* src, const void* params) {
    (void)params;  // unused
    binomial_beta_copy_stats(dst, src);
}

/* Gamma-Gamma vtable functions */
static size_t gamma_gamma_stats_size_fn(const void* params) {
    (void)params;  // unused
    return sizeof(GammaGammaStats);
}

static void gamma_gamma_prior_stats_fn(void* stats, const void* params) {
    (void)params;  // unused
    gamma_gamma_prior_stats((GammaGammaStats*)stats);
}

static void gamma_gamma_update_stats_fn(void* stats, const void* params, double x) {
    gamma_gamma_update_stats(
        (GammaGammaStats*)stats,
        (const GammaGammaParams*)params,
        x
    );
}

static double gamma_gamma_predictive_logpdf_fn(const void* stats, const void* params, double x) {
    return gamma_gamma_predictive_logpdf(
        (const GammaGammaParams*)params,
        (const GammaGammaStats*)stats,
        x
    );
}

static void gamma_gamma_copy_stats_fn(void* dst, const void* src, const void* params) {
    (void)params;  // unused
    gamma_gamma_copy_stats(dst, src);
}

static void init_obs_vtable(ObsModelVTable* vtable, ObsModelType type) {
    switch (type) {
        case OBS_MODEL_GAUSSIAN_NIG:
            vtable->stats_size = gaussian_nig_stats_size_fn;
            vtable->prior_stats = gaussian_nig_prior_stats_fn;
            vtable->update_stats = gaussian_nig_update_stats_fn;
            vtable->predictive_logpdf = gaussian_nig_predictive_logpdf_fn;
            vtable->copy_stats = gaussian_nig_copy_stats_fn;
            break;
        
        case OBS_MODEL_STUDENT_T_NG:
            vtable->stats_size = student_t_ng_stats_size_fn;
            vtable->prior_stats = student_t_ng_prior_stats_fn;
            vtable->update_stats = student_t_ng_update_stats_fn;
            vtable->predictive_logpdf = student_t_ng_predictive_logpdf_fn;
            vtable->copy_stats = student_t_ng_copy_stats_fn;
            break;
        
        case OBS_MODEL_STUDENT_T_NG_GRID:
            vtable->stats_size = student_t_ng_grid_stats_size_fn;
            vtable->prior_stats = student_t_ng_grid_prior_stats_fn;
            vtable->update_stats = student_t_ng_grid_update_stats_fn;
            vtable->predictive_logpdf = student_t_ng_grid_predictive_logpdf_fn;
            vtable->copy_stats = student_t_ng_grid_copy_stats_fn;
            break;
        
        case OBS_MODEL_POISSON_GAMMA:
            vtable->stats_size = poisson_gamma_stats_size_fn;
            vtable->prior_stats = poisson_gamma_prior_stats_fn;
            vtable->update_stats = poisson_gamma_update_stats_fn;
            vtable->predictive_logpdf = poisson_gamma_predictive_logpdf_fn;
            vtable->copy_stats = poisson_gamma_copy_stats_fn;
            break;
        
        case OBS_MODEL_BERNOULLI_BETA:
            vtable->stats_size = bernoulli_beta_stats_size_fn;
            vtable->prior_stats = bernoulli_beta_prior_stats_fn;
            vtable->update_stats = bernoulli_beta_update_stats_fn;
            vtable->predictive_logpdf = bernoulli_beta_predictive_logpdf_fn;
            vtable->copy_stats = bernoulli_beta_copy_stats_fn;
            break;
        
        case OBS_MODEL_BINOMIAL_BETA:
            vtable->stats_size = binomial_beta_stats_size_fn;
            vtable->prior_stats = binomial_beta_prior_stats_fn;
            vtable->update_stats = binomial_beta_update_stats_fn;
            vtable->predictive_logpdf = binomial_beta_predictive_logpdf_fn;
            vtable->copy_stats = binomial_beta_copy_stats_fn;
            break;
        
        case OBS_MODEL_GAMMA_GAMMA:
            vtable->stats_size = gamma_gamma_stats_size_fn;
            vtable->prior_stats = gamma_gamma_prior_stats_fn;
            vtable->update_stats = gamma_gamma_update_stats_fn;
            vtable->predictive_logpdf = gamma_gamma_predictive_logpdf_fn;
            vtable->copy_stats = gamma_gamma_copy_stats_fn;
            break;
        
        default:
            // This should never happen if bocpd_init validates properly
            memset(vtable, 0, sizeof(*vtable));
            break;
    }
}

// =============================================================================
// == Numerically Stable Log-Sum-Exp ===========================================
// =============================================================================

static double logsumexp_pair(double a, double b)
{
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    
    double m = (a > b) ? a : b;
    return m + log(exp(a - m) + exp(b - m));
}

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

// =============================================================================
// == State Initialization Helpers ============================================
// =============================================================================

static int copy_obs_params(BOCPDState* state,
                           ObsModelType obs_model_type,
                           const void* obs_params)
{
    switch (obs_model_type) {
        case OBS_MODEL_GAUSSIAN_NIG:
            state->obs_params.gaussian_nig = *(const GaussianNIGParams*)obs_params;
            state->obs_params_ptr = &state->obs_params.gaussian_nig;
            return 0;
            
        case OBS_MODEL_STUDENT_T_NG:
            state->obs_params.student_t_ng = *(const StudentTNGParams*)obs_params;
            state->obs_params_ptr = &state->obs_params.student_t_ng;
            return 0;
            
        case OBS_MODEL_STUDENT_T_NG_GRID: {
            const StudentTNGGridParams* grid_params = (const StudentTNGGridParams*)obs_params;
            
            if (student_t_ng_grid_validate_params(grid_params) != 0) {
                return -1;
            }
            
            int32_t K = grid_params->K;
            
            state->owned_nu_grid = (double*)malloc((size_t)K * sizeof(double));
            if (!state->owned_nu_grid) {
                return -1;
            }
            memcpy(state->owned_nu_grid, grid_params->nu_grid, (size_t)K * sizeof(double));
            
            state->owned_nu_prior = (double*)malloc((size_t)K * sizeof(double));
            if (!state->owned_nu_prior) {
                return -1;
            }
            if (student_t_ng_grid_normalize_prior(grid_params->nu_prior, K, state->owned_nu_prior) != 0) {
                return -1;
            }
            
            state->obs_params.student_t_ng_grid.mu0 = grid_params->mu0;
            state->obs_params.student_t_ng_grid.kappa0 = grid_params->kappa0;
            state->obs_params.student_t_ng_grid.alpha0 = grid_params->alpha0;
            state->obs_params.student_t_ng_grid.beta0 = grid_params->beta0;
            state->obs_params.student_t_ng_grid.K = K;
            state->obs_params.student_t_ng_grid.nu_grid = state->owned_nu_grid;
            state->obs_params.student_t_ng_grid.nu_prior = state->owned_nu_prior;
            state->obs_params_ptr = &state->obs_params.student_t_ng_grid;
            return 0;
        }
            
        case OBS_MODEL_POISSON_GAMMA: {
            const PoissonGammaParams* pg_params = (const PoissonGammaParams*)obs_params;
            if (!isfinite(pg_params->alpha0) || !isfinite(pg_params->beta0) ||
                !(pg_params->alpha0 > 0.0) || !(pg_params->beta0 > 0.0)) {
                return -1;
            }
            state->obs_params.poisson_gamma = *pg_params;
            state->obs_params_ptr = &state->obs_params.poisson_gamma;
            return 0;
        }
            
        case OBS_MODEL_BERNOULLI_BETA: {
            const BernoulliBetaParams* bb_params = (const BernoulliBetaParams*)obs_params;
            if (!isfinite(bb_params->alpha0) || !isfinite(bb_params->beta0) ||
                !(bb_params->alpha0 > 0.0) || !(bb_params->beta0 > 0.0)) {
                return -1;
            }
            state->obs_params.bernoulli_beta = *bb_params;
            state->obs_params_ptr = &state->obs_params.bernoulli_beta;
            return 0;
        }
        
        case OBS_MODEL_BINOMIAL_BETA: {
            const BinomialBetaParams* bb_params = (const BinomialBetaParams*)obs_params;
            if (!isfinite(bb_params->alpha0) || bb_params->alpha0 <= 0.0) {
                return -1;
            }
            if (!isfinite(bb_params->beta0) || bb_params->beta0 <= 0.0) {
                return -1;
            }
            if (bb_params->N < 1) {
                return -1;
            }
            state->obs_params.binomial_beta = *bb_params;
            state->obs_params.binomial_beta.log_N_factorial = lgamma((double)bb_params->N + 1.0);
            state->obs_params_ptr = &state->obs_params.binomial_beta;
            return 0;
        }
        
        case OBS_MODEL_GAMMA_GAMMA: {
            const GammaGammaParams* gg_params = (const GammaGammaParams*)obs_params;
            if (!isfinite(gg_params->alpha0) || gg_params->alpha0 <= 0.0) {
                return -1;
            }
            if (!isfinite(gg_params->beta0) || gg_params->beta0 <= 0.0) {
                return -1;
            }
            if (!isfinite(gg_params->shape) || gg_params->shape <= 0.0) {
                return -1;
            }
            state->obs_params.gamma_gamma = *gg_params;
            state->obs_params.gamma_gamma.log_gamma_k = lgamma(gg_params->shape);
            state->obs_params_ptr = &state->obs_params.gamma_gamma;
            return 0;
        }
    }
    return -1;
}

static int init_hazard_params(BOCPDState* state,
                              HazardType hazard_type,
                              const void* hazard_params)
{
    switch (hazard_type) {
        case HAZARD_CONSTANT:
            state->hazard_params.constant = *(const ConstantHazardParams*)hazard_params;
            return 0;
    }
    return -1;
}

static int allocate_state_buffers(BOCPDState* state)
{
    int32_t size = state->max_run_length + 1;
    size_t n_blobs = (size_t)size;
    
    if (state->stats_size != 0 && n_blobs > (SIZE_MAX / state->stats_size)) {
        return -1;
    }
    if (n_blobs > (SIZE_MAX / sizeof(double))) {
        return -1;
    }
    
    state->log_joint = (double*)malloc(n_blobs * sizeof(double));
    state->new_log_joint = (double*)malloc(n_blobs * sizeof(double));
    state->posterior_r = (double*)malloc(n_blobs * sizeof(double));
    state->stats = (uint8_t*)malloc(n_blobs * state->stats_size);
    state->new_stats = (uint8_t*)malloc(n_blobs * state->stats_size);
    
    if (!state->log_joint || !state->new_log_joint || !state->posterior_r ||
        !state->stats || !state->new_stats) {
        return -1;
    }
    
    bocpd_reset(state);
    return 0;
}

// =============================================================================
// == Public API ===============================================================
// =============================================================================

int bocpd_init(BOCPDState* state, ObsModelType obs_model_type, 
    const void* obs_params, HazardType hazard_type, const void* hazard_params,
    int32_t max_run_length) 
{
    if (!state || !obs_params || !hazard_params) {
        return -1;
    }
    
    if (max_run_length <= 0 || max_run_length >= INT32_MAX) {
        return -1;
    }
    
    memset(state, 0, sizeof(*state));
    state->max_run_length = max_run_length;
    state->obs_model_type = obs_model_type;
    state->hazard_type = hazard_type;
    
    if (copy_obs_params(state, obs_model_type, obs_params) != 0) {
        bocpd_free(state);
        return -1;
    }
    if (init_hazard_params(state, hazard_type, hazard_params) != 0) {
        bocpd_free(state);
        return -1;
    }
    
    init_obs_vtable(&state->obs_vtable, obs_model_type);
    
    size_t base_stats_size = state->obs_vtable.stats_size(state->obs_params_ptr);
    state->stats_size = round_up_align(base_stats_size, STATS_ALIGNMENT);
    
    if (allocate_state_buffers(state) != 0) {
        bocpd_free(state);
        return -1;
    }
    
    return 0;
}

void bocpd_free(BOCPDState* state) 
{
    if (!state) return;
    
    if (state->log_joint) free(state->log_joint);
    if (state->new_log_joint) free(state->new_log_joint);
    if (state->posterior_r) free(state->posterior_r);
    if (state->stats) free(state->stats);
    if (state->new_stats) free(state->new_stats);
    
    // Free owned grid arrays if present
    if (state->owned_nu_grid) free(state->owned_nu_grid);
    if (state->owned_nu_prior) free(state->owned_nu_prior);
    
    state->log_joint = NULL;
    state->new_log_joint = NULL;
    state->posterior_r = NULL;
    state->stats = NULL;
    state->new_stats = NULL;
    state->owned_nu_grid = NULL;
    state->owned_nu_prior = NULL;
}

void bocpd_reset(BOCPDState* state) 
{
    if (!state) return;
    
    int32_t size = state->max_run_length + 1;
    
    // Initialize log_joint to -inf
    for (int32_t i = 0; i < size; i++) {
        state->log_joint[i] = -INFINITY;
    }
    state->log_joint[0] = 0.0;  // log P(r_0=0, no data) = 0
    
    // Initialize posterior_r to prior belief (all mass at r=0)
    for (int32_t i = 0; i < size; i++) {
        state->posterior_r[i] = 0.0;
    }
    state->posterior_r[0] = 1.0;  // P(r_0=0 before any data) = 1
    
#ifdef BOCPD_DEBUG_CHECKS
    // Debug: Zero stats buffer to prevent reading uninitialized stats after reset
    memset(state->stats, 0, (size_t)size * state->stats_size);
#endif
    
    // Initialize stats[0] to prior using vtable
    void* stats_0 = stats_at(state->stats, 0, state->stats_size);
    state->obs_vtable.prior_stats(stats_0, state->obs_params_ptr);
}

double* bocpd_update(BOCPDState* state, double x, double* cp_prob_out) 
{
    if (!state) return NULL;
    
    int32_t R = state->max_run_length;
    size_t stats_size = state->stats_size;
    ObsModelVTable* vtable = &state->obs_vtable;
    const void* params = state->obs_params_ptr;  // Use stored pointer (proper aliasing)
    
    // -------------------------------------------------------------------------
    // Prepare working buffers for this update
    // -------------------------------------------------------------------------
    for (int32_t i = 0; i <= R; i++) {
        state->new_log_joint[i] = -INFINITY;
    }
    
#ifdef BOCPD_DEBUG_CHECKS
    // Debug: Zero new_stats buffer to catch any bugs where we read uninitialized stats
    // In release builds, this is unnecessary since we only read stats[r] if log_joint[r] != -inf,
    // and we guarantee to write stats[r] before setting log_joint[r] to non-inf.
    memset(state->new_stats, 0, (size_t)(R + 1) * stats_size);
#endif
    
    // -------------------------------------------------------------------------
    // Changepoint branch: start new run-length at r=0 using prior stats
    // -------------------------------------------------------------------------
    void* new_stats_0 = stats_at(state->new_stats, 0, stats_size);
    vtable->prior_stats(new_stats_0, params);
    
    // CRITICAL: Compute CP predictive using PRIOR-to-x stats
    double log_pred_cp = vtable->predictive_logpdf(new_stats_0, params, x);
    
    // Guard against invalid observations that would brick the filter
    // (e.g., non-binary for Bernoulli, non-integer for Poisson)
    if (log_pred_cp == -INFINITY || isnan(log_pred_cp)) {
        return NULL;  // Signal error upward (invalid data)
    }
    
    // Update r=0 stats to incorporate x (for next time step)
    vtable->update_stats(new_stats_0, params, x);
    
    // -------------------------------------------------------------------------
    // Continuation branches: extend every previous run-length
    // -------------------------------------------------------------------------
    for (int32_t r_prev = 0; r_prev <= R; r_prev++) {
        double lj_prev = state->log_joint[r_prev];
        
        if (lj_prev == -INFINITY) {
            continue;
        }
        
        // Get stats for this run length (pre-x stats, old buffer)
        const void* stats_prev = cstats_at(state->stats, r_prev, stats_size);
        
        // Predictive log likelihood using PRE-X stats (critical: use OLD stats)
        double log_pred = vtable->predictive_logpdf(stats_prev, params, x);
        
        // Hazard transitions (hazard-specific)
        double log_trans_cp, log_trans_cont;
        switch (state->hazard_type) {
            case HAZARD_CONSTANT:
                log_trans_cp = constant_hazard_log_transition_cp(&state->hazard_params.constant, r_prev);
                log_trans_cont = constant_hazard_log_transition_cont(&state->hazard_params.constant, r_prev);
                break;
            default:
                return NULL;
        }
        
        // Changepoint branch: r_t = 0
        double logp_cp = lj_prev + log_pred_cp + log_trans_cp;
        state->new_log_joint[0] = logsumexp_pair(state->new_log_joint[0], logp_cp);
        
        // Continuation branch: r_t = r_prev + 1
        // Each continuation run has a single parent (r_prev = r_cont - 1)
        int32_t r_cont = r_prev + 1;
        if (r_cont <= R) {
            double logp_cont = lj_prev + log_pred + log_trans_cont;
            state->new_log_joint[r_cont] = logsumexp_pair(state->new_log_joint[r_cont], logp_cont);
            
            // Copy stats from r_prev to r_cont (NEW buffer), then update with x.
            // If future pruning/binning maps multiple parents to one child,
            // this direct copy/update logic will need to aggregate stats.
            void* new_stats_cont = stats_at(state->new_stats, r_cont, stats_size);
            vtable->copy_stats(new_stats_cont, stats_prev, params);
            vtable->update_stats(new_stats_cont, params, x);
        }
    }
    
    // -------------------------------------------------------------------------
    // Normalize to get posterior over run length and swap buffers
    // -------------------------------------------------------------------------
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
    
    // Swap buffers (old <-> new)
    double* tmp_log = state->log_joint;
    state->log_joint = state->new_log_joint;
    state->new_log_joint = tmp_log;
    
    uint8_t* tmp_stats = state->stats;
    state->stats = state->new_stats;
    state->new_stats = tmp_stats;
    
    // Output changepoint probability if requested
    if (cp_prob_out) {
        *cp_prob_out = state->posterior_r[0];
    }
    
    return state->posterior_r;
}

int bocpd_batch_update(BOCPDState* state, const double* x_array, int32_t n_obs,
                        double* cp_probs_out) 
{
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

int32_t bocpd_get_map_run_length(const BOCPDState* state)
{
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

int bocpd_get_posterior(const BOCPDState* state, double* posterior_out)
{
    if (!state || !state->posterior_r || !posterior_out) {
        return -1;
    }
    
    for (int32_t r = 0; r <= state->max_run_length; r++) {
        posterior_out[r] = state->posterior_r[r];
    }
    
    return 0;
}
