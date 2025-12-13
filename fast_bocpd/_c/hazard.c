#include <math.h>

#include "hazard.h"

// =============================================================================
// == Public API ===============================================================
// =============================================================================

int constant_hazard_init(ConstantHazardParams* params, double lambda)
{
    if (!params) {
        return -1;
    }
    if (!isfinite(lambda) || lambda <= 0.0) {
        return -1;
    }

    params->lambda = lambda;
    double H = 1.0 / lambda;
    
    if (H <= 0.0 || H >= 1.0) {
        return -1;  // Hazard must be in (0, 1)
    }

    params->log_H = log(H);
    params->log_1mH = log(1.0 - H);
    
    return 0;
}

double constant_hazard_log_transition_cp(const ConstantHazardParams* params, int32_t r_prev)
{
    (void)r_prev;
    if (!params) {
        return -INFINITY;
    }
    return params->log_H;
}

double constant_hazard_log_transition_cont(const ConstantHazardParams* params, int32_t r_prev)
{
    (void)r_prev;
    if (!params) {
        return -INFINITY;
    }
    return params->log_1mH;
}
