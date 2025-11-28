#include "hazard.h"
#include <math.h>

int constant_hazard_init(ConstantHazardParams* params, double lambda) 
{
    if (lambda <= 0.0) {
        return -1;  // Invalid lambda
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
    // For constant hazard, changepoint probability is always log(H)
    // r_prev is unused
    (void)r_prev;  // Suppress unused parameter warning
    return params->log_H;
}

double constant_hazard_log_transition_cont(const ConstantHazardParams* params, int32_t r_prev)
{
    // For constant hazard, continuation probability is always log(1 - H)
    // r_prev is unused
    (void)r_prev;  // Suppress unused parameter warning
    return params->log_1mH;
}
