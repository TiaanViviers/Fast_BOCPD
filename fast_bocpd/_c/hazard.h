#ifndef HAZARD_H
#define HAZARD_H

#include <stdint.h>

/**
 * Constant hazard parameters
 */
typedef struct {
    double lambda;
    double log_H;      // Precomputed log(1/lambda)
    double log_1mH;    // Precomputed log(1 - 1/lambda)
} ConstantHazardParams;

/**
 * Initialize constant hazard parameters
 * 
 * @param params    Hazard parameters to initialize
 * @param lambda    Expected run length
 * @return          0 on success, -1 on error (invalid lambda)
 */
int constant_hazard_init(ConstantHazardParams* params, double lambda);

/**
 * Log probability of changepoint transition
 * 
 * @param params    Hazard parameters
 * @param r_prev    Previous run length (unused for constant hazard)
 * @return          log P(changepoint)
 */
double constant_hazard_log_transition_cp(const ConstantHazardParams* params, int32_t r_prev);

/**
 * Log probability of continuation transition
 * 
 * @param params    Hazard parameters
 * @param r_prev    Previous run length (unused for constant hazard)
 * @return          log P(continuation)
 */
double constant_hazard_log_transition_cont(const ConstantHazardParams* params, int32_t r_prev);

#endif // HAZARD_H
