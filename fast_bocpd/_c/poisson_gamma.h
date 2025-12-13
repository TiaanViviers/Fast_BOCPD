#ifndef POISSON_GAMMA_H
#define POISSON_GAMMA_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    double alpha0;
    double beta0;
} PoissonGammaParams;

typedef struct {
    int64_t n;      // Use int64_t to avoid overflow at 2.1B observations
    double sum_x;
} PoissonGammaStats;

size_t poisson_gamma_stats_size(void);
void poisson_gamma_prior_stats(PoissonGammaStats* stats);
void poisson_gamma_update_stats(PoissonGammaStats* stats,
                                const PoissonGammaParams* params,
                                double x);
double poisson_gamma_predictive_logpdf(const PoissonGammaParams* params,
                                       const PoissonGammaStats* stats,
                                       double x);
void poisson_gamma_copy_stats(void* dst, const void* src);

#endif
