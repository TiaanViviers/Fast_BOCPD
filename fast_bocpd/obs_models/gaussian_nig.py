import math
from typing import Tuple


class GaussianNIG:
    """
    1D Gaussian likelihood with Normal-Inverse-Gamma prior.

    Prior hyperparameters:
        mu0, kappa0, alpha0, beta0

    Sufficient stats per run:
        n, sum_x, sum_x2
    """

    def __init__(self, mu0: float, kappa0: float, alpha0: float, beta0: float):
        if kappa0 <= 0:
            raise ValueError("kappa0 must be > 0")
        if alpha0 <= 0:
            raise ValueError("alpha0 must be > 0")
        if beta0 <= 0:
            raise ValueError("beta0 must be > 0")

        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)


    # ---------- Stats management ----------
    def prior_stats(self) -> Tuple[int, float, float]:
        """Return empty-run sufficient stats: no data yet."""
        return (0, 0.0, 0.0)

    def update_stats(self, stats: Tuple[int, float, float], 
                     x: float) -> Tuple[int, float, float]:
        """
        Update sufficient stats with a new observation x.
        """
        n, s, s2 = stats
        n_new = n + 1
        return (n_new, s + x, s2 + x * x)


    # ---------- Internal: posterior hyperparams ----------
    def _posterior_hyperparams(self, stats: Tuple[int, float, float]
                              ) -> Tuple[float, float, float, float]:
        """
        Compute posterior NIG hyperparameters given sufficient stats.
        Returns (mu_n, kappa_n, alpha_n, beta_n).
        """
        n, sum_x, sum_x2 = stats

        if n == 0:
            # No data yet: posterior == prior
            return self.mu0, self.kappa0, self.alpha0, self.beta0

        x_bar = sum_x / n
        # Sum of squared deviations from the mean
        S = sum_x2 - n * x_bar * x_bar

        kappa_n = self.kappa0 + n
        mu_n = (self.kappa0 * self.mu0 + n * x_bar) / kappa_n
        alpha_n = self.alpha0 + 0.5 * n

        beta_n = self.beta0 + 0.5 * (
            S + (self.kappa0 * n / kappa_n) * (x_bar - self.mu0) ** 2
        )
        return mu_n, kappa_n, alpha_n, beta_n
    

    # ---------- Public: predictive log pdf ----------
    def predictive_logpdf(self, stats: Tuple[int, float, float], x: float
                          ) -> float:
        """
        Log predictive density of x under the Student-t induced by NIG posterior.

        x | data ~ Student-t(ν=2α_n, loc=μ_n, scale^2 = β_n (κ_n+1)/(α_n κ_n))
        """
        mu_n, kappa_n, alpha_n, beta_n = self._posterior_hyperparams(stats)

        nu = 2.0 * alpha_n  # degrees of freedom
        # predictive variance from conjugate formulas
        scale2 = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n)
        scale = math.sqrt(scale2)

        z = (x - mu_n) / scale

        # Student-t log pdf
        # log Γ((ν+1)/2) - log Γ(ν/2) - 0.5 log(νπ) - log(scale)
        # - (ν+1)/2 * log(1 + (z^2)/ν)
        log_norm = (
            math.lgamma((nu + 1.0) / 2.0)
            - math.lgamma(nu / 2.0)
            - 0.5 * math.log(nu * math.pi)
            - math.log(scale)
        )
        log_kernel = - (nu + 1.0) / 2.0 * math.log(1.0 + (z * z) / nu)
        return log_norm + log_kernel
