"""
Observation model parameter wrappers.

The actual implementations are in C, these are just Python wrappers for validation.
"""


class GaussianNIG:
    """
    1D Gaussian likelihood with Normal-Inverse-Gamma prior.

    Prior hyperparameters:
        mu0: Prior mean
        kappa0: Prior precision scaling (must be > 0)
        alpha0: Prior shape parameter (must be > 0)
        beta0: Prior scale parameter (must be > 0)
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


# Future models will be added here:
# class PoissonGamma:
#     def __init__(self, alpha0: float, beta0: float):
#         ...
