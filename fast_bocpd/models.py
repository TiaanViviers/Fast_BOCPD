"""
Observation model parameter wrappers.

The actual implementations are in C, these are just Python wrappers for validation.
"""
import numpy as np
from typing import Optional, Union, List


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


class StudentTNG:
    """
    Student-t likelihood with Normal-Gamma conjugate prior.
    
    More robust to outliers than Gaussian.
    
    Supports two modes:
    1. Fixed ν: Pass a single value (standard Student-t)
    2. Grid ν: Pass nu_grid and optional nu_prior for ν inference
    
    Prior hyperparameters:
        mu0: Prior mean
        kappa0: Prior precision scaling (must be > 0)
        alpha0: Prior shape parameter (must be > 0)
        beta0: Prior rate parameter (must be > 0)
        nu: Degrees of freedom (single value OR array for grid)
            - nu = 1: Cauchy distribution (very heavy tails)
            - nu = 3-5: Good for financial data
            - nu → ∞: Approaches Gaussian
        nu_prior: Prior weights over nu_grid (optional, defaults to uniform)
    
    Examples:
        # Fixed ν (standard Student-t)
        >>> model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=3.0)
        
        # Grid ν (infer best ν from data)
        >>> model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1,
        ...                    nu_grid=[2, 3, 5, 10, 20])
        
        # Grid ν with custom prior
        >>> model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1,
        ...                    nu_grid=[2, 3, 5], nu_prior=[0.2, 0.5, 0.3])
    """

    def __init__(
        self, 
        mu0: float, 
        kappa0: float, 
        alpha0: float, 
        beta0: float, 
        nu: Union[float, List[float], np.ndarray] = 3.0,
        nu_prior: Optional[Union[List[float], np.ndarray]] = None
    ):
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
        
        # Handle both fixed ν and grid ν (support scalar, list, array, tuple)
        nu_arr = np.asarray(nu, dtype=np.float64)
        
        if nu_arr.ndim == 0:
            # Fixed ν mode (scalar)
            nu_val = float(nu_arr)
            if nu_val <= 0:
                raise ValueError("nu (degrees of freedom) must be > 0")
            self.nu = nu_val
            self.is_grid = False
            self.nu_grid = None
            self.nu_prior = None
            self.K = None
        else:
            # Grid mode (array-like)
            self.nu_grid = np.ascontiguousarray(nu_arr, dtype=np.float64)
            if len(self.nu_grid) == 0:
                raise ValueError("nu_grid cannot be empty")
            if np.any(self.nu_grid <= 0):
                raise ValueError("All nu values must be > 0")
            if not np.all(np.isfinite(self.nu_grid)):
                raise ValueError("All nu values must be finite")
            
            self.K = len(self.nu_grid)
            self.is_grid = True
            self.nu = None  # Not used in grid mode
            
            # Handle prior
            if nu_prior is None:
                # Uniform prior
                self.nu_prior = np.ones(self.K, dtype=np.float64)
                self.nu_prior /= self.K
            else:
                prior_arr = np.asarray(nu_prior, dtype=np.float64)
                if prior_arr.ndim != 1:
                    raise ValueError("nu_prior must be 1-D array-like")
                self.nu_prior = np.ascontiguousarray(prior_arr, dtype=np.float64)
                
                if len(self.nu_prior) != self.K:
                    raise ValueError(f"nu_prior length ({len(self.nu_prior)}) must match nu_grid length ({self.K})")
                if np.any(self.nu_prior < 0):
                    raise ValueError("All nu_prior values must be >= 0")
                if not np.all(np.isfinite(self.nu_prior)):
                    raise ValueError("All nu_prior values must be finite")
                if np.sum(self.nu_prior) == 0:
                    raise ValueError("nu_prior must have at least one positive value")
                # Normalize
                self.nu_prior = self.nu_prior / np.sum(self.nu_prior)


