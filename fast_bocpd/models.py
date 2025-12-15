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
    r"""
    Student-t observation model with Normal-Gamma prior.

    This model is more robust to outliers than Gaussian-NIG. Two modes are
    supported:

    * **Fixed :math:`\nu`** – pass a scalar ``nu`` (standard Student-t).
    * **Grid :math:`\nu`** – pass ``nu`` as a list/array and optionally
      ``nu_prior`` to place a discrete prior over different degrees of
      freedom.

    Parameters
    ----------
    mu0 : float
        Prior mean.
    kappa0 : float
        Prior precision scaling (> 0).
    alpha0 : float
        Prior shape parameter (> 0).
    beta0 : float
        Prior rate parameter (> 0).
    nu : float or array-like, optional
        Degrees of freedom. When ``nu`` is an array we infer the best value
        online via a grid mixture. ``nu = 1`` behaves like Cauchy, ``nu``
        in [3, 5] is often used for financial data, and
        :math:`\nu \to \infty` approaches Gaussian.
    nu_prior : array-like, optional
        Prior weights over ``nu`` (grid mode). Defaults to uniform weights.

    Examples
    --------
    Fixed ν::

        >>> model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=3.0)

    Grid ν::

        >>> model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1,
        ...                    nu=[2, 3, 5, 10, 20])

    Grid ν with custom prior::

        >>> model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1,
        ...                    nu=[2, 3, 5], nu_prior=[0.2, 0.5, 0.3])
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


class PoissonGamma:
    """
    Poisson likelihood with Gamma prior on rate parameter (count data).
    
    Conjugate Bayesian model for non-negative integer count data.
    Predictive distribution is Negative Binomial.
    
    Use this for:
        - Event counts (clicks, transactions, arrivals)
        - Discrete data with λ > 0
        - Overdispersed counts (vs. fixed-rate Poisson)
    
    Prior hyperparameters:
        alpha0: Gamma prior shape (must be > 0)
                Controls prior belief about rate variability
                Larger values = more concentrated prior
        beta0:  Gamma prior rate (must be > 0)
                Controls prior belief about expected rate
                Prior mean rate = alpha0 / beta0
    
    Data requirements:
        - Must be non-negative integers (0, 1, 2, ...)
        - In Python, pass as int, float with .0, or integer array
        - Non-integer or negative values will be rejected (strict=True)
    
    Examples:
        # Event counts with vague prior
        >>> model = PoissonGamma(alpha0=1.0, beta0=1.0)
        
        # Prior belief: mean rate ≈ 5.0, concentrated
        >>> model = PoissonGamma(alpha0=50.0, beta0=10.0)  # mean = 50/10 = 5
        
        # Disable strict validation (use with caution)
        >>> model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=False)
    """
    
    def __init__(
        self, 
        alpha0: float, 
        beta0: float,
        *,
        strict: bool = True
    ):
        # Validate hyperparameters (always, even if strict=False)
        if not isinstance(alpha0, (int, float, np.number)):
            raise TypeError(f"alpha0 must be numeric, got {type(alpha0)}")
        if not isinstance(beta0, (int, float, np.number)):
            raise TypeError(f"beta0 must be numeric, got {type(beta0)}")
        
        alpha0 = float(alpha0)
        beta0 = float(beta0)
        
        if not np.isfinite(alpha0):
            raise ValueError("alpha0 must be finite")
        if not np.isfinite(beta0):
            raise ValueError("beta0 must be finite")
        if alpha0 <= 0:
            raise ValueError("alpha0 must be > 0")
        if beta0 <= 0:
            raise ValueError("beta0 must be > 0")
        
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.strict = bool(strict)
    
    def validate_data(self, x):
        """
        Validate a single observation (used in update()).
        
        Args:
            x: Observation (should be non-negative integer)
        
        Raises:
            ValueError: If x is invalid and strict=True
        """
        if not self.strict:
            return  # Skip validation
        
        # Check finite
        if not np.isfinite(x):
            raise ValueError(f"Observation must be finite, got {x}")
        
        # Check non-negative
        if x < 0:
            raise ValueError(f"Poisson counts must be >= 0, got {x}")
        
        # Check integer-ness (tolerance for floating point)
        if abs(x - round(x)) > 1e-9:
            raise ValueError(
                f"Poisson counts must be integers, got {x}. "
                f"If your data is truly continuous, use GaussianNIG or StudentTNG instead."
            )
    
    def validate_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Validate and convert batch data to contiguous float64 array.
        
        Args:
            data: Array-like of observations
        
        Returns:
            Validated, contiguous float64 array
        
        Raises:
            ValueError: If data is invalid and strict=True
        """
        # Convert to numpy array if not already
        data = np.asarray(data)
        
        # Fast-path for integer dtypes (common case)
        if np.issubdtype(data.dtype, np.integer):
            if self.strict and np.any(data < 0):
                raise ValueError("Poisson counts must be >= 0")
            return np.ascontiguousarray(data, dtype=np.float64)
        
        # Float array validation (if strict)
        if self.strict:
            if not np.all(np.isfinite(data)):
                raise ValueError("All observations must be finite")
            if np.any(data < 0):
                raise ValueError("Poisson counts must be >= 0")
            
            # Check integer-ness
            if not np.allclose(data, np.round(data), atol=1e-9):
                raise ValueError(
                    "Poisson counts must be integers. "
                    "If your data is truly continuous, use GaussianNIG or StudentTNG instead."
                )
        
        return np.ascontiguousarray(data, dtype=np.float64)


class BernoulliBeta:
    """
    Bernoulli likelihood with Beta prior on success probability (binary data).
    
    Conjugate Bayesian model for binary outcomes (0/1, success/failure).
    Predictive distribution is Beta-Bernoulli.
    
    Use this for:
        - Binary classification (yes/no, pass/fail)
        - Conversion rates (click/no-click)
        - A/B testing outcomes
        - Any {0, 1} data
    
    Prior hyperparameters:
        alpha0: Beta prior successes (must be > 0)
                Controls prior belief about success probability
                Larger values = more weight on high success rates
        beta0:  Beta prior failures (must be > 0)
                Controls prior belief about failure probability
                Larger values = more weight on low success rates
                Prior mean = alpha0 / (alpha0 + beta0)
    
    Data requirements:
        - Must be binary: 0 or 1
        - In Python, pass as int, bool, or float with .0
        - Non-binary values will be rejected (strict=True)
    
    Examples:
        # Vague prior (uniform over [0,1])
        >>> model = BernoulliBeta(alpha0=1.0, beta0=1.0)
        
        # Prior belief: success rate ≈ 0.3, concentrated
        >>> model = BernoulliBeta(alpha0=30.0, beta0=70.0)  # mean = 30/100 = 0.3
        
        # Disable strict validation (use with caution)
        >>> model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=False)
    """
    
    def __init__(
        self,
        alpha0: float,
        beta0: float,
        *,
        strict: bool = True
    ):
        # Validate hyperparameters (always, even if strict=False)
        if not isinstance(alpha0, (int, float, np.number)):
            raise TypeError(f"alpha0 must be numeric, got {type(alpha0)}")
        if not isinstance(beta0, (int, float, np.number)):
            raise TypeError(f"beta0 must be numeric, got {type(beta0)}")
        
        alpha0 = float(alpha0)
        beta0 = float(beta0)
        
        if not np.isfinite(alpha0):
            raise ValueError("alpha0 must be finite")
        if not np.isfinite(beta0):
            raise ValueError("beta0 must be finite")
        if alpha0 <= 0:
            raise ValueError("alpha0 must be > 0")
        if beta0 <= 0:
            raise ValueError("beta0 must be > 0")
        
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.strict = bool(strict)
    
    def validate_data(self, x):
        """
        Validate a single observation (used in update()).
        
        Args:
            x: Observation (should be 0 or 1)
        
        Raises:
            ValueError: If x is invalid and strict=True
        """
        if not self.strict:
            return  # Skip validation
        
        # Check finite
        if not np.isfinite(x):
            raise ValueError(f"Observation must be finite, got {x}")
        
        # Check binary-ness (tolerance for floating point)
        x_rounded = round(x)
        if abs(x - x_rounded) > 1e-9:
            raise ValueError(
                f"Bernoulli data must be binary (0 or 1), got {x}. "
                f"If your data is continuous, use GaussianNIG or StudentTNG instead."
            )
        
        # Check range
        if x_rounded not in (0, 1):
            raise ValueError(
                f"Bernoulli data must be 0 or 1, got {x}. "
                f"If your data is continuous, use GaussianNIG or StudentTNG instead."
            )
    
    def validate_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Validate and convert batch data to contiguous float64 array.
        
        Args:
            data: Array-like of observations
        
        Returns:
            Validated, contiguous float64 array
        
        Raises:
            ValueError: If data is invalid and strict=True
        """
        # Convert to numpy array if not already
        data = np.asarray(data)
        
        # Fast-path for boolean dtype
        if data.dtype == np.bool_:
            return np.ascontiguousarray(data, dtype=np.float64)
        
        # Fast-path for integer dtypes (common case)
        if np.issubdtype(data.dtype, np.integer):
            if self.strict:
                unique_vals = np.unique(data)
                if not np.all((unique_vals == 0) | (unique_vals == 1)):
                    raise ValueError("Bernoulli data must be binary (0 or 1)")
            return np.ascontiguousarray(data, dtype=np.float64)
        
        # Float array validation (if strict)
        if self.strict:
            if not np.all(np.isfinite(data)):
                raise ValueError("All observations must be finite")
            
            # Check binary-ness
            if not np.allclose(data, np.round(data), atol=1e-9):
                raise ValueError(
                    "Bernoulli data must be binary (0 or 1). "
                    "If your data is continuous, use GaussianNIG or StudentTNG instead."
                )
            
            # Check range
            rounded = np.round(data)
            if not np.all((rounded == 0) | (rounded == 1)):
                raise ValueError(
                    "Bernoulli data must be 0 or 1. "
                    "If your data is continuous, use GaussianNIG or StudentTNG instead."
                )
        
        return np.ascontiguousarray(data, dtype=np.float64)


class BinomialBeta:
    """
    Binomial likelihood with Beta prior on success probability (fixed-N count data).
    
    Conjugate Bayesian model for binomial outcomes with fixed number of trials.
    Predictive distribution is Beta-Binomial.
    
    Use this for:
        - Conversion rates per period (k successes in N trials)
        - A/B testing with fixed sample sizes
        - Aggregated binary outcomes
        - Any {0, 1, ..., N} count data
    
    Prior hyperparameters:
        alpha0: Beta prior successes (must be > 0)
                Controls prior belief about success probability
                Larger values = more weight on high success rates
        beta0:  Beta prior failures (must be > 0)
                Controls prior belief about failure probability
                Larger values = more weight on low success rates
                Prior mean = alpha0 / (alpha0 + beta0)
        n_trials: Fixed number of trials per observation (must be >= 1)
                  Each observation is k ∈ {0, 1, ..., n_trials}
    
    Data requirements:
        - Must be integers: 0, 1, 2, ..., n_trials
        - In Python, pass as int or float with .0
        - Values > n_trials or non-integer will be rejected (strict=True)
    
    Special case:
        - n_trials=1 reduces to Bernoulli-Beta (identical predictive)
    
    Examples:
        # Conversion rate: 10 trials per period
        >>> model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10)
        
        # Prior belief: success rate ≈ 0.3, N=20 trials
        >>> model = BinomialBeta(alpha0=30.0, beta0=70.0, n_trials=20)
        
        # Disable strict validation (use with caution)
        >>> model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=False)
    """
    
    def __init__(
        self,
        alpha0: float,
        beta0: float,
        n_trials: int,
        *,
        strict: bool = True
    ):
        # Validate hyperparameters (always, even if strict=False)
        if not isinstance(alpha0, (int, float, np.number)):
            raise TypeError(f"alpha0 must be numeric, got {type(alpha0)}")
        if not isinstance(beta0, (int, float, np.number)):
            raise TypeError(f"beta0 must be numeric, got {type(beta0)}")
        if not isinstance(n_trials, (int, np.integer)):
            raise TypeError(f"n_trials must be integer, got {type(n_trials)}")
        
        alpha0 = float(alpha0)
        beta0 = float(beta0)
        n_trials = int(n_trials)
        
        if not np.isfinite(alpha0):
            raise ValueError("alpha0 must be finite")
        if not np.isfinite(beta0):
            raise ValueError("beta0 must be finite")
        if alpha0 <= 0:
            raise ValueError("alpha0 must be > 0")
        if beta0 <= 0:
            raise ValueError("beta0 must be > 0")
        if n_trials < 1:
            raise ValueError("n_trials must be >= 1")
        
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.n_trials = n_trials
        self.strict = bool(strict)
    
    def validate_data(self, k):
        """
        Validate a single observation (used in update()).
        
        Args:
            k: Observation (should be integer in 0..n_trials)
        
        Raises:
            ValueError: If k is invalid and strict=True
        """
        if not self.strict:
            return  # Skip validation
        
        # Check finite
        if not np.isfinite(k):
            raise ValueError(f"Observation must be finite, got {k}")
        
        # Check non-negative
        if k < 0:
            raise ValueError(f"Binomial counts must be >= 0, got {k}")
        
        # Check integer-ness (tolerance for floating point)
        if abs(k - round(k)) > 1e-9:
            raise ValueError(
                f"Binomial counts must be integers, got {k}. "
                f"If your data is truly continuous, use GaussianNIG or StudentTNG instead."
            )
        
        # Check range
        if round(k) > self.n_trials:
            raise ValueError(
                f"Binomial count must be <= n_trials ({self.n_trials}), got {k}"
            )
    
    def validate_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Validate and convert batch data to contiguous float64 array.
        
        Args:
            data: Array-like of observations
        
        Returns:
            Validated, contiguous float64 array
        
        Raises:
            ValueError: If data is invalid and strict=True
        """
        # Convert to numpy array if not already
        data = np.asarray(data)
        
        # Fast-path for integer dtypes (common case)
        if np.issubdtype(data.dtype, np.integer):
            if self.strict:
                if np.any(data < 0):
                    raise ValueError("Binomial counts must be >= 0")
                if np.any(data > self.n_trials):
                    raise ValueError(f"Binomial counts must be <= n_trials ({self.n_trials})")
            return np.ascontiguousarray(data, dtype=np.float64)
        
        # Float array validation (if strict)
        if self.strict:
            if not np.all(np.isfinite(data)):
                raise ValueError("All observations must be finite")
            if np.any(data < 0):
                raise ValueError("Binomial counts must be >= 0")
            
            # Check integer-ness
            if not np.allclose(data, np.round(data), atol=1e-9):
                raise ValueError(
                    "Binomial counts must be integers. "
                    "If your data is truly continuous, use GaussianNIG or StudentTNG instead."
                )
            
            # Check range on rounded values (to handle near-integer floats correctly)
            rounded = np.round(data)
            if np.any(rounded > self.n_trials):
                raise ValueError(f"Binomial counts must be <= n_trials ({self.n_trials})")
        
        return np.ascontiguousarray(data, dtype=np.float64)


class GammaGamma:
    """
    Gamma likelihood with Gamma prior on rate parameter (fixed-shape scale data).
    
    Conjugate Bayesian model for positive continuous data with fixed shape.
    Predictive distribution is Beta-prime-like.
    
    Likelihood: x ~ Gamma(shape=k, rate=λ)  [fixed k, unknown λ]
    Prior:      λ ~ Gamma(alpha0, beta0)
    
    Use this for:
        - Positive continuous data with fixed shape
        - Exponential data (special case k=1)
        - Waiting times, survival data, reliability analysis
        - Data with constant coefficient of variation
    
    Prior hyperparameters:
        alpha0: Prior shape parameter on rate λ (must be > 0)
                Controls prior belief about the rate
                Larger values = stronger prior belief
        beta0:  Prior rate parameter on rate λ (must be > 0)
                Controls scale of prior on rate
                Prior mean E[λ] = alpha0 / beta0
        shape:  Fixed shape parameter k of Gamma likelihood (default 1.0)
                Recommended: k >= 1 for well-behaved densities
                Special case: k=1 gives Exponential distribution
    
    Data requirements:
        - Must be positive: x > 0
        - x=0 is rejected with -∞ log density (k > 1) or special case (k ≈ 1)
        - In strict mode: enforces shape >= 1 (recommended for stability)
    
    Parameterization note:
        Uses RATE parameterization (not scale):
        - Higher rate → smaller values
        - Mean of Gamma(k, λ) = k/λ
        - Variance = k/λ²
    
    Special case:
        - shape=1.0: Reduces to Exponential-Gamma (conjugate for exponential data)
    
    Examples:
        # Exponential waiting times (default)
        >>> model = GammaGamma(alpha0=1.0, beta0=1.0)  # shape=1.0
        
        # Prior: mean rate ≈ 2.0, shape=2.0
        >>> model = GammaGamma(alpha0=10.0, beta0=5.0, shape=2.0)
        
        # Disable strict mode (allow shape < 1, use with caution)
        >>> model = GammaGamma(alpha0=1.0, beta0=1.0, shape=0.5, strict=False)
    """
    
    def __init__(
        self,
        alpha0: float,
        beta0: float,
        shape: float = 1.0,
        *,
        strict: bool = True
    ):
        # Validate hyperparameters (always, even if strict=False)
        if not isinstance(alpha0, (int, float, np.number)):
            raise TypeError(f"alpha0 must be numeric, got {type(alpha0)}")
        if not isinstance(beta0, (int, float, np.number)):
            raise TypeError(f"beta0 must be numeric, got {type(beta0)}")
        if not isinstance(shape, (int, float, np.number)):
            raise TypeError(f"shape must be numeric, got {type(shape)}")
        
        alpha0 = float(alpha0)
        beta0 = float(beta0)
        shape = float(shape)
        
        if not np.isfinite(alpha0):
            raise ValueError("alpha0 must be finite")
        if not np.isfinite(beta0):
            raise ValueError("beta0 must be finite")
        if not np.isfinite(shape):
            raise ValueError("shape must be finite")
        
        if alpha0 <= 0:
            raise ValueError("alpha0 must be > 0")
        if beta0 <= 0:
            raise ValueError("beta0 must be > 0")
        if shape <= 0:
            raise ValueError("shape must be > 0")
        
        # Strict mode: enforce shape >= 1 recommendation
        if strict and shape < 1.0:
            raise ValueError(
                "shape must be >= 1.0 in strict mode (recommended for well-behaved densities). "
                "Set strict=False to allow shape < 1, but use with caution."
            )
        
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.shape = shape
        self.strict = bool(strict)
    
    def validate_data(self, x):
        """
        Validate a single observation (used in update()).
        
        Args:
            x: Observation (should be positive)
        
        Raises:
            ValueError: If x is invalid and strict=True
        """
        if not self.strict:
            return  # Skip validation
        
        # Check finite
        if not np.isfinite(x):
            raise ValueError(f"Observation must be finite, got {x}")
        
        # Check positive (x=0 is allowed but gives -∞ density in most cases)
        if x < 0:
            raise ValueError(f"Gamma data must be non-negative, got {x}")
    
    def validate_batch(self, data: np.ndarray) -> np.ndarray:
        """
        Validate and convert batch data to contiguous float64 array.
        
        Args:
            data: Array-like of observations
        
        Returns:
            Validated, contiguous float64 array
        
        Raises:
            ValueError: If data is invalid and strict=True
        """
        # Convert to numpy array if not already
        data = np.asarray(data)
        
        # Validation (if strict)
        if self.strict:
            if not np.all(np.isfinite(data)):
                raise ValueError("All observations must be finite")
            if np.any(data < 0):
                raise ValueError("Gamma data must be non-negative")
        
        return np.ascontiguousarray(data, dtype=np.float64)

