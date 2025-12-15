"""
Python API for C-based BOCPD implementation.
"""
import numpy as np
import ctypes
from typing import Tuple

from . import _bindings
from .hazard import ConstantHazard
from .models import (
    GaussianNIG,
    StudentTNG,
    PoissonGamma,
    BernoulliBeta,
    BinomialBeta,
    GammaGamma,
)


class BOCPD:
    """
    Bayesian Online Changepoint Detection.
    
    This is a Python wrapper around the C implementation for performance.
    """
    
    def __init__(self, obs_model, hazard, max_run_length: int = 200):
        """
        Initialize BOCPD detector.
        
        Args:
            obs_model: Observation model (e.g., GaussianNIG instance)
            hazard: Hazard function (e.g., ConstantHazard instance)
            max_run_length: Maximum run length to track
        """
        if not _bindings.is_c_available():
            raise RuntimeError(
                "C library not available. Please compile the C library first:\n"
                "  cd fast_bocpd/_c\n"
                "  make lib"
            )
        
        self.obs_model = obs_model
        self.hazard = hazard
        self.max_run_length = int(max_run_length)
        if self.max_run_length <= 0:
            raise ValueError("max_run_length must be > 0")
        self._state = None
        
        self._init_c_backend()
    
    def _init_c_backend(self) -> None:
        """Initialize C backend."""
        if not isinstance(self.hazard, ConstantHazard):
            raise ValueError(f"Unsupported hazard function: {type(self.hazard)}")
        
        # Determine observation model type and create params
        if isinstance(self.obs_model, GaussianNIG):
            obs_model_type = _bindings.OBS_MODEL_GAUSSIAN_NIG
            obs_params = _bindings.GaussianNIGParams(
                mu0=self.obs_model.mu0,
                kappa0=self.obs_model.kappa0,
                alpha0=self.obs_model.alpha0,
                beta0=self.obs_model.beta0
            )
        elif isinstance(self.obs_model, StudentTNG):
            if self.obs_model.is_grid:
                # Grid Student-t mode
                obs_model_type = _bindings.OBS_MODEL_STUDENT_T_NG_GRID
                
                # Ensure contiguous arrays and keep them alive (C will deep-copy but needs contiguous input)
                self._nu_grid_arr = np.ascontiguousarray(self.obs_model.nu_grid, dtype=np.float64)
                self._nu_prior_arr = np.ascontiguousarray(self.obs_model.nu_prior, dtype=np.float64)
                
                obs_params = _bindings.StudentTNGGridParams(
                    mu0=self.obs_model.mu0,
                    kappa0=self.obs_model.kappa0,
                    alpha0=self.obs_model.alpha0,
                    beta0=self.obs_model.beta0,
                    K=self.obs_model.K,
                    nu_grid=self._nu_grid_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    nu_prior=self._nu_prior_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                )
            else:
                # Fixed ν mode
                obs_model_type = _bindings.OBS_MODEL_STUDENT_T_NG
                obs_params = _bindings.StudentTNGParams(
                    mu0=self.obs_model.mu0,
                    kappa0=self.obs_model.kappa0,
                    alpha0=self.obs_model.alpha0,
                    beta0=self.obs_model.beta0,
                    nu=self.obs_model.nu
                )
        elif isinstance(self.obs_model, PoissonGamma):
            obs_model_type = _bindings.OBS_MODEL_POISSON_GAMMA
            obs_params = _bindings.PoissonGammaParams(
                alpha0=self.obs_model.alpha0,
                beta0=self.obs_model.beta0
            )
        elif isinstance(self.obs_model, BernoulliBeta):
            obs_model_type = _bindings.OBS_MODEL_BERNOULLI_BETA
            obs_params = _bindings.BernoulliBetaParams(
                alpha0=self.obs_model.alpha0,
                beta0=self.obs_model.beta0
            )
        elif isinstance(self.obs_model, BinomialBeta):
            obs_model_type = _bindings.OBS_MODEL_BINOMIAL_BETA
            obs_params = _bindings.BinomialBetaParams(
                alpha0=self.obs_model.alpha0,
                beta0=self.obs_model.beta0,
                N=self.obs_model.n_trials,
                log_N_factorial=0.0  # Will be set by C in bocpd_init
            )
        elif isinstance(self.obs_model, GammaGamma):
            obs_model_type = _bindings.OBS_MODEL_GAMMA_GAMMA
            obs_params = _bindings.GammaGammaParams(
                alpha0=self.obs_model.alpha0,
                beta0=self.obs_model.beta0,
                shape=self.obs_model.shape,
                log_gamma_k=0.0  # Will be set by C in bocpd_init
            )
        else:
            raise ValueError(f"Unsupported observation model: {type(self.obs_model)}")
        
        hazard_params = _bindings.ConstantHazardParams()
        ret = _bindings.constant_hazard_init(hazard_params, self.hazard.lambda_)
        if ret != 0:
            raise RuntimeError("Failed to initialize hazard function")

        # Initialize BOCPD state
        self._state = _bindings.BOCPDState()
        
        # Cast params to void* for C function (safer than implicit conversion)
        obs_params_ptr = ctypes.cast(ctypes.byref(obs_params), ctypes.c_void_p)
        haz_params_ptr = ctypes.cast(ctypes.byref(hazard_params), ctypes.c_void_p)
        
        ret = _bindings.bocpd_init(
            self._state,
            obs_model_type,
            obs_params_ptr,
            _bindings.HAZARD_CONSTANT,
            haz_params_ptr,
            self.max_run_length
        )
        
        if ret != 0:
            # Free any partially allocated resources
            self.close()
            raise RuntimeError("Failed to initialize BOCPD")
    
    def _require_state(self) -> _bindings.BOCPDState:
        """Return the live state or raise if closed."""
        if self._state is None:
            raise RuntimeError("BOCPD state is not initialized or has been closed")
        return self._state

    def reset(self) -> None:
        """Reset to prior (as if no data has been seen)."""
        _bindings.bocpd_reset(self._require_state())
    
    def update(self, x: float) -> Tuple[np.ndarray, float]:
        """
        Process one new observation (online mode).
        
        Args:
            x: New observation
            
        Returns:
            posterior_r: Array of P(r_t = r | x_1:t)
            cp_prob: Probability of changepoint (posterior_r[0])
        """
        # Validate data for discrete models (if applicable)
        if isinstance(self.obs_model, (PoissonGamma, BernoulliBeta, BinomialBeta)):
            self.obs_model.validate_data(x)
        
        state = self._require_state()
        cp_prob = ctypes.c_double()
        posterior_ptr = _bindings.bocpd_update(state, float(x), cp_prob)
        
        if not posterior_ptr:
            raise RuntimeError("BOCPD update failed")
        
        # Copy posterior to numpy array
        posterior_r = np.ctypeslib.as_array(
            posterior_ptr,
            shape=(self.max_run_length + 1,)
        ).copy()
        
        return posterior_r, cp_prob.value
    
    def batch_update(self, data: np.ndarray) -> np.ndarray:
        """
        Process multiple observations at once (offline mode).
        
        Args:
            data: Array of observations
            
        Returns:
            cp_probs: Array of changepoint probabilities for each time step
        """
        # Validate and convert data for discrete models (if applicable)
        if isinstance(self.obs_model, (PoissonGamma, BernoulliBeta, BinomialBeta)):
            data = self.obs_model.validate_batch(data)
        else:
            data = np.ascontiguousarray(data, dtype=np.float64)
        
        cp_probs = np.zeros(len(data), dtype=np.float64)
        
        state = self._require_state()
        ret = _bindings.bocpd_batch_update(
            state,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(data),
            cp_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        if ret != 0:
            raise RuntimeError("Batch update failed")
        
        return cp_probs
    
    def get_map_run_length(self) -> int:
        """
        Get the most likely (MAP) run length at current time.
        
        Returns:
            Most likely run length (0 means changepoint just occurred)
            
        Example:
            >>> map_r = bocpd.get_map_run_length()
            >>> if map_r == 0:
            ...     print("Changepoint detected!")
            >>> else:
            ...     print(f"Current regime is {map_r} observations old")
        """
        r_map = _bindings.bocpd_get_map_run_length(self._require_state())
        if r_map < 0:
            raise RuntimeError("Failed to get MAP run length")
        return r_map
    
    def get_map_confidence(self) -> float:
        """
        Get confidence in the MAP run length estimate.
        
        Returns:
            Probability mass at MAP run length (higher = more confident)
            
        Example:
            >>> map_r = bocpd.get_map_run_length()
            >>> confidence = bocpd.get_map_confidence()
            >>> print(f"MAP estimate: r={map_r} (confidence: {confidence:.1%})")
        """
        posterior = self.get_posterior()
        map_r = self.get_map_run_length()
        return posterior[map_r]
    
    def get_posterior(self) -> np.ndarray:
        """
        Get current posterior distribution over run lengths.
        
        Returns:
            posterior_r: Array of P(r_t = r | data) for r in [0, max_run_length]
        """
        posterior = np.zeros(self.max_run_length + 1, dtype=np.float64)
        ret = _bindings.bocpd_get_posterior(
            self._require_state(),
            posterior.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        if ret != 0:
            raise RuntimeError("Failed to get posterior")
        
        return posterior
    
    def close(self):
        """Explicitly free C resources (recommended for deterministic cleanup)."""
        if getattr(self, "_state", None) is not None:
            try:
                _bindings.bocpd_free(self._state)
            finally:
                self._state = None
            self._state = None
        
        # Clean up grid arrays if present
        if hasattr(self, '_nu_grid_arr'):
            del self._nu_grid_arr
        if hasattr(self, '_nu_prior_arr'):
            del self._nu_prior_arr
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def __del__(self):
        """Cleanup C resources (fallback if close() not called)."""
        try:
            self.close()
        except Exception:
            pass


def is_available() -> bool:
    """Check if C library is available."""
    return _bindings.is_c_available()
