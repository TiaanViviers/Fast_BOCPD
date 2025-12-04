"""
Python API for C-based BOCPD implementation.
"""
import numpy as np
import ctypes
from typing import Tuple

from . import _bindings
from .hazard import ConstantHazard
from .models import GaussianNIG, StudentTNG


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
        self.max_run_length = max_run_length
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
            obs_model_type = _bindings.OBS_MODEL_STUDENT_T_NG
            obs_params = _bindings.StudentTNGParams(
                mu0=self.obs_model.mu0,
                kappa0=self.obs_model.kappa0,
                alpha0=self.obs_model.alpha0,
                beta0=self.obs_model.beta0,
                nu=self.obs_model.nu
            )
        else:
            raise ValueError(f"Unsupported observation model: {type(self.obs_model)}")
        
        hazard_params = _bindings.ConstantHazardParams()
        ret = _bindings._lib.constant_hazard_init(
            ctypes.byref(hazard_params),
            self.hazard.lambda_
        )
        if ret != 0:
            raise RuntimeError("Failed to initialize hazard function")
        
        # Initialize BOCPD state
        self._state = _bindings.BOCPDState()
        ret = _bindings._lib.bocpd_init(
            ctypes.byref(self._state),
            obs_model_type,
            ctypes.byref(obs_params),
            _bindings.HAZARD_CONSTANT,
            ctypes.byref(hazard_params),
            self.max_run_length
        )
        
        if ret != 0:
            raise RuntimeError("Failed to initialize BOCPD")
    
    def reset(self) -> None:
        """Reset to prior (as if no data has been seen)."""
        _bindings._lib.bocpd_reset(ctypes.byref(self._state))
    
    def update(self, x: float) -> Tuple[np.ndarray, float]:
        """
        Process one new observation (online mode).
        
        Args:
            x: New observation
            
        Returns:
            posterior_r: Array of P(r_t = r | x_1:t)
            cp_prob: Probability of changepoint (posterior_r[0])
        """
        cp_prob = ctypes.c_double()
        posterior_ptr = _bindings._lib.bocpd_update(
            ctypes.byref(self._state),
            float(x),
            ctypes.byref(cp_prob)
        )
        
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
        data = np.asarray(data, dtype=np.float64)
        cp_probs = np.zeros(len(data), dtype=np.float64)
        
        ret = _bindings._lib.bocpd_batch_update(
            ctypes.byref(self._state),
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
        r_map = _bindings._lib.bocpd_get_map_run_length(ctypes.byref(self._state))
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
        ret = _bindings._lib.bocpd_get_posterior(
            ctypes.byref(self._state),
            posterior.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        if ret != 0:
            raise RuntimeError("Failed to get posterior")
        
        return posterior
    
    def __del__(self):
        """Cleanup C resources."""
        if self._state is not None:
            try:
                _bindings._lib.bocpd_free(ctypes.byref(self._state))
            except:
                pass


def is_available() -> bool:
    """Check if C library is available."""
    return _bindings.is_c_available()

