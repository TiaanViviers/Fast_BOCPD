"""
Python bindings to the C BOCPD implementation using ctypes.
"""
import os
import ctypes
import numpy as np
from pathlib import Path

# Find the compiled C library
def _load_library():
    """Try to load the compiled C library"""
    
    # Try 1: Extension module built by setuptools (pip install)
    try:
        # This will be at fast_bocpd/_core.*.so (e.g., _core.cpython-39-x86_64-linux-gnu.so)
        import importlib.util
        spec = importlib.util.find_spec("fast_bocpd._core")
        if spec and spec.origin:
            return ctypes.CDLL(spec.origin)
    except (ImportError, OSError):
        pass
    
    # Try 2: Manually built shared library (development mode)
    _LIB_DIR = Path(__file__).parent / "_c"
    _LIB_NAME = "libbocpd"
    for ext in ['.so', '.dylib', '.dll']:
        lib_path = _LIB_DIR / f"{_LIB_NAME}{ext}"
        if lib_path.exists():
            try:
                return ctypes.CDLL(str(lib_path))
            except OSError:
                pass
    
    # If not found, return None
    return None

_lib = _load_library()


# ============================================================================
# C Structure Definitions (must match C exactly)
# ============================================================================

class GaussianNIGParams(ctypes.Structure):
    """Matches GaussianNIGParams in C"""
    _fields_ = [
        ("mu0", ctypes.c_double),
        ("kappa0", ctypes.c_double),
        ("alpha0", ctypes.c_double),
        ("beta0", ctypes.c_double),
    ]


class GaussianNIGStats(ctypes.Structure):
    """Matches GaussianNIGStats in C"""
    _fields_ = [
        ("n", ctypes.c_int32),
        ("sum_x", ctypes.c_double),
        ("sum_x2", ctypes.c_double),
    ]


class ConstantHazardParams(ctypes.Structure):
    """Matches ConstantHazardParams in C"""
    _fields_ = [
        ("lambda_", ctypes.c_double),
        ("log_H", ctypes.c_double),
        ("log_1mH", ctypes.c_double),
    ]


class ObsModelParams(ctypes.Union):
    """Matches ObsModelParams union in C"""
    _fields_ = [
        ("gaussian_nig", GaussianNIGParams),
    ]


class ObsModelStats(ctypes.Union):
    """Matches ObsModelStats union in C"""
    _fields_ = [
        ("gaussian_nig", GaussianNIGStats),
    ]


class HazardParams(ctypes.Union):
    """Matches HazardParams union in C"""
    _fields_ = [
        ("constant", ConstantHazardParams),
    ]


class BOCPDState(ctypes.Structure):
    """Matches BOCPDState in C"""
    _fields_ = [
        ("max_run_length", ctypes.c_int32),
        ("obs_model_type", ctypes.c_int),  # enum
        ("hazard_type", ctypes.c_int),     # enum
        ("obs_params", ObsModelParams),
        ("hazard_params", HazardParams),
        ("log_joint", ctypes.POINTER(ctypes.c_double)),
        ("stats", ctypes.POINTER(ObsModelStats)),
        ("new_log_joint", ctypes.POINTER(ctypes.c_double)),
        ("new_stats", ctypes.POINTER(ObsModelStats)),
        ("posterior_r", ctypes.POINTER(ctypes.c_double)),
    ]


# Enums (must match C)
OBS_MODEL_GAUSSIAN_NIG = 0
HAZARD_CONSTANT = 0


# ============================================================================
# C Function Declarations
# ============================================================================

if _lib is not None:
    # constant_hazard_init
    _lib.constant_hazard_init.argtypes = [
        ctypes.POINTER(ConstantHazardParams),
        ctypes.c_double
    ]
    _lib.constant_hazard_init.restype = ctypes.c_int

    # bocpd_init
    _lib.bocpd_init.argtypes = [
        ctypes.POINTER(BOCPDState),
        ctypes.c_int,  # obs_model_type
        ctypes.c_void_p,  # obs_params
        ctypes.c_int,  # hazard_type
        ctypes.c_void_p,  # hazard_params
        ctypes.c_int32,  # max_run_length
    ]
    _lib.bocpd_init.restype = ctypes.c_int

    # bocpd_free
    _lib.bocpd_free.argtypes = [ctypes.POINTER(BOCPDState)]
    _lib.bocpd_free.restype = None

    # bocpd_reset
    _lib.bocpd_reset.argtypes = [ctypes.POINTER(BOCPDState)]
    _lib.bocpd_reset.restype = None

    # bocpd_update
    _lib.bocpd_update.argtypes = [
        ctypes.POINTER(BOCPDState),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double)
    ]
    _lib.bocpd_update.restype = ctypes.POINTER(ctypes.c_double)

    # bocpd_batch_update
    _lib.bocpd_batch_update.argtypes = [
        ctypes.POINTER(BOCPDState),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double)
    ]
    _lib.bocpd_batch_update.restype = ctypes.c_int


# ============================================================================
# Python API - will be exported
# ============================================================================

def is_c_available():
    """Check if C library is available"""
    return _lib is not None
