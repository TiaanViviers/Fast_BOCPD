"""
Test Python bindings to C library
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fast_bocpd._bindings import (
    is_c_available, _lib,
    GaussianNIGParams, ConstantHazardParams,
    BOCPDState, OBS_MODEL_GAUSSIAN_NIG, HAZARD_CONSTANT
)
import ctypes
import numpy as np

def test_c_bindings():
    print("Testing C bindings...")
    
    if not is_c_available():
        print("❌ C library not available!")
        print("Run 'make lib' in fast_bocpd/_c/ directory first")
        return False
    
    print("✓ C library loaded")
    
    # Test ConstantHazard initialization
    hazard_params = ConstantHazardParams()
    ret = _lib.constant_hazard_init(ctypes.byref(hazard_params), 100.0)
    if ret != 0:
        print("❌ Hazard initialization failed")
        return False
    print(f"✓ Hazard initialized: lambda={hazard_params.lambda_}, log_H={hazard_params.log_H:.6f}")
    
    # Test BOCPD initialization
    obs_params = GaussianNIGParams(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    state = BOCPDState()
    
    ret = _lib.bocpd_init(
        ctypes.byref(state),
        OBS_MODEL_GAUSSIAN_NIG,
        ctypes.byref(obs_params),
        HAZARD_CONSTANT,
        ctypes.byref(hazard_params),
        50
    )
    
    if ret != 0:
        print("❌ BOCPD initialization failed")
        return False
    print(f"✓ BOCPD initialized with max_run_length={state.max_run_length}")
    
    # Test update
    test_data = [0.1, 0.2, 0.15, 5.0, 5.1, 4.9]
    print("\nProcessing test data:")
    
    for x in test_data:
        cp_prob = ctypes.c_double()
        posterior_ptr = _lib.bocpd_update(ctypes.byref(state), x, ctypes.byref(cp_prob))
        
        if not posterior_ptr:
            print(f"❌ Update failed for x={x}")
            _lib.bocpd_free(ctypes.byref(state))
            return False
        
        print(f"  x={x:.2f}: cp_prob={cp_prob.value:.6f}")
    
    print("✓ All updates successful")
    
    # Test batch update
    _lib.bocpd_reset(ctypes.byref(state))
    batch_data = np.array([0.0, 0.1, 0.0, 5.0, 5.0, 5.0], dtype=np.float64)
    cp_probs = np.zeros(len(batch_data), dtype=np.float64)
    
    ret = _lib.bocpd_batch_update(
        ctypes.byref(state),
        batch_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(batch_data),
        cp_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    if ret != 0:
        print("❌ Batch update failed")
        _lib.bocpd_free(ctypes.byref(state))
        return False
    
    print("\nBatch update results:")
    for i, cp_prob in enumerate(cp_probs):
        print(f"  t={i}: cp_prob={cp_prob:.6f}")
    print("✓ Batch update successful")
    
    # Cleanup
    _lib.bocpd_free(ctypes.byref(state))
    print("\n✅ All C binding tests passed!")
    return True

if __name__ == "__main__":
    success = test_c_bindings()
    sys.exit(0 if success else 1)
