"""
Integration tests for BOCPD with Gaussian NIG model.

Tests the full pipeline: BOCPD core + Gaussian model + hazard function.
All tests use deterministic random seeds for reproducibility.
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, ConstantHazard


class TestBOCPDGaussianBasic:
    """Basic functionality tests with Gaussian NIG."""
    
    def test_initialization(self, gaussian_model, constant_hazard):
        """BOCPD should initialize correctly with Gaussian model."""
        bocpd = BOCPD(gaussian_model, constant_hazard, max_run_length=200)
        assert bocpd.max_run_length == 200
        assert bocpd.obs_model == gaussian_model
        assert bocpd.hazard == constant_hazard
    
    def test_single_update(self, gaussian_model, constant_hazard):
        """Single update should produce valid posterior."""
        bocpd = BOCPD(gaussian_model, constant_hazard, max_run_length=200)
        
        posterior, cp_prob = bocpd.update(0.5)
        
        assert len(posterior) == 201
        assert 0.0 <= cp_prob <= 1.0
        assert not np.any(np.isnan(posterior))
    
    def test_multiple_updates(self, gaussian_model, constant_hazard, test_data_simple):
        """Multiple updates should work without errors."""
        bocpd = BOCPD(gaussian_model, constant_hazard, max_run_length=200)
        
        for x in test_data_simple:
            posterior, cp_prob = bocpd.update(x)
            assert len(posterior) == 201
            assert 0.0 <= cp_prob <= 1.0
    
    def test_batch_update(self, gaussian_model, constant_hazard):
        """Batch update should process all observations."""
        bocpd = BOCPD(gaussian_model, constant_hazard, max_run_length=200)
        
        np.random.seed(42)
        data = np.random.randn(500)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 500
        assert np.all(cp_probs >= 0.0)
        assert np.all(cp_probs <= 1.0)
    
    def test_reset(self, gaussian_model, constant_hazard):
        """Reset should restore initial state."""
        bocpd = BOCPD(gaussian_model, constant_hazard, max_run_length=200)
        
        # Process some data
        np.random.seed(42)
        for _ in range(50):
            bocpd.update(np.random.randn())
        
        # Reset
        bocpd.reset()
        bocpd.update(0.0)
        posterior = bocpd.get_posterior()
        
        # After reset + 1 update, most mass should be in r=0 and r=1
        assert posterior[0] + posterior[1] > 0.95


class TestBOCPDGaussianChangeDetection:
    """Changepoint detection tests with Gaussian NIG."""
    
    def test_detects_mean_shift(self, gaussian_model, constant_hazard):
        """Should detect clear mean shifts."""
        bocpd = BOCPD(gaussian_model, constant_hazard, max_run_length=200)
        
        # Segment 1: mean=0
        np.random.seed(42)
        for _ in range(100):
            bocpd.update(np.random.randn())
        
        # Segment 2: mean=5 (strong shift)
        changepoints_detected = []
        for t in range(100):
            posterior, cp_prob = bocpd.update(np.random.randn() + 5.0)
            map_r = bocpd.get_map_run_length()
            if map_r == 0:
                changepoints_detected.append(t)
        
        # Should detect at least one changepoint
        assert len(changepoints_detected) > 0
    
    def test_robust_to_outliers(self, gaussian_model, constant_hazard):
        """Should handle outliers without crashing."""
        bocpd = BOCPD(gaussian_model, constant_hazard, max_run_length=200)
        
        # Normal data with occasional outliers
        np.random.seed(42)
        for t in range(100):
            if t % 20 == 0:
                x = np.random.randn() * 10  # Outlier
            else:
                x = np.random.randn()  # Normal
            
            posterior, cp_prob = bocpd.update(x)
            
            # Should not produce NaN
            assert not np.any(np.isnan(posterior))
            assert not np.isnan(cp_prob)


class TestBOCPDGaussianOnlineDetector:
    """OnlineChangeDetector integration tests with Gaussian."""
    
    def test_online_detector_basic(self, gaussian_model):
        """OnlineChangeDetector should work with Gaussian model."""
        from fast_bocpd.utils import OnlineChangeDetector
        
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(gaussian_model, hazard, max_run_length=150)
        detector = OnlineChangeDetector(bocpd, min_cp_prob=0.3)
        
        # Segment 1: mean=0 (no changepoints)
        np.random.seed(42)
        for _ in range(50):
            cp = detector.update(np.random.randn())
            assert cp is None
        
        # Segment 2: mean=5 (should detect)
        detected_cp = False
        for _ in range(70):
            cp = detector.update(np.random.randn() + 5.0)
            if cp is not None:
                detected_cp = True
                assert cp.confidence >= 0.3
                break
        
        assert detected_cp, "Should detect changepoint after mean shift"
