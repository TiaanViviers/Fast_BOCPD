"""
Integration tests for BOCPD with Student-t NG model.

Tests the full pipeline: BOCPD core + Student-t model + hazard function.
All tests use deterministic random seeds for reproducibility.
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, ConstantHazard


class TestBOCPDStudentTBasic:
    """Basic functionality tests with Student-t NG."""
    
    def test_initialization(self, student_t_model, constant_hazard):
        """BOCPD should initialize correctly with Student-t model."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        assert bocpd.max_run_length == 200
        assert bocpd.obs_model == student_t_model
        assert bocpd.hazard == constant_hazard
    
    def test_single_update(self, student_t_model, constant_hazard):
        """Single update should produce valid posterior."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        posterior, cp_prob = bocpd.update(0.5)
        
        assert len(posterior) == 201
        assert 0.0 <= cp_prob <= 1.0
        assert not np.any(np.isnan(posterior))
    
    def test_multiple_updates(self, student_t_model, constant_hazard, test_data_simple):
        """Multiple updates should work without errors."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        for x in test_data_simple:
            posterior, cp_prob = bocpd.update(x)
            assert len(posterior) == 201
            assert 0.0 <= cp_prob <= 1.0
    
    def test_batch_update(self, student_t_model, constant_hazard):
        """Batch update should process all observations."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        np.random.seed(42)
        data = np.random.randn(500)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 500
        assert np.all(cp_probs >= 0.0)
        assert np.all(cp_probs <= 1.0)
    
    def test_reset(self, student_t_model, constant_hazard):
        """Reset should restore initial state."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
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


class TestBOCPDStudentTChangeDetection:
    """Changepoint detection tests with Student-t NG."""
    
    def test_detects_mean_shift(self, student_t_model, constant_hazard):
        """Should detect clear mean shifts."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        # Segment 1: mean=0
        np.random.seed(42)
        prev_map_r = 0
        for _ in range(100):
            bocpd.update(np.random.randn())
            prev_map_r = bocpd.get_map_run_length()
        
        # Segment 2: MASSIVE shift (Student-t is robust)
        changepoints_detected = []
        for t in range(100):
            posterior, cp_prob = bocpd.update(np.random.randn() + 50.0)
            map_r = bocpd.get_map_run_length()
            
            # Detect changepoint: run length reset or sharp drop
            if map_r <= 2 or map_r < prev_map_r / 2 or cp_prob > 0.05:
                changepoints_detected.append(t)
            
            prev_map_r = map_r
        
        # Should detect at least one changepoint
        assert len(changepoints_detected) > 0
    
    def test_robust_to_outliers(self, student_t_model, constant_hazard):
        """Should handle outliers without crashing (better than Gaussian)."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
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


class TestBOCPDStudentTOnlineDetector:
    """OnlineChangeDetector integration tests with Student-t."""
    
    def test_online_detector_basic(self, student_t_model):
        """OnlineChangeDetector should work with Student-t model."""
        from fast_bocpd.utils import OnlineChangeDetector
        
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(student_t_model, hazard, max_run_length=150)
        detector = OnlineChangeDetector(bocpd, min_cp_prob=0.05)  # Lower threshold for robust model
        
        # Segment 1: mean=0 (no changepoints expected, but may get false positives)
        np.random.seed(42)
        false_positives = 0
        for _ in range(50):
            cp = detector.update(np.random.randn())
            if cp is not None:
                false_positives += 1
        
        # Should have very few false positives (Student-t is robust but not perfect)
        assert false_positives < 5, f"Too many false positives: {false_positives}"
        
        # Segment 2: mean=50 (MASSIVE shift)
        detected_cp = False
        for _ in range(70):
            cp = detector.update(np.random.randn() + 50.0)
            if cp is not None:
                detected_cp = True
                assert cp.confidence >= 0.05
                break
        
        assert detected_cp, "Should detect changepoint after strong mean shift"
