"""
Test OnlineChangeDetector utility
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard, OnlineChangeDetector, Changepoint


class TestOnlineChangeDetector:
    """Test OnlineChangeDetector wrapper"""
    
    def test_initialization(self):
        """Test basic initialization"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        detector = OnlineChangeDetector(bocpd, min_confidence=0.3)
        
        assert detector.bocpd is bocpd
        assert detector.min_confidence == 0.3
        assert detector.get_current_run_length() == 0
    
    def test_no_changepoint_detection(self):
        """Test with constant signal (no changepoints)"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        detector = OnlineChangeDetector(bocpd, min_confidence=0.3)
        
        # Feed constant data
        data = np.random.randn(50) * 0.5 + 5.0
        
        changepoints = []
        for x in data:
            cp = detector.update(x)
            if cp:
                changepoints.append(cp)
        
        # Should have very few (if any) changepoints
        assert len(changepoints) < 3
    
    def test_changepoint_detection(self):
        """Test detection of actual changepoint"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(model, hazard, max_run_length=100)
        detector = OnlineChangeDetector(bocpd, min_confidence=0.3)
        
        # Generate data with changepoint at t=25
        data = np.concatenate([
            np.random.randn(25) * 0.5 + 0.0,
            np.random.randn(25) * 0.5 + 5.0
        ])
        
        for t, x in enumerate(data):
            cp = detector.update(x, metadata=f"t={t}")
        
        changepoints = detector.get_changepoints()
        
        # Should detect at least one changepoint
        assert len(changepoints) > 0
        
        # Should detect changepoint near t=25
        cp_indices = [cp.index for cp in changepoints]
        assert any(20 <= idx <= 30 for idx in cp_indices)
    
    def test_changepoint_properties(self):
        """Test Changepoint dataclass properties"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(model, hazard, max_run_length=100)
        detector = OnlineChangeDetector(bocpd, min_confidence=0.3)
        
        data = np.concatenate([
            np.random.randn(25) * 0.5 + 0.0,
            np.random.randn(25) * 0.5 + 5.0
        ])
        
        for t, x in enumerate(data):
            cp = detector.update(x, metadata={"time": t, "label": "test"})
            
            if cp:
                # Check all properties exist and are valid
                assert isinstance(cp.index, int)
                assert cp.index >= 0
                assert cp.prev_run_length >= 0
                assert 0.0 <= cp.confidence <= 1.0
                assert isinstance(cp.observation, float)
                assert cp.metadata is not None
                
                # Check string representation
                cp_str = str(cp)
                assert "Changepoint" in cp_str
                assert str(cp.index) in cp_str
    
    def test_current_run_length(self):
        """Test get_current_run_length tracking"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        detector = OnlineChangeDetector(bocpd)
        
        run_lengths = []
        for i in range(10):
            detector.update(0.1)
            run_lengths.append(detector.get_current_run_length())
        
        # Run length should generally increase
        assert run_lengths[-1] >= run_lengths[0]
    
    def test_map_history(self):
        """Test get_map_history tracking"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        detector = OnlineChangeDetector(bocpd)
        
        for i in range(10):
            detector.update(0.1 * i)
        
        history = detector.get_map_history()
        
        assert len(history) == 10
        assert isinstance(history, np.ndarray)
        assert np.all(history >= 0)
    
    def test_segments(self):
        """Test get_segments"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(model, hazard, max_run_length=100)
        detector = OnlineChangeDetector(bocpd, min_confidence=0.3)
        
        # Generate data with 2 changepoints
        data = np.concatenate([
            np.random.randn(25) * 0.5 + 0.0,
            np.random.randn(25) * 0.5 + 5.0,
            np.random.randn(25) * 0.5 + -3.0
        ])
        
        for x in data:
            detector.update(x)
        
        segments = detector.get_segments()
        
        # Should have segments
        assert len(segments) > 0
        
        # Each segment should be (start, end) tuple
        for start, end in segments:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert start < end
            assert start >= 0
            assert end <= len(data)
        
        # Segments should cover entire data range
        assert segments[0][0] == 0
        assert segments[-1][1] == len(data)
    
    def test_reset(self):
        """Test reset functionality"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        detector = OnlineChangeDetector(bocpd)
        
        # Process some data
        for i in range(10):
            detector.update(0.1 * i)
        
        assert len(detector.get_map_history()) == 10
        assert detector.get_current_run_length() > 0
        
        # Reset
        detector.reset()
        
        # Should be back to initial state
        assert len(detector.get_map_history()) == 0
        assert len(detector.get_changepoints()) == 0
        assert detector.get_current_run_length() == 0
    
    def test_confidence_filtering(self):
        """Test min_confidence filtering"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(model, hazard, max_run_length=100)
        
        # Low confidence threshold
        detector_low = OnlineChangeDetector(bocpd, min_confidence=0.1)
        
        # High confidence threshold
        model2 = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard2 = ConstantHazard(lambda_=50)
        bocpd2 = BOCPD(model2, hazard2, max_run_length=100)
        detector_high = OnlineChangeDetector(bocpd2, min_confidence=0.8)
        
        data = np.concatenate([
            np.random.randn(25) * 0.5 + 0.0,
            np.random.randn(25) * 0.5 + 3.0
        ])
        
        for x in data:
            detector_low.update(x)
            detector_high.update(x)
        
        # Low threshold should detect more changepoints
        assert len(detector_low.get_changepoints()) >= len(detector_high.get_changepoints())
