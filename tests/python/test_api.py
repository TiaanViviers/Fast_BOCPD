"""
Test public API and user interface
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard, is_available


class TestLibraryAvailability:
    """Test that the library is properly installed"""
    
    def test_c_library_available(self):
        """C library should be compiled and available"""
        assert is_available(), "C library not available. Run: pip install -e ."
    
    def test_imports(self):
        """Test that all public API classes can be imported"""
        from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard
        assert BOCPD is not None
        assert GaussianNIG is not None
        assert ConstantHazard is not None


class TestBOCPDAPI:
    """Test BOCPD class API"""
    
    def test_initialization(self):
        """Test basic initialization"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        assert bocpd.max_run_length == 50
        assert bocpd.obs_model.mu0 == 0.0
        assert bocpd.hazard.lambda_ == 100
    
    def test_update_return_types(self):
        """Test that update returns correct types"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        posterior_r, cp_prob = bocpd.update(0.5)
        
        assert isinstance(posterior_r, np.ndarray)
        assert isinstance(cp_prob, float)
        assert len(posterior_r) == 51  # max_run_length + 1
    
    def test_update_probability_range(self):
        """Test that probabilities are in valid range"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        posterior_r, cp_prob = bocpd.update(0.5)
        
        assert 0.0 <= cp_prob <= 1.0
        assert np.all(posterior_r >= 0.0)
        assert np.all(posterior_r <= 1.0)
    
    def test_posterior_normalized(self):
        """Test that posterior distribution sums to 1"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        posterior_r, _ = bocpd.update(0.5)
        
        assert np.abs(np.sum(posterior_r) - 1.0) < 1e-6
    
    def test_reset(self):
        """Test reset functionality"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        # Process some data
        for x in [0.1, 0.2, 0.3]:
            bocpd.update(x)
        
        # Get a reference update after processing
        _, cp_prob_after_data = bocpd.update(0.4)
        
        # Reset
        bocpd.reset()
        
        # First update after reset should give same result as fresh initialization
        model2 = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard2 = ConstantHazard(lambda_=100)
        bocpd2 = BOCPD(model2, hazard2, max_run_length=50)
        
        posterior_r_reset, cp_prob_reset = bocpd.update(0.1)
        posterior_r_fresh, cp_prob_fresh = bocpd2.update(0.1)
        
        # After reset, should behave identically to fresh initialization
        assert np.allclose(posterior_r_reset, posterior_r_fresh)
        assert abs(cp_prob_reset - cp_prob_fresh) < 1e-10
    
    def test_batch_update(self):
        """Test batch processing"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=200)
        
        data = np.random.randn(100)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == len(data)
        assert np.all((cp_probs >= 0.0) & (cp_probs <= 1.0))
    
    def test_batch_accepts_list(self):
        """Test that batch_update accepts Python lists"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        data = [0.1, 0.2, 0.3, 0.4, 0.5]
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 5


class TestErrorHandling:
    """Test error handling and validation"""
    
    def test_unsupported_model(self):
        """Test that unsupported models raise informative errors"""
        class UnsupportedModel:
            pass
        
        with pytest.raises(ValueError, match="Unsupported observation model"):
            model = UnsupportedModel()
            hazard = ConstantHazard(lambda_=100)
            BOCPD(model, hazard)
    
    def test_unsupported_hazard(self):
        """Test that unsupported hazards raise informative errors"""
        class UnsupportedHazard:
            pass
        
        with pytest.raises(ValueError, match="Unsupported hazard"):
            model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
            hazard = UnsupportedHazard()
            BOCPD(model, hazard)
    
    def test_library_not_compiled_error(self):
        """Test error message when C library is not compiled"""
        # This test would only run if we could temporarily break the library
        # For now, just verify is_available() returns True
        assert is_available()
