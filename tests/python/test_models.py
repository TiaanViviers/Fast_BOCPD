"""
Test observation model wrappers and parameter validation
"""
import pytest
from fast_bocpd.models import GaussianNIG


class TestGaussianNIG:
    """Test GaussianNIG parameter wrapper"""
    
    def test_valid_initialization(self):
        """Test initialization with valid parameters"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        
        assert model.mu0 == 0.0
        assert model.kappa0 == 1.0
        assert model.alpha0 == 1.0
        assert model.beta0 == 1.0
    
    def test_type_conversion(self):
        """Test that parameters are converted to float"""
        model = GaussianNIG(mu0=0, kappa0=1, alpha0=1, beta0=1)
        
        assert isinstance(model.mu0, float)
        assert isinstance(model.kappa0, float)
        assert isinstance(model.alpha0, float)
        assert isinstance(model.beta0, float)
    
    def test_invalid_kappa0(self):
        """Test that invalid kappa0 is rejected"""
        with pytest.raises(ValueError, match="kappa0 must be > 0"):
            GaussianNIG(mu0=0.0, kappa0=0.0, alpha0=1.0, beta0=1.0)
        
        with pytest.raises(ValueError, match="kappa0 must be > 0"):
            GaussianNIG(mu0=0.0, kappa0=-1.0, alpha0=1.0, beta0=1.0)
    
    def test_invalid_alpha0(self):
        """Test that invalid alpha0 is rejected"""
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=0.0, beta0=1.0)
        
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=-1.0, beta0=1.0)
    
    def test_invalid_beta0(self):
        """Test that invalid beta0 is rejected"""
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=0.0)
        
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=-1.0)
    
    def test_mu0_can_be_any_value(self):
        """Test that mu0 can be positive, negative, or zero"""
        model1 = GaussianNIG(mu0=10.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        model2 = GaussianNIG(mu0=-10.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        model3 = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        
        assert model1.mu0 == 10.0
        assert model2.mu0 == -10.0
        assert model3.mu0 == 0.0
    
    def test_different_parameter_scales(self):
        """Test with very small and very large parameter values"""
        # Very small positive values
        model_small = GaussianNIG(mu0=0.0, kappa0=1e-10, alpha0=1e-10, beta0=1e-10)
        assert model_small.kappa0 > 0
        
        # Very large values
        model_large = GaussianNIG(mu0=0.0, kappa0=1e10, alpha0=1e10, beta0=1e10)
        assert model_large.kappa0 == 1e10
