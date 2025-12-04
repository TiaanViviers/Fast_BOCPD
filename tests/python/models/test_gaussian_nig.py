"""
Unit tests for Gaussian NIG observation model.

Tests parameter validation and model instantiation.
"""
import pytest
from fast_bocpd.models import GaussianNIG


class TestGaussianNIGValidation:
    """Test parameter validation for Gaussian NIG model."""
    
    def test_valid_parameters(self, gaussian_nig_params):
        """Valid parameters should be accepted."""
        model = GaussianNIG(**gaussian_nig_params)
        assert model.mu0 == 0.0
        assert model.kappa0 == 1.0
        assert model.alpha0 == 1.0
        assert model.beta0 == 1.0
    
    def test_invalid_kappa0(self, gaussian_nig_params):
        """kappa0 must be positive."""
        with pytest.raises(ValueError, match="kappa0 must be > 0"):
            GaussianNIG(**{**gaussian_nig_params, 'kappa0': 0.0})
        
        with pytest.raises(ValueError, match="kappa0 must be > 0"):
            GaussianNIG(**{**gaussian_nig_params, 'kappa0': -1.0})
    
    def test_invalid_alpha0(self, gaussian_nig_params):
        """alpha0 must be positive."""
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            GaussianNIG(**{**gaussian_nig_params, 'alpha0': 0.0})
        
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            GaussianNIG(**{**gaussian_nig_params, 'alpha0': -1.0})
    
    def test_invalid_beta0(self, gaussian_nig_params):
        """beta0 must be positive."""
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            GaussianNIG(**{**gaussian_nig_params, 'beta0': 0.0})
        
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            GaussianNIG(**{**gaussian_nig_params, 'beta0': -1.0})
    
    def test_float_conversion(self):
        """Parameters should be converted to float."""
        model = GaussianNIG(mu0=0, kappa0=1, alpha0=1, beta0=1)
        assert isinstance(model.mu0, float)
        assert isinstance(model.kappa0, float)
        assert isinstance(model.alpha0, float)
        assert isinstance(model.beta0, float)
