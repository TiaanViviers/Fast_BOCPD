"""
Unit tests for Student-t NG observation model.

Tests parameter validation and model instantiation.
"""
import pytest
from fast_bocpd.models import StudentTNG


class TestStudentTNGValidation:
    """Test parameter validation for Student-t NG model."""
    
    def test_valid_parameters(self, student_t_ng_params):
        """Valid parameters should be accepted."""
        model = StudentTNG(**student_t_ng_params)
        assert model.mu0 == 0.0
        assert model.kappa0 == 1.0
        assert model.alpha0 == 1.0
        assert model.beta0 == 1.0
        assert model.nu == 3.0
    
    def test_default_nu(self, student_t_ng_params):
        """nu should default to 3.0."""
        params = {k: v for k, v in student_t_ng_params.items() if k != 'nu'}
        model = StudentTNG(**params)
        assert model.nu == 3.0
    
    def test_invalid_kappa0(self, student_t_ng_params):
        """kappa0 must be positive."""
        with pytest.raises(ValueError, match="kappa0 must be > 0"):
            StudentTNG(**{**student_t_ng_params, 'kappa0': 0.0})
        
        with pytest.raises(ValueError, match="kappa0 must be > 0"):
            StudentTNG(**{**student_t_ng_params, 'kappa0': -1.0})
    
    def test_invalid_alpha0(self, student_t_ng_params):
        """alpha0 must be positive."""
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            StudentTNG(**{**student_t_ng_params, 'alpha0': 0.0})
    
    def test_invalid_beta0(self, student_t_ng_params):
        """beta0 must be positive."""
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            StudentTNG(**{**student_t_ng_params, 'beta0': 0.0})
    
    def test_invalid_nu(self, student_t_ng_params):
        """nu (degrees of freedom) must be positive."""
        with pytest.raises(ValueError, match="nu.*must be > 0"):
            StudentTNG(**{**student_t_ng_params, 'nu': 0.0})
        
        with pytest.raises(ValueError, match="nu.*must be > 0"):
            StudentTNG(**{**student_t_ng_params, 'nu': -1.0})
    
    def test_different_nu_values(self):
        """Various valid nu values should work."""
        # Cauchy (very heavy tails)
        model1 = StudentTNG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0, nu=1.0)
        assert model1.nu == 1.0
        
        # Financial data (moderate tails)
        model2 = StudentTNG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0, nu=5.0)
        assert model2.nu == 5.0
        
        # Nearly Gaussian (light tails)
        model3 = StudentTNG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0, nu=30.0)
        assert model3.nu == 30.0
    
    def test_float_conversion(self):
        """Parameters should be converted to float."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=3)
        assert isinstance(model.mu0, float)
        assert isinstance(model.kappa0, float)
        assert isinstance(model.alpha0, float)
        assert isinstance(model.beta0, float)
        assert isinstance(model.nu, float)
