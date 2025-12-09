"""
Unit tests for GammaGamma observation model.
"""
import pytest
import numpy as np
from fast_bocpd.models import GammaGamma


class TestGammaGammaInit:
    """Test initialization and parameter validation."""
    
    def test_basic_initialization(self):
        """Should initialize with valid parameters."""
        model = GammaGamma(alpha0=2.0, beta0=3.0, shape=1.5)
        assert model.alpha0 == 2.0
        assert model.beta0 == 3.0
        assert model.shape == 1.5
        assert model.strict is True
    
    def test_default_shape(self):
        """Should default to Exponential (shape=1.0)."""
        model = GammaGamma(alpha0=2.0, beta0=3.0)
        assert model.shape == 1.0
    
    def test_accepts_int_and_numpy_scalars(self):
        """Should accept int and numpy scalar types."""
        model = GammaGamma(
            alpha0=np.int64(3), 
            beta0=np.float64(4.5), 
            shape=np.float32(2.0)
        )
        assert isinstance(model.alpha0, float)
        assert isinstance(model.beta0, float)
        assert isinstance(model.shape, float)
        assert model.alpha0 == 3.0
        assert model.beta0 == 4.5
        assert model.shape == 2.0
    
    def test_strict_parameter(self):
        """Should respect strict parameter."""
        model_strict = GammaGamma(alpha0=2.0, beta0=3.0, shape=1.5, strict=True)
        assert model_strict.strict is True
        
        model_non_strict = GammaGamma(alpha0=2.0, beta0=3.0, shape=0.7, strict=False)
        assert model_non_strict.strict is False
    
    @pytest.mark.parametrize("alpha0,beta0,shape,msg", [
        (0.0, 1.0, 1.5, "alpha0 must be > 0"),
        (-1.0, 1.0, 1.5, "alpha0 must be > 0"),
        (1.0, 0.0, 1.5, "beta0 must be > 0"),
        (1.0, -2.0, 1.5, "beta0 must be > 0"),
        (1.0, 1.0, 0.0, "shape must be > 0"),
        (1.0, 1.0, -1.0, "shape must be > 0"),
        (np.inf, 1.0, 1.5, "alpha0 must be finite"),
        (np.nan, 1.0, 1.5, "alpha0 must be finite"),
        (1.0, np.inf, 1.5, "beta0 must be finite"),
        (1.0, np.nan, 1.5, "beta0 must be finite"),
        (1.0, 1.0, np.inf, "shape must be finite"),
        (1.0, 1.0, np.nan, "shape must be finite"),
    ])
    def test_parameter_validation(self, alpha0, beta0, shape, msg):
        """Should reject invalid parameters."""
        with pytest.raises(ValueError, match=msg):
            GammaGamma(alpha0=alpha0, beta0=beta0, shape=shape)
    
    def test_strict_mode_rejects_small_shape(self):
        """Should reject shape < 1 in strict mode."""
        with pytest.raises(ValueError, match="shape must be >= 1.0 in strict mode"):
            GammaGamma(alpha0=1.0, beta0=1.0, shape=0.7, strict=True)
    
    def test_non_strict_allows_small_shape(self):
        """Should allow shape < 1 in non-strict mode."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=0.7, strict=False)
        assert model.shape == 0.7
    
    def test_type_errors(self):
        """Should reject non-numeric types."""
        with pytest.raises(TypeError, match="alpha0 must be numeric"):
            GammaGamma(alpha0="1", beta0=1.0, shape=1.5)
        
        with pytest.raises(TypeError, match="beta0 must be numeric"):
            GammaGamma(alpha0=1.0, beta0="1", shape=1.5)
        
        with pytest.raises(TypeError, match="shape must be numeric"):
            GammaGamma(alpha0=1.0, beta0=1.0, shape="1.5")


class TestGammaGammaValidateData:
    """Test single-value data validation."""
    
    def test_accepts_valid_positive_values(self):
        """Should accept valid positive values."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        # Should not raise
        model.validate_data(1.0)
        model.validate_data(0.001)
        model.validate_data(100.0)
        model.validate_data(1e6)
    
    def test_accepts_zero(self):
        """Should accept zero (though may have -inf density for some shapes)."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        # Should not raise (validation accepts x=0, predictive handles edge case)
        model.validate_data(0.0)
    
    def test_rejects_negative_strict(self):
        """Should reject negative values in strict mode."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            model.validate_data(-1.0)
        
        with pytest.raises(ValueError, match="must be non-negative"):
            model.validate_data(-0.001)
    
    def test_rejects_nan_strict(self):
        """Should reject NaN in strict mode."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        with pytest.raises(ValueError, match="must be finite"):
            model.validate_data(np.nan)
    
    def test_rejects_inf_strict(self):
        """Should reject infinity in strict mode."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        with pytest.raises(ValueError, match="must be finite"):
            model.validate_data(np.inf)
    
    def test_non_strict_skips_validation(self):
        """Non-strict mode should skip validation."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=False)
        
        # Should not raise (even for invalid values)
        model.validate_data(-1.0)
        model.validate_data(np.nan)
        model.validate_data(np.inf)


class TestGammaGammaValidateBatch:
    """Test batch data validation."""
    
    def test_accepts_valid_array(self):
        """Should accept and convert valid arrays."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        data = np.array([1.0, 2.5, 0.1, 100.0])
        validated = model.validate_batch(data)
        
        assert validated.dtype == np.float64
        assert np.array_equal(validated, data)
    
    def test_accepts_list(self):
        """Should convert lists to arrays."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        data = [1.0, 2.5, 0.1, 100.0]
        validated = model.validate_batch(data)
        
        assert isinstance(validated, np.ndarray)
        assert validated.dtype == np.float64
        assert np.array_equal(validated, np.array(data))
    
    def test_accepts_integers(self):
        """Should accept integer arrays and convert to float."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        data = np.array([1, 2, 3, 5, 10], dtype=np.int32)
        validated = model.validate_batch(data)
        
        assert validated.dtype == np.float64
        assert np.array_equal(validated, np.array([1.0, 2.0, 3.0, 5.0, 10.0]))
    
    def test_contiguous_output(self):
        """Should ensure output is contiguous."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        # Create non-contiguous array
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])[:, 0]
        assert not data.flags['C_CONTIGUOUS']
        
        validated = model.validate_batch(data)
        assert validated.flags['C_CONTIGUOUS']
    
    def test_rejects_negative_values_strict(self):
        """Should reject negative values in strict mode."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        data = np.array([1.0, 2.0, -0.5, 4.0])
        with pytest.raises(ValueError, match="must be non-negative"):
            model.validate_batch(data)
    
    def test_rejects_nan_strict(self):
        """Should reject NaN in strict mode."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        data = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="must be finite"):
            model.validate_batch(data)
    
    def test_rejects_inf_strict(self):
        """Should reject infinity in strict mode."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=True)
        
        data = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValueError, match="must be finite"):
            model.validate_batch(data)
    
    def test_non_strict_allows_invalid(self):
        """Non-strict mode should skip validation."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.5, strict=False)
        
        data = np.array([1.0, -1.0, np.nan, np.inf])
        validated = model.validate_batch(data)
        
        # Should convert without error
        assert validated.dtype == np.float64


class TestGammaGammaSpecialCases:
    """Test special cases and edge behaviors."""
    
    def test_exponential_is_shape_one(self):
        """Exponential distribution is Gamma with shape=1."""
        model = GammaGamma(alpha0=2.0, beta0=3.0, shape=1.0)
        assert model.shape == 1.0
    
    def test_default_is_exponential(self):
        """Default shape should give Exponential."""
        model = GammaGamma(alpha0=2.0, beta0=3.0)
        assert model.shape == 1.0
    
    def test_shape_exactly_one(self):
        """Shape exactly 1.0 should be allowed."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.0, strict=True)
        assert model.shape == 1.0
    
    def test_very_large_shape(self):
        """Should handle very large shape values."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=100.0, strict=True)
        assert model.shape == 100.0
    
    def test_shape_near_one(self):
        """Should handle shape values very close to 1."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.0 + 1e-10, strict=True)
        assert model.shape == 1.0 + 1e-10
        
        # Shape slightly below 1 requires strict=False
        model2 = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.0 - 1e-10, strict=False)
        assert model2.shape == 1.0 - 1e-10


class TestGammaGammaDocumentation:
    """Test documentation examples work."""
    
    def test_docstring_example_exponential(self):
        """Docstring example for Exponential waiting times."""
        model = GammaGamma(alpha0=1.0, beta0=1.0)  # shape=1.0
        assert model.shape == 1.0
    
    def test_docstring_example_gamma(self):
        """Docstring example for Gamma with shape=2."""
        model = GammaGamma(alpha0=10.0, beta0=5.0, shape=2.0)
        assert model.alpha0 == 10.0
        assert model.beta0 == 5.0
        assert model.shape == 2.0
    
    def test_docstring_example_non_strict(self):
        """Docstring example for non-strict mode."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=0.5, strict=False)
        assert model.shape == 0.5
        assert model.strict is False


class TestGammaGammaRateParameterization:
    """Test that users understand rate vs scale parameterization."""
    
    def test_prior_mean_rate(self):
        """Prior mean of rate should be alpha0/beta0."""
        # For Gamma(alpha, beta) in rate parameterization: E[λ] = alpha/beta
        model = GammaGamma(alpha0=10.0, beta0=5.0, shape=2.0)
        
        # Expected prior rate: 10/5 = 2.0
        # So expected mean of data (Gamma likelihood): shape/rate = 2.0/2.0 = 1.0
        # This is just for documentation - the model uses rate parameterization
        expected_prior_rate = model.alpha0 / model.beta0
        assert expected_prior_rate == 2.0
    
    def test_weak_prior(self):
        """Weak prior with small alpha0, beta0."""
        model = GammaGamma(alpha0=0.1, beta0=0.1, shape=1.0)
        assert model.alpha0 == 0.1
        assert model.beta0 == 0.1
    
    def test_strong_prior(self):
        """Strong prior with large alpha0, beta0."""
        model = GammaGamma(alpha0=100.0, beta0=50.0, shape=1.0)
        assert model.alpha0 == 100.0
        assert model.beta0 == 50.0
