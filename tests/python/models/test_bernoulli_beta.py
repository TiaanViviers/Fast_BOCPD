"""
Unit tests for BernoulliBeta observation model.
"""
import pytest
import numpy as np
from fast_bocpd.models import BernoulliBeta


class TestBernoulliBetaInit:
    """Test initialization and parameter validation."""
    
    def test_basic_initialization(self):
        """Should initialize with valid parameters."""
        model = BernoulliBeta(alpha0=1.0, beta0=2.0)
        assert model.alpha0 == 1.0
        assert model.beta0 == 2.0
        assert model.strict is True
    
    def test_accepts_int_and_numpy_scalars(self):
        """Should accept int and numpy scalar types."""
        model = BernoulliBeta(alpha0=np.int64(3), beta0=np.float64(4.5))
        assert isinstance(model.alpha0, float)
        assert isinstance(model.beta0, float)
        assert model.alpha0 == 3.0
        assert model.beta0 == 4.5
    
    def test_strict_parameter(self):
        """Should respect strict parameter."""
        model_strict = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        assert model_strict.strict is True
        
        model_non_strict = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=False)
        assert model_non_strict.strict is False
    
    @pytest.mark.parametrize("alpha0,beta0,msg", [
        (0.0, 1.0, "alpha0 must be > 0"),
        (-1.0, 1.0, "alpha0 must be > 0"),
        (1.0, 0.0, "beta0 must be > 0"),
        (1.0, -2.0, "beta0 must be > 0"),
        (np.inf, 1.0, "alpha0 must be finite"),
        (np.nan, 1.0, "alpha0 must be finite"),
        (1.0, np.inf, "beta0 must be finite"),
        (1.0, np.nan, "beta0 must be finite"),
    ])
    def test_parameter_validation(self, alpha0, beta0, msg):
        """Should reject invalid parameters."""
        with pytest.raises(ValueError, match=msg):
            BernoulliBeta(alpha0=alpha0, beta0=beta0)
    
    def test_type_errors(self):
        """Should reject non-numeric types."""
        with pytest.raises(TypeError, match="alpha0 must be numeric"):
            BernoulliBeta(alpha0="1", beta0=1.0)
        
        with pytest.raises(TypeError, match="beta0 must be numeric"):
            BernoulliBeta(alpha0=1.0, beta0="1")


class TestBernoulliBetaValidateData:
    """Test single-value data validation."""
    
    def test_accepts_valid_binary(self):
        """Should accept valid binary values."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        
        # Should not raise
        model.validate_data(0)
        model.validate_data(1)
        model.validate_data(0.0)
        model.validate_data(1.0)
        model.validate_data(True)
        model.validate_data(False)
    
    def test_accepts_near_binary(self):
        """Should accept values very close to 0 or 1."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        
        # Within tolerance (1e-9)
        model.validate_data(1.0 + 1e-10)
        model.validate_data(0.0 - 1e-10)
        model.validate_data(1.0 - 1e-10)
        model.validate_data(0.0 + 1e-10)
    
    @pytest.mark.parametrize("x", [
        -1, 2, 0.5, 0.2, 0.9, 
        1.0 + 1e-6,  # Too far from 1
        -1e-6,       # Too far from 0
    ])
    def test_rejects_non_binary(self, x):
        """Should reject non-binary values in strict mode."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        with pytest.raises(ValueError, match="0 or 1"):
            model.validate_data(x)
    
    def test_rejects_nan_inf(self):
        """Should reject NaN and infinity."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        
        for x in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError, match="finite"):
                model.validate_data(x)
    
    def test_non_strict_skips_validation(self):
        """Non-strict mode should skip validation."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=False)
        
        # Should not raise
        model.validate_data(-1)
        model.validate_data(0.5)
        model.validate_data(2)


class TestBernoulliBetaValidateBatch:
    """Test batch data validation."""
    
    def test_accepts_bool_array_fastpath(self):
        """Should handle boolean arrays efficiently."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        data = np.array([True, False, True, False], dtype=bool)
        
        out = model.validate_batch(data)
        
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
        assert np.array_equal(out, np.array([1.0, 0.0, 1.0, 0.0]))
    
    def test_accepts_int_array(self):
        """Should handle integer arrays."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        data = np.array([0, 1, 1, 0], dtype=np.int32)
        
        out = model.validate_batch(data)
        
        assert out.dtype == np.float64
        assert np.array_equal(out, np.array([0.0, 1.0, 1.0, 0.0]))
    
    def test_accepts_float_near_binary(self):
        """Should accept float arrays near 0/1."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        data = np.array([0.0, 1.0, 1.0 + 1e-10, 0.0 - 1e-10], dtype=np.float64)
        
        out = model.validate_batch(data)
        
        assert out.dtype == np.float64
        # Should round to nearest binary
        assert np.allclose(out, [0, 1, 1, 0], atol=1e-9)
    
    def test_rejects_non_binary_in_int_array(self):
        """Should reject non-binary values in integer array."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        data = np.array([0, 1, 2, 0], dtype=np.int32)
        
        with pytest.raises(ValueError, match="binary"):
            model.validate_batch(data)
    
    def test_rejects_non_binary_in_float_array(self):
        """Should reject non-binary values in float array."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        data = np.array([0.0, 1.0, 0.5, 0.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="binary"):
            model.validate_batch(data)
    
    def test_rejects_nan_inf(self):
        """Should reject NaN and infinity in arrays."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        
        for bad in (np.nan, np.inf, -np.inf):
            data = np.array([0.0, 1.0, bad])
            with pytest.raises(ValueError, match="finite"):
                model.validate_batch(data)
    
    def test_list_input(self):
        """Should accept Python lists."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        
        out = model.validate_batch([0, 1, 1, 0])
        
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
        assert np.array_equal(out, [0, 1, 1, 0])
    
    def test_non_strict_accepts_any_value(self):
        """Non-strict mode should skip validation."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=False)
        
        out = model.validate_batch([0, 0.5, 2, -1])
        
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
    
    def test_empty_array(self):
        """Should handle empty arrays."""
        model = BernoulliBeta(alpha0=1.0, beta0=1.0, strict=True)
        
        out = model.validate_batch(np.array([]))
        
        assert len(out) == 0
        assert out.dtype == np.float64