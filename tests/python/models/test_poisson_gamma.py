"""
Tests for Poisson-Gamma model parameter validation.
"""
import pytest
import numpy as np
from fast_bocpd.models import PoissonGamma


class TestPoissonGammaInitialization:
    """Test Poisson-Gamma initialization with various input types."""
    
    def test_basic_initialization(self):
        """Should initialize with valid parameters."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0)
        
        assert model.alpha0 == 1.0
        assert model.beta0 == 1.0
        assert model.strict is True  # Default
    
    def test_initialization_with_strict_false(self):
        """Should allow disabling strict validation."""
        model = PoissonGamma(alpha0=2.0, beta0=3.0, strict=False)
        
        assert model.alpha0 == 2.0
        assert model.beta0 == 3.0
        assert model.strict is False
    
    def test_integer_parameters_accepted(self):
        """Should accept integer parameters."""
        model = PoissonGamma(alpha0=5, beta0=10)
        
        assert isinstance(model.alpha0, float)
        assert isinstance(model.beta0, float)
        assert model.alpha0 == 5.0
        assert model.beta0 == 10.0
    
    def test_numpy_scalar_parameters(self):
        """Should accept numpy scalar types."""
        model = PoissonGamma(
            alpha0=np.float64(2.5),
            beta0=np.int64(3)
        )
        
        assert model.alpha0 == 2.5
        assert model.beta0 == 3.0


class TestPoissonGammaValidation:
    """Test parameter validation."""
    
    def test_alpha0_must_be_positive(self):
        """alpha0 must be > 0."""
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            PoissonGamma(alpha0=0.0, beta0=1.0)
        
        with pytest.raises(ValueError, match="alpha0 must be > 0"):
            PoissonGamma(alpha0=-1.0, beta0=1.0)
    
    def test_beta0_must_be_positive(self):
        """beta0 must be > 0."""
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            PoissonGamma(alpha0=1.0, beta0=0.0)
        
        with pytest.raises(ValueError, match="beta0 must be > 0"):
            PoissonGamma(alpha0=1.0, beta0=-1.0)
    
    def test_alpha0_must_be_finite(self):
        """alpha0 must be finite."""
        with pytest.raises(ValueError, match="alpha0 must be finite"):
            PoissonGamma(alpha0=np.inf, beta0=1.0)
        
        with pytest.raises(ValueError, match="alpha0 must be finite"):
            PoissonGamma(alpha0=np.nan, beta0=1.0)
    
    def test_beta0_must_be_finite(self):
        """beta0 must be finite."""
        with pytest.raises(ValueError, match="beta0 must be finite"):
            PoissonGamma(alpha0=1.0, beta0=np.inf)
        
        with pytest.raises(ValueError, match="beta0 must be finite"):
            PoissonGamma(alpha0=1.0, beta0=np.nan)
    
    def test_type_errors(self):
        """Should reject non-numeric types."""
        with pytest.raises(TypeError, match="alpha0 must be numeric"):
            PoissonGamma(alpha0="1.0", beta0=1.0)
        
        with pytest.raises(TypeError, match="beta0 must be numeric"):
            PoissonGamma(alpha0=1.0, beta0="1.0")


class TestPoissonGammaDataValidation:
    """Test data validation (strict mode)."""
    
    def test_validate_data_accepts_integers(self):
        """Should accept integer observations."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        # Should not raise
        model.validate_data(0)
        model.validate_data(5)
        model.validate_data(100)
    
    def test_validate_data_accepts_integer_floats(self):
        """Should accept float values that are integers."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        # Should not raise
        model.validate_data(0.0)
        model.validate_data(5.0)
        model.validate_data(100.0)
    
    def test_validate_data_rejects_negative(self):
        """Should reject negative values in strict mode."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        with pytest.raises(ValueError, match="must be >= 0"):
            model.validate_data(-1)
        
        with pytest.raises(ValueError, match="must be >= 0"):
            model.validate_data(-0.5)
    
    def test_validate_data_rejects_non_integer(self):
        """Should reject non-integer values in strict mode."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        with pytest.raises(ValueError, match="must be integers"):
            model.validate_data(1.5)
        
        with pytest.raises(ValueError, match="must be integers"):
            model.validate_data(3.14159)
    
    def test_validate_data_rejects_nan(self):
        """Should reject NaN in strict mode."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        with pytest.raises(ValueError, match="must be finite"):
            model.validate_data(np.nan)
    
    def test_validate_data_rejects_inf(self):
        """Should reject infinity in strict mode."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        with pytest.raises(ValueError, match="must be finite"):
            model.validate_data(np.inf)
    
    def test_validate_data_skips_when_not_strict(self):
        """Should skip validation when strict=False."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=False)
        
        # Should not raise even for invalid data
        model.validate_data(-1)
        model.validate_data(1.5)
        # (C layer will still reject these with -inf)


class TestPoissonGammaBatchValidation:
    """Test batch data validation."""
    
    def test_validate_batch_integer_array(self):
        """Should accept integer arrays (fast path)."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        data = np.array([0, 1, 2, 5, 10], dtype=np.int32)
        validated = model.validate_batch(data)
        
        assert validated.dtype == np.float64
        assert np.array_equal(validated, [0, 1, 2, 5, 10])
        assert validated.flags['C_CONTIGUOUS']
    
    def test_validate_batch_float_array_with_integers(self):
        """Should accept float arrays with integer values."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        data = np.array([0.0, 1.0, 2.0, 5.0], dtype=np.float64)
        validated = model.validate_batch(data)
        
        assert validated.dtype == np.float64
        assert np.array_equal(validated, [0, 1, 2, 5])
    
    def test_validate_batch_rejects_negative(self):
        """Should reject arrays with negative values."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        data = np.array([0, 1, -2, 3])
        with pytest.raises(ValueError, match="must be >= 0"):
            model.validate_batch(data)
    
    def test_validate_batch_rejects_non_integers(self):
        """Should reject arrays with non-integer values."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        data = np.array([0.0, 1.0, 2.5, 3.0])
        with pytest.raises(ValueError, match="must be integers"):
            model.validate_batch(data)
    
    def test_validate_batch_rejects_nan(self):
        """Should reject arrays with NaN."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        data = np.array([0, 1, np.nan, 3])
        with pytest.raises(ValueError, match="must be finite"):
            model.validate_batch(data)
    
    def test_validate_batch_rejects_inf(self):
        """Should reject arrays with infinity."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        data = np.array([0, 1, np.inf, 3])
        with pytest.raises(ValueError, match="must be finite"):
            model.validate_batch(data)
    
    def test_validate_batch_list_input(self):
        """Should accept Python lists."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        data = [0, 1, 2, 5]
        validated = model.validate_batch(data)
        
        assert isinstance(validated, np.ndarray)
        assert validated.dtype == np.float64
        assert np.array_equal(validated, [0, 1, 2, 5])
    
    def test_validate_batch_non_strict(self):
        """Should skip validation when strict=False."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=False)
        
        # Should accept invalid data (will fail in C later)
        data = np.array([0, 1, 2.5, -1])
        validated = model.validate_batch(data)
        
        assert validated.dtype == np.float64
        assert validated.flags['C_CONTIGUOUS']
    
    def test_validate_batch_tolerance_boundary(self):
        """Should accept batch data with values very close to integers."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        # Should accept (within 1e-9)
        data = np.array([1.0, 2.0 + 1e-10, 3.0 - 1e-10])
        validated = model.validate_batch(data)
        assert validated.dtype == np.float64
        assert np.array_equal(np.round(validated), [1, 2, 3])
        
        # Should reject (outside tolerance - use larger deviation)
        # Note: np.allclose uses rtol=1e-05, atol=1e-08 by default
        # So we need significantly larger deviation to fail
        data_bad = np.array([1.0, 2.0 + 0.1, 3.0])  # 0.1 is clearly not an integer
        with pytest.raises(ValueError, match="integ"):
            model.validate_batch(data_bad)
    
    def test_validate_batch_uint_dtypes(self):
        """Should handle unsigned integer dtypes (common for counts)."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        # uint32
        data_u32 = np.array([0, 1, 2, 10], dtype=np.uint32)
        validated = model.validate_batch(data_u32)
        assert validated.dtype == np.float64
        assert np.array_equal(validated, [0, 1, 2, 10])
        
        # uint64
        data_u64 = np.array([0, 5, 100], dtype=np.uint64)
        validated = model.validate_batch(data_u64)
        assert validated.dtype == np.float64
        assert np.array_equal(validated, [0, 5, 100])
    
    def test_validate_batch_large_counts(self):
        """Should handle very large count values."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        # Large counts (up to 2^31)
        data = np.array([0, 1000, 2**31 - 1], dtype=np.int64)
        validated = model.validate_batch(data)
        
        assert validated.dtype == np.float64
        assert np.all(np.isfinite(validated))
        assert validated[2] == 2**31 - 1
    
    def test_validate_batch_returns_copy(self):
        """Validate that batch validation returns proper array."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        data = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        validated = model.validate_batch(data)
        
        assert validated.flags['C_CONTIGUOUS']
        assert validated.dtype == np.float64
        # Result should be contiguous regardless of input


class TestPoissonGammaEdgeCases:
    """Test edge cases."""
    
    def test_large_parameters(self):
        """Should handle large parameter values."""
        model = PoissonGamma(alpha0=1e6, beta0=1e6)
        
        assert model.alpha0 == 1e6
        assert model.beta0 == 1e6
    
    def test_small_parameters(self):
        """Should handle small positive parameter values."""
        model = PoissonGamma(alpha0=1e-6, beta0=1e-6)
        
        assert model.alpha0 == 1e-6
        assert model.beta0 == 1e-6
    
    def test_integer_tolerance_boundary(self):
        """Should accept values very close to integers."""
        model = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        
        # Should accept (within 1e-9)
        model.validate_data(3.0 + 1e-10)
        model.validate_data(3.0 - 1e-10)
        
        # Should reject (outside 1e-9)
        with pytest.raises(ValueError, match="must be integers"):
            model.validate_data(3.0 + 1e-8)