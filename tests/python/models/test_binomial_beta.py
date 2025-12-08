"""
Unit tests for BinomialBeta observation model.
"""
import pytest
import numpy as np
from fast_bocpd.models import BinomialBeta


class TestBinomialBetaInit:
    """Test initialization and parameter validation."""
    
    def test_basic_initialization(self):
        """Should initialize with valid parameters."""
        model = BinomialBeta(alpha0=1.0, beta0=2.0, n_trials=10)
        assert model.alpha0 == 1.0
        assert model.beta0 == 2.0
        assert model.n_trials == 10
        assert model.strict is True
    
    def test_accepts_int_and_numpy_scalars(self):
        """Should accept int and numpy scalar types."""
        model = BinomialBeta(
            alpha0=np.int64(3), 
            beta0=np.float64(4.5), 
            n_trials=np.int32(20)
        )
        assert isinstance(model.alpha0, float)
        assert isinstance(model.beta0, float)
        assert isinstance(model.n_trials, int)
        assert model.alpha0 == 3.0
        assert model.beta0 == 4.5
        assert model.n_trials == 20
    
    def test_strict_parameter(self):
        """Should respect strict parameter."""
        model_strict = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        assert model_strict.strict is True
        
        model_non_strict = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=False)
        assert model_non_strict.strict is False
    
    @pytest.mark.parametrize("alpha0,beta0,n_trials,msg", [
        (0.0, 1.0, 10, "alpha0 must be > 0"),
        (-1.0, 1.0, 10, "alpha0 must be > 0"),
        (1.0, 0.0, 10, "beta0 must be > 0"),
        (1.0, -2.0, 10, "beta0 must be > 0"),
        (1.0, 1.0, 0, "n_trials must be >= 1"),
        (1.0, 1.0, -5, "n_trials must be >= 1"),
        (np.inf, 1.0, 10, "alpha0 must be finite"),
        (np.nan, 1.0, 10, "alpha0 must be finite"),
        (1.0, np.inf, 10, "beta0 must be finite"),
        (1.0, np.nan, 10, "beta0 must be finite"),
    ])
    def test_parameter_validation(self, alpha0, beta0, n_trials, msg):
        """Should reject invalid parameters."""
        with pytest.raises(ValueError, match=msg):
            BinomialBeta(alpha0=alpha0, beta0=beta0, n_trials=n_trials)
    
    def test_type_errors(self):
        """Should reject non-numeric types."""
        with pytest.raises(TypeError, match="alpha0 must be numeric"):
            BinomialBeta(alpha0="1", beta0=1.0, n_trials=10)
        
        with pytest.raises(TypeError, match="beta0 must be numeric"):
            BinomialBeta(alpha0=1.0, beta0="1", n_trials=10)
        
        with pytest.raises(TypeError, match="n_trials must be integer"):
            BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10.5)
    
    def test_n_trials_one(self):
        """Should accept n_trials=1 (reduces to Bernoulli)."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=1)
        assert model.n_trials == 1


class TestBinomialBetaValidateData:
    """Test single-value data validation."""
    
    def test_accepts_valid_counts(self):
        """Should accept valid count values."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        
        # Should not raise
        for k in range(11):  # 0..10
            model.validate_data(k)
            model.validate_data(float(k))
    
    def test_accepts_near_integer(self):
        """Should accept values very close to integers."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        
        # Within tolerance (1e-9)
        model.validate_data(5.0 + 1e-10)
        model.validate_data(5.0 - 1e-10)
        model.validate_data(0.0 + 1e-10)
        model.validate_data(10.0 - 1e-10)
    
    @pytest.mark.parametrize("k", [
        -1, -0.5, 11, 12, 100,  # Out of range
        5.5, 3.2, 0.1,          # Non-integer
        10.0 + 1e-6,            # Too far from integer
    ])
    def test_rejects_invalid_counts(self, k):
        """Should reject invalid count values in strict mode."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        with pytest.raises(ValueError):
            model.validate_data(k)
    
    def test_rejects_negative(self):
        """Should reject negative counts."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        with pytest.raises(ValueError, match="must be >= 0"):
            model.validate_data(-1)
    
    def test_rejects_exceeding_n_trials(self):
        """Should reject counts > n_trials."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        with pytest.raises(ValueError, match="<= n_trials"):
            model.validate_data(11)
    
    def test_rejects_nan_inf(self):
        """Should reject NaN and infinity."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        
        for x in (np.nan, np.inf, -np.inf):
            with pytest.raises(ValueError, match="finite"):
                model.validate_data(x)
    
    def test_non_strict_skips_validation(self):
        """Non-strict mode should skip validation."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=False)
        
        # Should not raise
        model.validate_data(-1)
        model.validate_data(5.5)
        model.validate_data(20)


class TestBinomialBetaValidateBatch:
    """Test batch data validation."""
    
    def test_accepts_int_array(self):
        """Should handle integer arrays efficiently."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        data = np.array([0, 3, 7, 10, 5], dtype=np.int32)
        
        out = model.validate_batch(data)
        
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
        assert np.array_equal(out, np.array([0.0, 3.0, 7.0, 10.0, 5.0]))
    
    def test_accepts_various_int_dtypes(self):
        """Should handle different integer dtypes."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        
        for dtype in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16):
            data = np.array([0, 5, 10], dtype=dtype)
            out = model.validate_batch(data)
            assert out.dtype == np.float64
            assert np.array_equal(out, [0, 5, 10])
    
    def test_accepts_float_near_integer(self):
        """Should accept float arrays near integers."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        data = np.array([0.0, 5.0, 10.0 + 1e-10, 3.0 - 1e-10], dtype=np.float64)
        
        out = model.validate_batch(data)
        
        assert out.dtype == np.float64
        # Should round to nearest integer
        assert np.allclose(out, [0, 5, 10, 3], atol=1e-9)
    
    def test_rejects_negative_in_array(self):
        """Should reject negative values."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        data = np.array([0, 5, -1, 3], dtype=np.int32)
        
        with pytest.raises(ValueError, match="must be >= 0"):
            model.validate_batch(data)
    
    def test_rejects_exceeding_n_trials_in_array(self):
        """Should reject values > n_trials."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        data = np.array([0, 5, 11, 3], dtype=np.int32)
        
        with pytest.raises(ValueError, match="<= n_trials"):
            model.validate_batch(data)
    
    def test_rejects_non_integer_in_float_array(self):
        """Should reject non-integer values in float array."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        data = np.array([0.0, 5.0, 3.5, 7.0], dtype=np.float64)
        
        with pytest.raises(ValueError, match="must be integers"):
            model.validate_batch(data)
    
    def test_rejects_nan_inf(self):
        """Should reject NaN and infinity in arrays."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        
        for bad in (np.nan, np.inf, -np.inf):
            data = np.array([0.0, 5.0, bad])
            with pytest.raises(ValueError, match="finite"):
                model.validate_batch(data)
    
    def test_list_input(self):
        """Should accept Python lists."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        
        out = model.validate_batch([0, 3, 7, 10])
        
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
        assert np.array_equal(out, [0, 3, 7, 10])
    
    def test_non_strict_accepts_any_value(self):
        """Non-strict mode should skip validation."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=False)
        
        out = model.validate_batch([0, 5.5, 20, -1])
        
        assert out.dtype == np.float64
        assert out.flags["C_CONTIGUOUS"]
    
    def test_empty_array(self):
        """Should handle empty arrays."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        
        out = model.validate_batch(np.array([]))
        
        assert len(out) == 0
        assert out.dtype == np.float64
    
    def test_boundary_values(self):
        """Should handle boundary values correctly."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        
        # Test exact boundaries
        data = np.array([0, 10], dtype=np.int32)
        out = model.validate_batch(data)
        assert np.array_equal(out, [0, 10])


class TestBinomialBetaSpecialCases:
    """Test special cases and edge conditions."""
    
    def test_n_equals_1_similar_to_bernoulli(self):
        """With n_trials=1, should behave like Bernoulli."""
        model = BinomialBeta(alpha0=2.0, beta0=3.0, n_trials=1, strict=True)
        
        # Should only accept 0 and 1
        model.validate_data(0)
        model.validate_data(1)
        
        with pytest.raises(ValueError):
            model.validate_data(2)
    
    def test_large_n_trials(self):
        """Should handle large n_trials."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=1000, strict=True)
        
        assert model.n_trials == 1000
        model.validate_data(0)
        model.validate_data(500)
        model.validate_data(1000)
        
        with pytest.raises(ValueError):
            model.validate_data(1001)
    
    def test_uniform_prior(self):
        """Should create uniform prior with alpha0=beta0=1."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10)
        
        # Prior mean = alpha0 / (alpha0 + beta0) = 0.5
        assert model.alpha0 == 1.0
        assert model.beta0 == 1.0
    
    def test_informative_prior(self):
        """Should support informative priors."""
        model = BinomialBeta(alpha0=30.0, beta0=70.0, n_trials=100)
        
        # Prior mean ≈ 0.3 (30 successes, 70 failures)
        assert model.alpha0 == 30.0
        assert model.beta0 == 70.0
