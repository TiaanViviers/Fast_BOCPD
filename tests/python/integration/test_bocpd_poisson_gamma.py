"""
Integration tests for BOCPD with Poisson-Gamma model.
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, PoissonGamma, ConstantHazard


class TestBOCPDPoissonGammaBasic:
    """Basic BOCPD functionality with Poisson-Gamma model."""
    
    def test_initialization(self, poisson_model, constant_hazard):
        """Should initialize BOCPD with Poisson-Gamma model."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        assert bocpd.max_run_length == 50
        assert isinstance(bocpd.obs_model, PoissonGamma)
    
    def test_single_update(self, poisson_model, constant_hazard):
        """Should process single count observation."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(5)
        
        assert len(posterior) == 51
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert np.isclose(cp_prob, posterior[0])
    
    def test_update_with_zero(self, poisson_model, constant_hazard):
        """Should handle zero counts correctly."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(0)
        
        assert np.isfinite(cp_prob)
        assert np.all(np.isfinite(posterior))
    
    def test_batch_update(self, poisson_model, constant_hazard):
        """Should process batch of count data."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        data = np.array([1, 2, 3, 0, 5], dtype=np.int32)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 5
        assert np.all(np.isfinite(cp_probs))
        assert np.all(cp_probs >= 0)
        assert np.all(cp_probs <= 1)
    
    def test_posterior_sums_to_one(self, poisson_model, constant_hazard):
        """Posterior should always sum to 1."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        for _ in range(20):
            count = rng.poisson(lam=3)
            posterior, _ = bocpd.update(count)
            assert np.isclose(posterior.sum(), 1.0, atol=1e-9)


class TestBOCPDPoissonGammaBatchVsStep:
    """Test batch vs step-by-step equivalence."""
    
    def test_batch_equals_sequential_updates(
        self, poisson_model, constant_hazard, test_count_data_simple
    ):
        """Batch update should match sequential updates."""
        # Batch processing
        bocpd_batch = BOCPD(poisson_model, constant_hazard, max_run_length=100)
        cp_probs_batch = bocpd_batch.batch_update(test_count_data_simple)
        
        # Sequential processing
        bocpd_seq = BOCPD(poisson_model, constant_hazard, max_run_length=100)
        cp_probs_seq = []
        for x in test_count_data_simple:
            _, cp_prob = bocpd_seq.update(x)
            cp_probs_seq.append(cp_prob)
        
        cp_probs_seq = np.array(cp_probs_seq)
        
        # Should match very tightly
        assert np.allclose(cp_probs_batch, cp_probs_seq, atol=1e-12)
    
    def test_posterior_matches_after_batch(
        self, poisson_model, constant_hazard
    ):
        """Posterior should match between batch and sequential."""
        data = np.array([1, 2, 3, 0, 5, 2], dtype=np.int32)
        
        bocpd_batch = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        bocpd_batch.batch_update(data)
        posterior_batch = bocpd_batch.get_posterior()
        
        bocpd_seq = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        for x in data:
            bocpd_seq.update(x)
        posterior_seq = bocpd_seq.get_posterior()
        
        assert np.allclose(posterior_batch, posterior_seq, atol=1e-12)


class TestBOCPDPoissonGammaChangepoints:
    """Test changepoint detection with count data."""
    
    def test_detects_rate_change(
        self, poisson_model, constant_hazard, test_count_data_with_changepoint
    ):
        """Should detect rate change in Poisson data."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=150)
        cp_probs = bocpd.batch_update(test_count_data_with_changepoint)
        
        # Should detect spike near changepoint (t=100)
        # Use argmax instead of absolute threshold (more robust)
        peak_idx = np.argmax(cp_probs)
        assert 95 <= peak_idx <= 110, f"Peak at {peak_idx}, expected near 100"
        
        # Also check relative strength vs baseline
        baseline = np.mean(cp_probs[:80])
        peak_value = cp_probs[peak_idx]
        assert peak_value > 5 * baseline, "Changepoint spike should be significant"
    
    def test_stable_under_constant_rate(self, poisson_model, constant_hazard):
        """Should remain stable under constant rate."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=100)
        
        # Constant rate data
        rng = np.random.default_rng(42)
        data = rng.poisson(lam=5.0, size=200)
        cp_probs = bocpd.batch_update(data)
        
        # After warmup, cp_probs should be low and stable (use relative check)
        warmup = cp_probs[:20]
        stable_region = cp_probs[50:]
        
        # 95th percentile should be low
        assert np.quantile(stable_region, 0.95) < 0.1
        
        # Stable region should not be significantly higher than warmup
        assert np.mean(stable_region) <= np.mean(warmup) * 1.5


class TestBOCPDPoissonGammaRobustness:
    """Test robustness to edge cases."""
    
    def test_handles_many_zeros(self, poisson_model, constant_hazard):
        """Should handle data with many zeros (sparse counts)."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        # Sparse data (90% zeros)
        data = np.array([0, 0, 0, 1, 0, 0, 0, 0, 2, 0] * 10)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 100
        assert np.all(np.isfinite(cp_probs))
    
    def test_handles_large_counts(self, poisson_model, constant_hazard):
        """Should handle large count values."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        data = np.array([100, 150, 200, 95, 105])
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_max_run_length_clamp(self, poisson_model, constant_hazard):
        """Should handle time > max_run_length gracefully."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=20)
        
        # Process more data than max_run_length
        rng = np.random.default_rng(42)
        data = rng.poisson(lam=3, size=100)
        cp_probs = bocpd.batch_update(data)
        
        # Check no crash and posterior properties maintained
        posterior = bocpd.get_posterior()
        assert len(posterior) == 21  # max_run_length + 1
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert len(cp_probs) == 100
    
    def test_reset(self, poisson_model, constant_hazard):
        """Reset should restore initial state."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        # Process some data
        rng = np.random.default_rng(42)
        for _ in range(50):
            bocpd.update(rng.poisson(lam=3))
        
        # Reset
        bocpd.reset()
        posterior, cp_prob = bocpd.update(2)
        
        # After reset: use invariants
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert np.isclose(cp_prob, posterior[0])
        assert bocpd.get_map_run_length() in (0, 1)


class TestBOCPDPoissonGammaContextManager:
    """Test context manager support."""
    
    def test_context_manager_cleanup(self, poisson_model, constant_hazard):
        """Should cleanup resources with context manager."""
        with BOCPD(poisson_model, constant_hazard, max_run_length=50) as bocpd:
            bocpd.update(5)
            bocpd.update(3)
            posterior = bocpd.get_posterior()
            assert len(posterior) == 51
        
        # After exiting, state should be cleaned up
        assert bocpd._state is None
    
    def test_explicit_close(self, poisson_model, constant_hazard):
        """Should cleanup resources with explicit close()."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        bocpd.update(5)
        bocpd.close()
        
        # Should cleanup internal state
        assert bocpd._state is None


class TestBOCPDPoissonGammaValidation:
    """Test data validation integration."""
    
    def test_strict_mode_rejects_negative(self, constant_hazard):
        """Should reject negative counts in strict mode."""
        model_strict = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        bocpd = BOCPD(model_strict, constant_hazard, max_run_length=50)
        
        with pytest.raises(ValueError, match=">="):
            bocpd.update(-1)
    
    def test_strict_mode_rejects_fractional(self, constant_hazard):
        """Should reject fractional counts in strict mode."""
        model_strict = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        bocpd = BOCPD(model_strict, constant_hazard, max_run_length=50)
        
        with pytest.raises(ValueError, match="integ"):
            bocpd.update(2.5)
    
    def test_non_strict_passes_to_c_layer(self, constant_hazard):
        """Non-strict mode: invalid data passes Python but is rejected by C layer."""
        model_non_strict = PoissonGamma(alpha0=1.0, beta0=1.0, strict=False)
        bocpd = BOCPD(model_non_strict, constant_hazard, max_run_length=50)
        
        # Process some valid data first
        bocpd.update(2)
        bocpd.update(3)
        
        # Non-integer data passes Python validation (strict=False)
        # but C layer rejects it with -inf predictive, returning NULL
        # This prevents the filter from bricking (correct behavior)
        with pytest.raises(RuntimeError, match="BOCPD update failed"):
            bocpd.update(2.5)
    
    def test_batch_strict_mode(self, constant_hazard):
        """Should validate batch data in strict mode."""
        model_strict = PoissonGamma(alpha0=1.0, beta0=1.0, strict=True)
        bocpd = BOCPD(model_strict, constant_hazard, max_run_length=50)
        
        # Valid data should work
        valid_data = np.array([0, 1, 2, 5])
        cp_probs = bocpd.batch_update(valid_data)
        assert len(cp_probs) == 4
        
        # Invalid data should fail
        bocpd2 = BOCPD(model_strict, constant_hazard, max_run_length=50)
        invalid_data = np.array([0, 1, 2.5, 3])
        with pytest.raises(ValueError):
            bocpd2.batch_update(invalid_data)


class TestBOCPDPoissonGammaMetrics:
    """Test MAP and confidence metrics."""
    
    def test_get_map_run_length(self, poisson_model, constant_hazard):
        """Should return valid MAP run length."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        for _ in range(10):
            bocpd.update(rng.poisson(lam=3))
        
        map_r = bocpd.get_map_run_length()
        assert 0 <= map_r <= 50
    
    def test_get_map_confidence(self, poisson_model, constant_hazard):
        """Should return valid confidence value."""
        bocpd = BOCPD(poisson_model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        for _ in range(10):
            bocpd.update(rng.poisson(lam=3))
        
        confidence = bocpd.get_map_confidence()
        assert 0.0 <= confidence <= 1.0