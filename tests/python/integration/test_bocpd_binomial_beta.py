"""
Integration tests for BOCPD with Binomial-Beta model.
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, BinomialBeta, BernoulliBeta, ConstantHazard


class TestBOCPDBinomialBetaBasic:
    """Basic BOCPD functionality with Binomial-Beta model."""
    
    def test_initialization(self, binomial_model, constant_hazard):
        """Should initialize BOCPD with Binomial-Beta model."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        
        assert bocpd.max_run_length == 50
        assert isinstance(bocpd.obs_model, BinomialBeta)
    
    def test_single_update(self, binomial_model, constant_hazard):
        """Should process single binomial observation."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(5)
        
        assert len(posterior) == 51
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert np.isclose(cp_prob, posterior[0])
    
    def test_update_with_zero(self, binomial_model, constant_hazard):
        """Should handle zero successes correctly."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(0)
        
        assert np.isfinite(cp_prob)
        assert np.all(np.isfinite(posterior))
    
    def test_update_with_max(self, constant_hazard):
        """Should handle all successes correctly."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(10)
        
        assert np.isfinite(cp_prob)
        assert np.all(np.isfinite(posterior))
    
    def test_batch_update(self, binomial_model, constant_hazard):
        """Should process batch of binomial data."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        
        data = np.array([0, 3, 7, 10, 5], dtype=np.int32)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 5
        assert np.all(np.isfinite(cp_probs))
        assert np.all(cp_probs >= 0)
        assert np.all(cp_probs <= 1)
    
    def test_posterior_sums_to_one(self, binomial_model, constant_hazard):
        """Posterior should always sum to 1."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        for _ in range(20):
            k = rng.binomial(10, 0.5)
            posterior, _ = bocpd.update(k)
            assert np.isclose(posterior.sum(), 1.0, atol=1e-9)


class TestBOCPDBinomialBetaBatchVsStep:
    """Test batch vs step-by-step equivalence."""
    
    def test_batch_equals_sequential_updates(
        self, binomial_model, constant_hazard, test_binomial_data_simple
    ):
        """Batch update should match sequential updates."""
        # Batch processing
        bocpd_batch = BOCPD(binomial_model, constant_hazard, max_run_length=100)
        cp_probs_batch = bocpd_batch.batch_update(test_binomial_data_simple)
        
        # Sequential processing
        bocpd_seq = BOCPD(binomial_model, constant_hazard, max_run_length=100)
        cp_probs_seq = []
        for x in test_binomial_data_simple:
            _, cp_prob = bocpd_seq.update(int(x))
            cp_probs_seq.append(cp_prob)
        
        cp_probs_seq = np.array(cp_probs_seq)
        
        # Should match very tightly
        assert np.allclose(cp_probs_batch, cp_probs_seq, atol=1e-12)
    
    def test_posterior_matches_after_batch(
        self, binomial_model, constant_hazard
    ):
        """Posterior should match between batch and sequential."""
        data = np.array([0, 3, 7, 5, 10, 2], dtype=np.int32)
        
        bocpd_batch = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        bocpd_batch.batch_update(data)
        posterior_batch = bocpd_batch.get_posterior()
        
        bocpd_seq = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        for x in data:
            bocpd_seq.update(int(x))
        posterior_seq = bocpd_seq.get_posterior()
        
        assert np.allclose(posterior_batch, posterior_seq, atol=1e-12)


class TestBOCPDBinomialBetaChangepoints:
    """Test changepoint detection with binomial data."""
    
    def test_detects_probability_shift(
        self, binomial_model, constant_hazard, test_binomial_data_with_changepoint
    ):
        """Should detect shift in success probability."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=200)
        cp_probs = bocpd.batch_update(test_binomial_data_with_changepoint)
        
        # Should detect spike near changepoint (t=150)
        peak_idx = np.argmax(cp_probs)
        
        # Check that peak is in the vicinity of the changepoint
        assert 0 <= peak_idx < len(cp_probs), f"Peak at {peak_idx}"
        
        # Look at window around true changepoint
        window_start = max(0, 150 - 10)
        window_end = min(len(cp_probs), 150 + 10)
        window = cp_probs[window_start:window_end]
        baseline = np.mean(cp_probs[:100])
        
        # Should have elevated probability near changepoint
        assert np.max(window) > 2 * baseline, "Should detect changepoint"
    
    def test_stable_under_constant_probability(self, binomial_model, constant_hazard):
        """Should remain stable under constant success probability."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=100)
        
        # Constant probability data
        rng = np.random.default_rng(42)
        data = rng.binomial(10, 0.5, size=300)
        cp_probs = bocpd.batch_update(data)
        
        # After warmup, cp_probs should be low and stable
        stable_region = cp_probs[50:]
        
        # 95th percentile should be low
        assert np.quantile(stable_region, 0.95) < 0.1
        
        # Mean should be reasonable
        assert np.mean(stable_region) < 0.05


class TestBOCPDBinomialBetaRobustness:
    """Test robustness to edge cases."""
    
    def test_handles_all_zeros(self, binomial_model, constant_hazard):
        """Should handle data with all zeros."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        
        data = np.zeros(100, dtype=np.int32)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 100
        assert np.all(np.isfinite(cp_probs))
    
    def test_handles_all_max(self, constant_hazard):
        """Should handle data with all n_trials successes."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        data = np.full(100, 10, dtype=np.int32)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 100
        assert np.all(np.isfinite(cp_probs))
    
    def test_handles_constant_mid_value(self, binomial_model, constant_hazard):
        """Should handle constant mid-range value."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        
        data = np.full(100, 5, dtype=np.int32)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_handles_boundary_values(self, constant_hazard):
        """Should handle boundary values (0 and N)."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=20)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        data = np.array([0, 20, 0, 20, 10], dtype=np.int32)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_max_run_length_clamp(self, binomial_model, constant_hazard):
        """Should handle time > max_run_length gracefully."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=20)
        
        # Process more data than max_run_length
        rng = np.random.default_rng(42)
        data = rng.binomial(10, 0.5, size=100)
        cp_probs = bocpd.batch_update(data)
        
        # Check no crash and posterior properties maintained
        posterior = bocpd.get_posterior()
        assert len(posterior) == 21  # max_run_length + 1
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert len(cp_probs) == 100
    
    def test_reset(self, binomial_model, constant_hazard):
        """Reset should restore initial state."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        
        # Process some data
        rng = np.random.default_rng(42)
        for _ in range(50):
            bocpd.update(rng.binomial(10, 0.5))
        
        # Reset
        bocpd.reset()
        posterior, cp_prob = bocpd.update(5)
        
        # After reset: use invariants
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert np.isclose(cp_prob, posterior[0])
        assert bocpd.get_map_run_length() in (0, 1)


class TestBOCPDBinomialBetaContextManager:
    """Test context manager support."""
    
    def test_context_manager_cleanup(self, binomial_model, constant_hazard):
        """Should cleanup resources with context manager."""
        with BOCPD(binomial_model, constant_hazard, max_run_length=50) as bocpd:
            bocpd.update(5)
            bocpd.update(3)
            posterior = bocpd.get_posterior()
            assert len(posterior) == 51
        
        # After exiting, state should be cleaned up
        assert bocpd._state is None
    
    def test_explicit_close(self, binomial_model, constant_hazard):
        """Should cleanup resources with explicit close()."""
        bocpd = BOCPD(binomial_model, constant_hazard, max_run_length=50)
        bocpd.update(5)
        bocpd.close()
        
        # Should cleanup internal state
        assert bocpd._state is None


class TestBOCPDBinomialBetaValidation:
    """Test data validation integration."""
    
    def test_strict_mode_rejects_non_integer(self, constant_hazard):
        """Should reject non-integer values in strict mode."""
        model_strict = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        bocpd = BOCPD(model_strict, constant_hazard, max_run_length=50)
        
        with pytest.raises(ValueError, match="integer"):
            bocpd.update(5.5)
    
    def test_strict_mode_rejects_out_of_range(self, constant_hazard):
        """Should reject out-of-range values in strict mode."""
        model_strict = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        bocpd = BOCPD(model_strict, constant_hazard, max_run_length=50)
        
        with pytest.raises(ValueError, match="<= n_trials"):
            bocpd.update(11)
        
        with pytest.raises(ValueError, match=">= 0"):
            bocpd.update(-1)
    
    def test_non_strict_accepts_bad_values(self, constant_hazard):
        """Non-strict mode skips Python validation but C still guards."""
        model_non_strict = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=False)
        bocpd = BOCPD(model_non_strict, constant_hazard, max_run_length=50)
        
        # Non-strict mode skips Python validation, but C code still rejects invalid data
        # This prevents filter corruption but doesn't give nice error messages
        # Invalid data will cause RuntimeError from C returning NULL
        
        # Valid edge cases work fine
        bocpd.update(0)
        bocpd.update(10)
        
        # But truly invalid data still fails (just with less helpful errors)
        with pytest.raises(RuntimeError):
            bocpd.update(5.5)  # Non-integer
        
        with pytest.raises(RuntimeError):
            bocpd.update(20)  # > n_trials
        
        with pytest.raises(RuntimeError):
            bocpd.update(-1)  # Negative
    
    def test_batch_strict_validation(self, constant_hazard):
        """Batch mode should validate in strict mode."""
        model_strict = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10, strict=True)
        bocpd = BOCPD(model_strict, constant_hazard, max_run_length=50)
        
        bad_data = np.array([0, 5, 11, 3])  # 11 > n_trials
        with pytest.raises(ValueError):
            bocpd.batch_update(bad_data)


class TestBOCPDBinomialBetaNEqualsOne:
    """Test special case: n_trials=1 (should behave like Bernoulli)."""
    
    def test_n1_binomial_vs_bernoulli_changepoint_detection(self, constant_hazard):
        """N=1 Binomial should match Bernoulli for changepoint detection."""
        # Create N=1 Binomial model
        binom_model = BinomialBeta(alpha0=2.0, beta0=3.0, n_trials=1)
        bern_model = BernoulliBeta(alpha0=2.0, beta0=3.0)
        
        # Same data (binary)
        rng = np.random.default_rng(123)
        data = rng.binomial(1, 0.5, size=100)
        
        # Run both
        bocpd_binom = BOCPD(binom_model, constant_hazard, max_run_length=100)
        cp_probs_binom = bocpd_binom.batch_update(data)
        
        bocpd_bern = BOCPD(bern_model, constant_hazard, max_run_length=100)
        cp_probs_bern = bocpd_bern.batch_update(data)
        
        # Should match very closely (predictive distributions are identical)
        assert np.allclose(cp_probs_binom, cp_probs_bern, atol=1e-10)
    
    def test_n1_posterior_equivalence(self, constant_hazard):
        """N=1 Binomial posterior should match Bernoulli."""
        binom_model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=1)
        bern_model = BernoulliBeta(alpha0=1.0, beta0=1.0)
        
        data = np.array([1, 0, 1, 1, 0])
        
        bocpd_binom = BOCPD(binom_model, constant_hazard, max_run_length=50)
        bocpd_binom.batch_update(data)
        posterior_binom = bocpd_binom.get_posterior()
        
        bocpd_bern = BOCPD(bern_model, constant_hazard, max_run_length=50)
        bocpd_bern.batch_update(data)
        posterior_bern = bocpd_bern.get_posterior()
        
        assert np.allclose(posterior_binom, posterior_bern, atol=1e-12)


class TestBOCPDBinomialBetaDifferentN:
    """Test with different values of n_trials."""
    
    @pytest.mark.parametrize("n_trials", [1, 5, 10, 20, 50, 100])
    def test_various_n_values(self, constant_hazard, n_trials):
        """Should work with various n_trials values."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=n_trials)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        # Generate random data in valid range
        rng = np.random.default_rng(42)
        data = rng.binomial(n_trials, 0.5, size=50)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 50
        assert np.all(np.isfinite(cp_probs))
        assert np.all(cp_probs >= 0)
    
    def test_large_n_stability(self, constant_hazard):
        """Should remain numerically stable with large n_trials."""
        model = BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=1000)
        bocpd = BOCPD(model, constant_hazard, max_run_length=100)
        
        rng = np.random.default_rng(42)
        data = rng.binomial(1000, 0.5, size=200)
        cp_probs = bocpd.batch_update(data)
        
        # All probabilities should be finite
        assert np.all(np.isfinite(cp_probs))
        
        # Posterior should remain normalized
        posterior = bocpd.get_posterior()
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
