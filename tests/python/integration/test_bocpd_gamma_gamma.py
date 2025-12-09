"""
Integration tests for BOCPD with Gamma-Gamma (Fixed Shape) model.
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, GammaGamma, ConstantHazard


class TestBOCPDGammaGammaBasic:
    """Basic BOCPD functionality with Gamma-Gamma model."""
    
    def test_initialization(self, gamma_gamma_model, constant_hazard):
        """Should initialize BOCPD with Gamma-Gamma model."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        assert bocpd.max_run_length == 50
        assert isinstance(bocpd.obs_model, GammaGamma)
    
    def test_single_update(self, gamma_gamma_model, constant_hazard):
        """Should process single positive observation."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(1.5)
        
        assert len(posterior) == 51
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert np.isclose(cp_prob, posterior[0])
    
    def test_update_with_small_value(self, gamma_gamma_model, constant_hazard):
        """Should handle very small positive values."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(0.001)
        
        assert np.isfinite(cp_prob)
        assert np.all(np.isfinite(posterior))
    
    def test_update_with_large_value(self, gamma_gamma_model, constant_hazard):
        """Should handle large values."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(1000.0)
        
        assert np.isfinite(cp_prob)
        assert np.all(np.isfinite(posterior))
    
    def test_batch_update(self, gamma_gamma_model, constant_hazard):
        """Should process batch of positive data."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        data = np.array([0.5, 1.0, 2.5, 5.0, 0.1])
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 5
        assert np.all(np.isfinite(cp_probs))
        assert np.all(cp_probs >= 0)
        assert np.all(cp_probs <= 1)
    
    def test_posterior_sums_to_one(self, gamma_gamma_model, constant_hazard):
        """Posterior should always sum to 1."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.exponential(scale=2.0)
            posterior, _ = bocpd.update(x)
            assert np.isclose(posterior.sum(), 1.0, atol=1e-9)


class TestBOCPDGammaGammaBatchVsStep:
    """Test batch vs step-by-step equivalence."""
    
    def test_batch_equals_sequential_updates(
        self, gamma_gamma_model, constant_hazard, test_gamma_data_simple
    ):
        """Batch update should match sequential updates."""
        # Batch processing
        bocpd_batch = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=100)
        cp_probs_batch = bocpd_batch.batch_update(test_gamma_data_simple)
        
        # Sequential processing
        bocpd_seq = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=100)
        cp_probs_seq = []
        for x in test_gamma_data_simple:
            _, cp_prob = bocpd_seq.update(float(x))
            cp_probs_seq.append(cp_prob)
        
        cp_probs_seq = np.array(cp_probs_seq)
        
        # Should match very tightly
        assert np.allclose(cp_probs_batch, cp_probs_seq, atol=1e-12)
    
    def test_posterior_matches_after_batch(
        self, gamma_gamma_model, constant_hazard
    ):
        """Posterior should match between batch and sequential."""
        data = np.array([0.5, 1.2, 2.8, 0.9, 3.5, 1.1])
        
        bocpd_batch = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        bocpd_batch.batch_update(data)
        posterior_batch = bocpd_batch.get_posterior()
        
        bocpd_seq = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        for x in data:
            bocpd_seq.update(float(x))
        posterior_seq = bocpd_seq.get_posterior()
        
        assert np.allclose(posterior_batch, posterior_seq, atol=1e-12)


class TestBOCPDGammaGammaChangepoints:
    """Test changepoint detection with Gamma data."""
    
    def test_detects_rate_shift_exponential(
        self, constant_hazard, test_gamma_data_with_changepoint
    ):
        """Should detect shift in rate for Exponential data (shape=1)."""
        # Exponential model (shape=1.0)
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=200)
        cp_probs = bocpd.batch_update(test_gamma_data_with_changepoint)
        
        # Should detect spike near changepoint (t=100)
        # The changepoint might not always be the global max, but should be locally elevated
        
        # Look at window around true changepoint
        window_start = max(0, 100 - 10)
        window_end = min(len(cp_probs), 100 + 10)
        window = cp_probs[window_start:window_end]
        baseline = np.median(cp_probs[:50])  # Use median for robustness
        
        # Should have elevated probability near changepoint
        # Using more lenient threshold since Gamma data can be noisy
        assert np.max(window) > 1.5 * baseline, "Should detect changepoint"
    
    def test_detects_rate_shift_gamma(
        self, constant_hazard
    ):
        """Should detect shift in rate for Gamma data (shape>1)."""
        # Generate data with rate shift
        rng = np.random.default_rng(42)
        # Rate 2.0 (small values, mean = shape/rate = 2.0/2.0 = 1.0)
        segment1 = rng.gamma(shape=2.0, scale=1.0/2.0, size=100)
        # Rate 0.5 (large values, mean = 2.0/0.5 = 4.0)
        segment2 = rng.gamma(shape=2.0, scale=1.0/0.5, size=100)
        data = np.concatenate([segment1, segment2])
        
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=2.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=200)
        cp_probs = bocpd.batch_update(data)
        
        # Should detect spike near changepoint
        window_start = max(0, 100 - 10)
        window_end = min(len(cp_probs), 100 + 10)
        window = cp_probs[window_start:window_end]
        baseline = np.mean(cp_probs[:50])
        
        assert np.max(window) > 2 * baseline, "Should detect changepoint"
    
    def test_stable_under_constant_rate(self, constant_hazard):
        """Should remain stable under constant rate."""
        model = GammaGamma(alpha0=1.0, beta0=1.0, shape=1.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=100)
        
        # Constant rate Exponential data
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=2.0, size=300)
        cp_probs = bocpd.batch_update(data)
        
        # After warmup, cp_probs should be low and stable
        stable_region = cp_probs[50:]
        
        # 95th percentile should be low
        assert np.percentile(stable_region, 95) < 0.2, "Should remain stable"
    
    def test_changepoint_at_start(self, gamma_gamma_model, constant_hazard):
        """Should handle data processing from start."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        # First few observations
        data = np.array([1.0, 2.0, 3.0])
        cp_probs = bocpd.batch_update(data)
        
        # All cp_probs should be valid
        assert len(cp_probs) == 3
        assert np.all(np.isfinite(cp_probs))
        assert np.all(cp_probs >= 0)
        assert np.all(cp_probs <= 1)


class TestBOCPDGammaGammaReset:
    """Test reset functionality."""
    
    def test_reset_clears_state(self, gamma_gamma_model, constant_hazard):
        """Reset should clear state to initial."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        # Process some data
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        bocpd.batch_update(data)
        
        # Reset
        bocpd.reset()
        
        # Should be back to initial state
        posterior = bocpd.get_posterior()
        assert posterior[0] == 1.0
        assert np.all(posterior[1:] == 0.0)


class TestBOCPDGammaGammaNumericalStability:
    """Test numerical stability edge cases."""
    
    def test_very_small_values(self, gamma_gamma_model, constant_hazard):
        """Should handle very small positive values."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        data = np.array([1e-6, 1e-5, 1e-4, 1e-3])
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_very_large_values(self, gamma_gamma_model, constant_hazard):
        """Should handle very large values."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        data = np.array([1e6, 1e5, 1e4, 1e3])
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_mixed_scales(self, gamma_gamma_model, constant_hazard):
        """Should handle mixed scale data."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=50)
        
        data = np.array([1e-3, 1e3, 1e-2, 1e2, 1.0])
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_long_sequence(self, gamma_gamma_model, constant_hazard):
        """Should remain stable over long sequences."""
        bocpd = BOCPD(gamma_gamma_model, constant_hazard, max_run_length=500)
        
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=2.0, size=1000)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 1000
        assert np.all(np.isfinite(cp_probs))
        assert np.all(cp_probs >= 0)
        assert np.all(cp_probs <= 1)


class TestBOCPDGammaGammaShapeVariants:
    """Test different shape parameter values."""
    
    def test_exponential_shape_one(self, constant_hazard):
        """Test with Exponential (shape=1.0)."""
        model = GammaGamma(alpha0=2.0, beta0=1.0, shape=1.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=2.0, size=100)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_shape_two(self, constant_hazard):
        """Test with shape=2.0."""
        model = GammaGamma(alpha0=2.0, beta0=1.0, shape=2.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=2.0, scale=1.0, size=100)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_shape_five(self, constant_hazard):
        """Test with larger shape=5.0."""
        model = GammaGamma(alpha0=2.0, beta0=1.0, shape=5.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=5.0, scale=1.0, size=100)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_shape_fractional(self, constant_hazard):
        """Test with fractional shape=1.5."""
        model = GammaGamma(alpha0=2.0, beta0=1.0, shape=1.5)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        data = rng.gamma(shape=1.5, scale=1.0, size=100)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_shape_near_one(self, constant_hazard):
        """Test with shape very close to 1.0."""
        model = GammaGamma(alpha0=2.0, beta0=1.0, shape=1.0 + 1e-6)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=1.0, size=100)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))


class TestBOCPDGammaGammaPriorVariants:
    """Test different prior parameter configurations."""
    
    def test_weak_prior(self, constant_hazard):
        """Test with weak prior (small alpha0, beta0)."""
        model = GammaGamma(alpha0=0.1, beta0=0.1, shape=1.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=2.0, size=100)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_strong_prior(self, constant_hazard):
        """Test with strong prior (large alpha0, beta0)."""
        model = GammaGamma(alpha0=100.0, beta0=50.0, shape=1.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=2.0, size=100)
        cp_probs = bocpd.batch_update(data)
        
        assert np.all(np.isfinite(cp_probs))
    
    def test_informative_prior_matches_data(self, constant_hazard):
        """Test with informative prior matching data."""
        # Prior mean rate: alpha0/beta0 = 2.0/1.0 = 2.0
        # Data mean: 1/scale = 1/0.5 = 2.0 (rate)
        model = GammaGamma(alpha0=20.0, beta0=10.0, shape=1.0)
        bocpd = BOCPD(model, constant_hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=0.5, size=100)  # rate=2.0
        cp_probs = bocpd.batch_update(data)
        
        # Should be stable (no false positives)
        stable_region = cp_probs[20:]
        assert np.percentile(stable_region, 95) < 0.15
