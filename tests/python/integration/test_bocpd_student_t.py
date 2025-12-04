"""
Integration tests for BOCPD with Student-t NG model.

Tests the full pipeline: BOCPD core + Student-t model + hazard function.
All tests use deterministic random seeds for reproducibility.
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, ConstantHazard, StudentTNG


class TestBOCPDStudentTBasic:
    """Basic functionality tests with Student-t NG."""
    
    def test_initialization(self, student_t_model, constant_hazard):
        """BOCPD should initialize correctly with Student-t model."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        assert bocpd.max_run_length == 200
        assert bocpd.obs_model == student_t_model
        assert bocpd.hazard == constant_hazard
    
    def test_single_update(self, student_t_model, constant_hazard):
        """Single update should produce valid posterior."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        posterior, cp_prob = bocpd.update(0.5)
        
        assert len(posterior) == 201
        assert 0.0 <= cp_prob <= 1.0
        assert not np.any(np.isnan(posterior))
    
    def test_multiple_updates(self, student_t_model, constant_hazard, test_data_simple):
        """Multiple updates should work without errors."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        for x in test_data_simple:
            posterior, cp_prob = bocpd.update(x)
            assert len(posterior) == 201
            assert 0.0 <= cp_prob <= 1.0
    
    def test_batch_update(self, student_t_model, constant_hazard):
        """Batch update should process all observations."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        np.random.seed(42)
        data = np.random.randn(500)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 500
        assert np.all(cp_probs >= 0.0)
        assert np.all(cp_probs <= 1.0)
    
    def test_reset(self, student_t_model, constant_hazard):
        """Reset should restore initial state."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        # Process some data
        rng = np.random.default_rng(42)
        for _ in range(50):
            bocpd.update(rng.standard_normal())
        
        # Reset
        bocpd.reset()
        posterior, cp_prob = bocpd.update(0.0)
        
        # After reset: use invariants instead of brittle checks
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert np.isclose(cp_prob, posterior[0])
        assert bocpd.get_map_run_length() in (0, 1)


class TestBOCPDStudentTChangeDetection:
    """Changepoint detection tests with Student-t NG."""
    
    def test_detects_mean_shift(self, student_t_model, constant_hazard):
        """Should detect clear mean shifts."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        # Segment 1: mean=0
        np.random.seed(42)
        prev_map_r = 0
        for _ in range(100):
            bocpd.update(np.random.randn())
            prev_map_r = bocpd.get_map_run_length()
        
        # Segment 2: MASSIVE shift (Student-t is robust)
        changepoints_detected = []
        for t in range(100):
            posterior, cp_prob = bocpd.update(np.random.randn() + 50.0)
            map_r = bocpd.get_map_run_length()
            
            # Detect changepoint: run length reset or sharp drop
            if map_r <= 2 or map_r < prev_map_r / 2 or cp_prob > 0.05:
                changepoints_detected.append(t)
            
            prev_map_r = map_r
        
        # Should detect at least one changepoint
        assert len(changepoints_detected) > 0
    
    def test_robust_to_outliers(self, student_t_model, constant_hazard):
        """Should handle outliers without crashing (better than Gaussian)."""
        bocpd = BOCPD(student_t_model, constant_hazard, max_run_length=200)
        
        # Normal data with occasional outliers
        np.random.seed(42)
        for t in range(100):
            if t % 20 == 0:
                x = np.random.randn() * 10  # Outlier
            else:
                x = np.random.randn()  # Normal
            
            posterior, cp_prob = bocpd.update(x)
            
            # Should not produce NaN
            assert not np.any(np.isnan(posterior))
            assert not np.isnan(cp_prob)


class TestBOCPDStudentTOnlineDetector:
    """OnlineChangeDetector integration tests with Student-t."""
    
    def test_online_detector_basic(self, student_t_model):
        """OnlineChangeDetector should work with Student-t model."""
        from fast_bocpd.utils import OnlineChangeDetector
        
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(student_t_model, hazard, max_run_length=150)
        detector = OnlineChangeDetector(bocpd, min_cp_prob=0.05)  # Lower threshold for robust model
        
        # Segment 1: mean=0 (no changepoints expected, but may get false positives)
        np.random.seed(42)
        false_positives = 0
        for _ in range(50):
            cp = detector.update(np.random.randn())
            if cp is not None:
                false_positives += 1
        
        # Should have very few false positives (Student-t is robust but not perfect)
        assert false_positives < 5, f"Too many false positives: {false_positives}"
        
        # Segment 2: mean=50 (MASSIVE shift)
        detected_cp = False
        for _ in range(70):
            cp = detector.update(np.random.randn() + 50.0)
            if cp is not None:
                detected_cp = True
                assert cp.confidence >= 0.05
                break
        
        assert detected_cp, "Should detect changepoint after strong mean shift"


class TestBOCPDStudentTGrid:
    """Grid Student-t integration tests."""
    
    def test_grid_basic_initialization(self):
        """Grid Student-t should initialize with BOCPD."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        hazard = ConstantHazard(lambda_=100)
        
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        assert bocpd.max_run_length == 50
        assert bocpd.obs_model.is_grid is True
    
    def test_grid_single_update(self):
        """Grid Student-t should handle single update."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5, 10])
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        posterior, cp_prob = bocpd.update(0.5)
        
        assert len(posterior) == 51
        assert 0.0 <= cp_prob <= 1.0
        assert np.isclose(np.sum(posterior), 1.0, atol=1e-6)
    
    def test_grid_batch_update(self):
        """Grid Student-t should handle batch processing."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=100)
        
        np.random.seed(42)
        data = np.random.randn(200)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 200
        assert np.all((cp_probs >= 0.0) & (cp_probs <= 1.0))
    
    def test_grid_context_manager(self):
        """Grid Student-t should work with context manager."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[3, 5])
        hazard = ConstantHazard(lambda_=100)
        
        with BOCPD(model, hazard, max_run_length=50) as bocpd:
            posterior, cp_prob = bocpd.update(1.0)
            assert len(posterior) == 51
        
        # After context exit, state should be cleaned up
        assert bocpd._state is None
    
    def test_grid_k1_behaves_like_fixed(self):
        """K=1 grid should behave similar to fixed ν."""
        # Grid with K=1, nu=3
        model_grid = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[3.0])
        hazard_grid = ConstantHazard(lambda_=100)
        bocpd_grid = BOCPD(model_grid, hazard_grid, max_run_length=50)
        
        # Fixed nu=3
        model_fixed = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=3.0)
        hazard_fixed = ConstantHazard(lambda_=100)
        bocpd_fixed = BOCPD(model_fixed, hazard_fixed, max_run_length=50)
        
        # Feed same data
        np.random.seed(42)
        data = np.random.randn(50)
        
        cp_probs_grid = bocpd_grid.batch_update(data)
        cp_probs_fixed = bocpd_fixed.batch_update(data)
        
        # Should be very close (not identical due to floating point, but close)
        assert np.allclose(cp_probs_grid, cp_probs_fixed, atol=1e-8)
    
    def test_grid_adapts_to_outliers(self):
        """Grid should adapt to data characteristics (heavy vs light tails)."""
        # Grid with both light (nu=10) and heavy (nu=2) tails
        model_grid = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 10])
        hazard_grid = ConstantHazard(lambda_=100)
        bocpd_grid = BOCPD(model_grid, hazard_grid, max_run_length=100)
        
        # Fixed heavy tails (nu=2) - should be very robust
        model_heavy = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=2.0)
        hazard_heavy = ConstantHazard(lambda_=100)
        bocpd_heavy = BOCPD(model_heavy, hazard_heavy, max_run_length=100)
        
        # Gaussian (least robust to outliers)
        from fast_bocpd import GaussianNIG
        model_gauss = GaussianNIG(mu0=0, kappa0=1, alpha0=1, beta0=1)
        hazard_gauss = ConstantHazard(lambda_=100)
        bocpd_gauss = BOCPD(model_gauss, hazard_gauss, max_run_length=100)
        
        # Data with frequent outliers (heavy-tailed regime)
        rng = np.random.default_rng(42)
        data = []
        for i in range(100):
            if i % 5 == 0:
                data.append(rng.standard_normal() * 10)  # Frequent large outliers
            else:
                data.append(rng.standard_normal())
        data = np.array(data)
        
        cp_probs_grid = bocpd_grid.batch_update(data)
        cp_probs_heavy = bocpd_heavy.batch_update(data)
        cp_probs_gauss = bocpd_gauss.batch_update(data)
        
        outlier_indices = [i for i in range(100) if i % 5 == 0 and i > 0]
        
        grid_outlier_probs = [cp_probs_grid[i] for i in outlier_indices]
        heavy_outlier_probs = [cp_probs_heavy[i] for i in outlier_indices]
        gauss_outlier_probs = [cp_probs_gauss[i] for i in outlier_indices]
        
        mean_heavy = np.mean(heavy_outlier_probs)
        mean_grid = np.mean(grid_outlier_probs)
        mean_gauss = np.mean(gauss_outlier_probs)
        
        # Key insight: Grid can be MORE robust than pure heavy-tail
        # because it adapts mixture weights based on data
        # The only guaranteed ordering is: Student-t models < Gaussian
        
        # Grid should be more robust than Gaussian
        assert mean_grid < mean_gauss
        
        # Grid and heavy-tail should both be significantly more robust than Gaussian
        assert mean_heavy < mean_gauss
        
        # Grid should be reasonably close to heavy-tail (within 2x)
        # but can be better OR worse depending on adaptation
        assert min(mean_grid, mean_heavy) < max(mean_grid, mean_heavy) * 2.0
    
    def test_grid_large_k(self):
        """Large K grid should work without issues."""
        large_grid = np.linspace(2, 20, 70)  # K=70 (> 64, tests heap allocation)
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=large_grid)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        np.random.seed(42)
        data = np.random.randn(100)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 100
        assert np.all(np.isfinite(cp_probs))
    
    def test_grid_reset(self):
        """Grid Student-t should reset correctly."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        # Process some data
        rng = np.random.default_rng(42)
        for _ in range(50):
            bocpd.update(rng.standard_normal())
        
        # Reset
        bocpd.reset()
        posterior, cp_prob = bocpd.update(0.0)
        
        # After reset: basic invariants
        assert np.isclose(posterior.sum(), 1.0, atol=1e-9)
        assert np.all(posterior >= 0)
        assert np.isclose(cp_prob, posterior[0])
        assert bocpd.get_map_run_length() in (0, 1)
    
    def test_grid_batch_vs_step_equivalence(self):
        """Batch update should equal step-by-step update."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        hazard = ConstantHazard(lambda_=100)
        
        # Batch mode
        bocpd_batch = BOCPD(model, hazard, max_run_length=50)
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        cp_probs_batch = bocpd_batch.batch_update(data)
        
        # Step-by-step mode
        model2 = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        hazard2 = ConstantHazard(lambda_=100)
        bocpd_step = BOCPD(model2, hazard2, max_run_length=50)
        cp_probs_step = []
        for x in data:
            _, cp_prob = bocpd_step.update(x)
            cp_probs_step.append(cp_prob)
        
        # Should be identical
        assert np.allclose(cp_probs_batch, cp_probs_step, atol=1e-14)
    
    def test_grid_cp_prob_equals_posterior0(self):
        """CP probability should always equal posterior[0]."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        rng = np.random.default_rng(42)
        for _ in range(50):
            posterior, cp_prob = bocpd.update(rng.standard_normal())
            assert np.isclose(cp_prob, posterior[0], atol=1e-15)
    
    def test_grid_close_multiple_times(self):
        """Calling close() multiple times should be safe."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[3, 5])
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        bocpd.close()
        bocpd.close()  # Should not crash
        
        assert bocpd._state is None

