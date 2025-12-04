"""
Cross-model comparison tests.

Tests that compare behavior of different observation models.
All tests use deterministic random seeds for reproducibility.
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, ConstantHazard


class TestModelComparison:
    """Compare Gaussian vs Student-t behavior."""
    
    def test_both_models_work_on_same_data(self, gaussian_model, student_t_model):
        """Both models should process the same data successfully."""
        bocpd_gaussian = BOCPD(gaussian_model, ConstantHazard(100), max_run_length=200)
        bocpd_student_t = BOCPD(student_t_model, ConstantHazard(100), max_run_length=200)
        
        np.random.seed(42)
        data = np.random.randn(100)
        
        for x in data:
            _, cp_prob_g = bocpd_gaussian.update(x)
            _, cp_prob_t = bocpd_student_t.update(x)
            
            # Both should give valid probabilities
            assert 0.0 <= cp_prob_g <= 1.0
            assert 0.0 <= cp_prob_t <= 1.0
    
    def test_student_t_more_robust_to_single_outlier(self, gaussian_model, student_t_model):
        """Student-t should give LOWER changepoint probability for outliers."""
        # Use lower hazard for more stable long runs
        bocpd_gaussian = BOCPD(gaussian_model, ConstantHazard(500), max_run_length=600)
        bocpd_student_t = BOCPD(student_t_model, ConstantHazard(500), max_run_length=600)
        
        # Build very strong confidence with lots of normal data
        np.random.seed(42)
        for _ in range(400):
            x = np.random.randn()
            bocpd_gaussian.update(x)
            bocpd_student_t.update(x)
        
        # Moderate outlier (5 std deviations)
        outlier = 5.0
        _, cp_prob_g = bocpd_gaussian.update(outlier)
        _, cp_prob_t = bocpd_student_t.update(outlier)
        
        # Student-t should be MORE robust (strictly lower CP probability)
        assert cp_prob_t < cp_prob_g, (
            f"Student-t should be more robust to outliers. "
            f"Got: Gaussian={cp_prob_g:.6f}, Student-t={cp_prob_t:.6f}"
        )
    
    def test_similar_behavior_on_normal_data(self, gaussian_model, student_t_model):
        """On normal data without outliers, models should behave similarly."""
        bocpd_gaussian = BOCPD(gaussian_model, ConstantHazard(100), max_run_length=200)
        bocpd_student_t = BOCPD(student_t_model, ConstantHazard(100), max_run_length=200)
        
        # Normal data, no outliers
        np.random.seed(42)
        data = np.random.randn(50) * 0.5  # Small variance
        
        cp_probs_g = []
        cp_probs_t = []
        
        for x in data:
            _, cp_prob_g = bocpd_gaussian.update(x)
            _, cp_prob_t = bocpd_student_t.update(x)
            cp_probs_g.append(cp_prob_g)
            cp_probs_t.append(cp_prob_t)
        
        # Average CP probabilities should be similar (no outliers)
        avg_g = np.mean(cp_probs_g)
        avg_t = np.mean(cp_probs_t)
        
        # Should be within reasonable range of each other
        assert abs(avg_g - avg_t) < 0.05, (
            f"On normal data, models should behave similarly. "
            f"Got: Gaussian avg={avg_g:.4f}, Student-t avg={avg_t:.4f}"
        )
    
    def test_heavy_tails_on_extreme_outlier(self, gaussian_model, student_t_model):
        """Student-t with fixed ν and weighted updates behaves differently.
        
        Key insight: Fixed-ν Student-t with ν=3 gives HIGHER probability to outliers
        in the predictive PDF (heavier tails). This means outliers look LESS surprising,
        resulting in LOWER changepoint probability.
        
        This is correct behavior: Student-t treats outliers as normal data variations
        rather than regime changes. For detecting regime changes on outliers, you'd want
        the grid-based approach with ν marginalization.
        """
        bocpd_gaussian = BOCPD(gaussian_model, ConstantHazard(250), max_run_length=300)
        bocpd_student_t = BOCPD(student_t_model, ConstantHazard(250), max_run_length=300)
        
        # Build up confidence with normal data
        np.random.seed(42)
        for _ in range(150):
            x = np.random.randn()
            bocpd_gaussian.update(x)
            bocpd_student_t.update(x)
        
        # Extreme outlier (15 std deviations)
        extreme_outlier = 15.0
        _, cp_prob_g = bocpd_gaussian.update(extreme_outlier)
        _, cp_prob_t = bocpd_student_t.update(extreme_outlier)
        
        # Gaussian should flag it as unusual (high CP prob)
        assert cp_prob_g > 0.3, f"Gaussian should flag extreme outlier. Got: {cp_prob_g:.4f}"
        
        # Student-t should be MORE robust (LOWER CP prob)
        # This is the correct behavior: heavy tails make outliers look normal
        assert cp_prob_t < cp_prob_g, (
            f"Student-t should be more robust (lower CP prob). "
            f"Got: Gaussian={cp_prob_g:.4f}, Student-t={cp_prob_t:.4f}"
        )
        
        # Both should give valid probabilities
        assert 0.0 <= cp_prob_g <= 1.0
        assert 0.0 <= cp_prob_t <= 1.0
