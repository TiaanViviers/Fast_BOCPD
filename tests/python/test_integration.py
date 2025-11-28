"""
End-to-end integration tests
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard


class TestChangepointDetection:
    """Test actual changepoint detection scenarios"""
    
    def test_no_changepoint(self):
        """Test with constant signal (no changepoints)"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=200)
        
        # Generate constant signal with small noise
        data = np.random.randn(200) * 0.5 + 5.0
        
        cp_probs = []
        for x in data:
            _, cp_prob = bocpd.update(x)
            cp_probs.append(cp_prob)
        
        # CP probabilities should stay low
        assert np.mean(cp_probs) < 0.05
        assert np.max(cp_probs) < 0.3
    
    def test_single_changepoint(self):
        """Test detection of single changepoint"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=200)
        
        # Generate data with changepoint at t=100
        data = np.concatenate([
            np.random.randn(100) * 0.5 + 0.0,
            np.random.randn(100) * 0.5 + 5.0
        ])
        
        cp_probs = []
        for x in data:
            _, cp_prob = bocpd.update(x)
            cp_probs.append(cp_prob)
        
        # Should detect changepoint around t=100
        window = range(95, 110)
        max_prob_in_window = max(cp_probs[i] for i in window)
        assert max_prob_in_window > 0.1
    
    def test_multiple_changepoints(self):
        """Test detection of multiple changepoints"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(model, hazard, max_run_length=100)
        
        # Generate data with changepoints at t=50, 100, 150
        data = np.concatenate([
            np.random.randn(50) * 0.5 + 0.0,
            np.random.randn(50) * 0.5 + 5.0,
            np.random.randn(50) * 0.5 + -5.0,
            np.random.randn(50) * 0.5 + 10.0
        ])
        
        cp_probs = []
        for x in data:
            _, cp_prob = bocpd.update(x)
            cp_probs.append(cp_prob)
        
        # Should detect elevated probabilities around changepoints
        cp_locations = [50, 100, 150]
        for cp_loc in cp_locations:
            window = range(max(0, cp_loc-5), min(len(data), cp_loc+10))
            max_prob = max(cp_probs[i] for i in window)
            assert max_prob > 0.05, f"Failed to detect changepoint at {cp_loc}"


class TestBatchVsOnlineConsistency:
    """Test that batch and online modes produce same results"""
    
    def test_batch_equals_online(self):
        """Test that batch_update gives same results as sequential updates"""
        np.random.seed(42)
        data = np.random.randn(50)
        
        # Online mode
        model1 = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard1 = ConstantHazard(lambda_=100)
        bocpd1 = BOCPD(model1, hazard1, max_run_length=200)
        
        online_probs = []
        for x in data:
            _, cp_prob = bocpd1.update(x)
            online_probs.append(cp_prob)
        
        # Batch mode
        model2 = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard2 = ConstantHazard(lambda_=100)
        bocpd2 = BOCPD(model2, hazard2, max_run_length=200)
        
        batch_probs = bocpd2.batch_update(data)
        
        # Should be identical
        assert np.allclose(online_probs, batch_probs)


class TestReproducibility:
    """Test that results are reproducible"""
    
    def test_same_seed_same_results(self):
        """Test that same random seed gives same results"""
        model1 = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard1 = ConstantHazard(lambda_=100)
        
        # Run 1
        np.random.seed(42)
        bocpd1 = BOCPD(model1, hazard1, max_run_length=50)
        data1 = np.random.randn(20)
        probs1 = []
        for x in data1:
            _, cp_prob = bocpd1.update(x)
            probs1.append(cp_prob)
        
        # Run 2
        np.random.seed(42)
        model2 = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard2 = ConstantHazard(lambda_=100)
        bocpd2 = BOCPD(model2, hazard2, max_run_length=50)
        data2 = np.random.randn(20)
        probs2 = []
        for x in data2:
            _, cp_prob = bocpd2.update(x)
            probs2.append(cp_prob)
        
        # Data should be identical
        assert np.allclose(data1, data2)
        # Results should be identical
        assert np.allclose(probs1, probs2)


class TestKnownValues:
    """Test against known reference values"""
    
    def test_reference_implementation(self):
        """Test that we match known good values"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        # Known test case from C tests
        test_data = [0.1, 0.2, 0.15, 5.0, 5.1, 4.9]
        expected_probs = [0.010000, 0.006854, 0.005487, 0.188768, 0.003517, 0.001671]
        
        actual_probs = []
        for x in test_data:
            _, cp_prob = bocpd.update(x)
            actual_probs.append(cp_prob)
        
        # Check each probability
        for actual, expected in zip(actual_probs, expected_probs):
            assert abs(actual - expected) < 1e-4, f"Expected {expected}, got {actual}"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_batch(self):
        """Test batch update with empty data"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        data = np.array([])
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 0
    
    def test_single_observation(self):
        """Test with single observation"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        posterior_r, cp_prob = bocpd.update(0.5)
        
        assert isinstance(cp_prob, float)
        assert 0.0 <= cp_prob <= 1.0
    
    def test_large_dataset(self):
        """Test with large dataset"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=500)
        
        data = np.random.randn(1000)
        cp_probs = bocpd.batch_update(data)
        
        assert len(cp_probs) == 1000
        assert np.all((cp_probs >= 0.0) & (cp_probs <= 1.0))
    
    def test_extreme_values(self):
        """Test with extreme observation values"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        extreme_data = [1e10, -1e10, 1e-10, -1e-10]
        
        for x in extreme_data:
            posterior_r, cp_prob = bocpd.update(x)
            assert np.isfinite(cp_prob)
            assert np.all(np.isfinite(posterior_r))
