"""
Test new offline features: MAP run length, confidence, posterior
"""
import pytest
import numpy as np
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard


class TestMAPRunLength:
    """Test MAP run length functionality"""
    
    def test_map_at_initialization(self):
        """MAP should be 0 at initialization"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        # Before any data, posterior should be all at r=0
        map_r = bocpd.get_map_run_length()
        assert map_r == 0
    
    def test_map_increases_without_changepoint(self):
        """MAP should increase when no changepoint occurs"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        # Feed similar observations
        map_values = []
        for _ in range(10):
            bocpd.update(0.1)
            map_values.append(bocpd.get_map_run_length())
        
        # MAP should generally increase (allowing for occasional decreases)
        assert map_values[-1] > map_values[0]
    
    def test_map_resets_on_changepoint(self):
        """MAP should jump to 0 on changepoint"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        # Feed constant data
        for _ in range(10):
            bocpd.update(0.1)
        
        map_before = bocpd.get_map_run_length()
        assert map_before > 5  # Should have grown
        
        # Feed very different observation
        bocpd.update(10.0)
        map_after = bocpd.get_map_run_length()
        
        # Should jump to 0 (changepoint)
        assert map_after == 0


class TestMAPConfidence:
    """Test MAP confidence functionality"""
    
    def test_confidence_is_probability(self):
        """Confidence should be a valid probability"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        for x in [0.1, 0.2, 0.15]:
            bocpd.update(x)
            confidence = bocpd.get_map_confidence()
            assert 0.0 <= confidence <= 1.0
    
    def test_confidence_matches_posterior(self):
        """Confidence should match posterior at MAP index"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        bocpd.update(0.5)
        
        map_r = bocpd.get_map_run_length()
        confidence = bocpd.get_map_confidence()
        posterior = bocpd.get_posterior()
        
        assert abs(confidence - posterior[map_r]) < 1e-10


class TestGetPosterior:
    """Test get_posterior functionality"""
    
    def test_posterior_is_distribution(self):
        """Posterior should sum to 1"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        bocpd.update(0.5)
        posterior = bocpd.get_posterior()
        
        assert len(posterior) == 51  # max_run_length + 1
        assert abs(np.sum(posterior) - 1.0) < 1e-6
        assert np.all(posterior >= 0.0)
    
    def test_posterior_matches_update(self):
        """get_posterior should match update return value"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        posterior_from_update, _ = bocpd.update(0.5)
        posterior_from_get = bocpd.get_posterior()
        
        assert np.allclose(posterior_from_update, posterior_from_get)
    
    def test_posterior_evolves(self):
        """Posterior should change as data arrives"""
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=100)
        bocpd = BOCPD(model, hazard, max_run_length=50)
        
        bocpd.update(0.1)
        posterior1 = bocpd.get_posterior().copy()
        
        bocpd.update(0.2)
        posterior2 = bocpd.get_posterior().copy()
        
        # Posteriors should be different
        assert not np.allclose(posterior1, posterior2)


class TestOfflineFeaturesCombined:
    """Test offline features working together"""
    
    def test_complete_workflow(self):
        """Test realistic usage of all offline features"""
        np.random.seed(42)
        model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
        hazard = ConstantHazard(lambda_=50)
        bocpd = BOCPD(model, hazard, max_run_length=100)
        
        # Generate data with changepoint at t=25
        data = np.concatenate([
            np.random.randn(25) * 0.5 + 0.0,
            np.random.randn(25) * 0.5 + 5.0
        ])
        
        map_history = []
        confidence_history = []
        
        for x in data:
            bocpd.update(x)
            map_history.append(bocpd.get_map_run_length())
            confidence_history.append(bocpd.get_map_confidence())
        
        # Check that MAP detected the changepoint
        # Around t=25, MAP should have jumped to 0
        assert 0 in map_history[23:28]
        
        # Confidence should always be valid
        assert all(0.0 <= c <= 1.0 for c in confidence_history)
