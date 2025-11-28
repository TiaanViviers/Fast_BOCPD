"""
Test hazard function wrappers and parameter validation
"""
import pytest
from fast_bocpd.hazard import ConstantHazard


class TestConstantHazard:
    """Test ConstantHazard parameter wrapper"""
    
    def test_valid_initialization(self):
        """Test initialization with valid lambda"""
        hazard = ConstantHazard(lambda_=100.0)
        
        assert hazard.lambda_ == 100.0
        assert hazard.H == 0.01
    
    def test_type_conversion(self):
        """Test that lambda is converted to float"""
        hazard = ConstantHazard(lambda_=100)
        
        assert isinstance(hazard.lambda_, float)
        assert isinstance(hazard.H, float)
    
    def test_hazard_calculation(self):
        """Test that hazard H = 1/lambda is calculated correctly"""
        hazard = ConstantHazard(lambda_=200.0)
        assert hazard.H == 0.005
        
        hazard = ConstantHazard(lambda_=50.0)
        assert hazard.H == 0.02
    
    def test_invalid_negative_lambda(self):
        """Test that negative lambda is rejected"""
        with pytest.raises(ValueError, match="lambda_ must be > 0"):
            ConstantHazard(lambda_=-1.0)
    
    def test_invalid_zero_lambda(self):
        """Test that zero lambda is rejected"""
        with pytest.raises(ValueError, match="lambda_ must be > 0"):
            ConstantHazard(lambda_=0.0)
    
    def test_very_small_lambda(self):
        """Test with very small lambda (high hazard)"""
        hazard = ConstantHazard(lambda_=1.1)
        assert 0.0 < hazard.H < 1.0
    
    def test_very_large_lambda(self):
        """Test with very large lambda (low hazard)"""
        hazard = ConstantHazard(lambda_=10000.0)
        assert hazard.H == 0.0001
        assert 0.0 < hazard.H < 1.0
    
    def test_lambda_one_should_fail(self):
        """Test that lambda=1 fails (would give H=1, not in (0,1))"""
        # Actually lambda=1 gives H=1, which is exactly 1, not in open interval (0,1)
        # But due to floating point, it might work. Let's test edge case.
        # Based on our validation, H must be in (0, 1) exclusive
        with pytest.raises(ValueError):
            ConstantHazard(lambda_=1.0)
    
    def test_different_lambda_values(self):
        """Test various lambda values produce correct hazards"""
        test_cases = [
            (10.0, 0.1),
            (50.0, 0.02),
            (100.0, 0.01),
            (1000.0, 0.001),
        ]
        
        for lambda_val, expected_H in test_cases:
            hazard = ConstantHazard(lambda_=lambda_val)
            assert abs(hazard.H - expected_H) < 1e-10
