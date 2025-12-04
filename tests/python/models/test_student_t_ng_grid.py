# filepath: /home/tiaan/Projects/Fast_BOCPD/tests/python/models/test_student_t_ng_grid.py
"""
Tests for Student-t NG Grid model parameter validation.
"""
import pytest
import numpy as np
from fast_bocpd.models import StudentTNG


class TestStudentTNGGridInitialization:
    """Test grid mode initialization with various input types."""
    
    def test_grid_from_list(self):
        """Grid should accept Python list for nu."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        
        assert model.is_grid is True
        assert model.K == 3
        assert model.nu is None
        assert np.allclose(model.nu_grid, [2, 3, 5])
        assert np.allclose(model.nu_prior, [1/3, 1/3, 1/3])  # Uniform default
    
    def test_grid_from_numpy_array(self):
        """Grid should accept NumPy array for nu."""
        nu_arr = np.array([2.0, 3.0, 5.0, 10.0])
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=nu_arr)
        
        assert model.is_grid is True
        assert model.K == 4
        assert np.allclose(model.nu_grid, nu_arr)
    
    def test_grid_from_tuple(self):
        """Grid should accept tuple for nu."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=(2, 3, 5))
        
        assert model.is_grid is True
        assert model.K == 3
    
    def test_grid_with_custom_prior(self):
        """Grid should accept custom prior weights."""
        model = StudentTNG(
            mu0=0, kappa0=1, alpha0=1, beta0=1,
            nu=[2, 3, 5],
            nu_prior=[0.2, 0.5, 0.3]
        )
        
        assert model.is_grid is True
        assert np.allclose(model.nu_prior, [0.2, 0.5, 0.3])
    
    def test_grid_normalizes_prior(self):
        """Grid should normalize unnormalized prior."""
        model = StudentTNG(
            mu0=0, kappa0=1, alpha0=1, beta0=1,
            nu=[2, 3, 5],
            nu_prior=[1, 2, 3]  # Unnormalized
        )
        
        assert model.is_grid is True
        assert np.allclose(model.nu_prior, [1/6, 2/6, 3/6])
        assert np.isclose(np.sum(model.nu_prior), 1.0)
    
    def test_fixed_nu_from_scalar(self):
        """Fixed ν mode with scalar nu."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=3.0)
        
        assert model.is_grid is False
        assert model.nu == 3.0
        assert model.nu_grid is None
        assert model.nu_prior is None
        assert model.K is None
    
    def test_fixed_nu_from_0d_array(self):
        """Fixed ν mode with 0-D NumPy array."""
        nu_0d = np.array(3.0)  # 0-D array
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=nu_0d)
        
        assert model.is_grid is False
        assert model.nu == 3.0
    
    def test_fixed_nu_from_numpy_scalar(self):
        """Fixed ν mode with NumPy scalar."""
        nu_scalar = np.float64(3.0)
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=nu_scalar)
        
        assert model.is_grid is False
        assert model.nu == 3.0


class TestStudentTNGGridValidation:
    """Test validation of grid parameters."""
    
    def test_empty_grid_rejected(self):
        """Empty grid should raise ValueError."""
        with pytest.raises(ValueError, match="nu_grid cannot be empty"):
            StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[])
    
    def test_negative_nu_rejected(self):
        """Negative nu in grid should raise ValueError."""
        with pytest.raises(ValueError, match="All nu values must be > 0"):
            StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, -3, 5])
    
    def test_zero_nu_rejected(self):
        """Zero nu in grid should raise ValueError."""
        with pytest.raises(ValueError, match="All nu values must be > 0"):
            StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 0, 5])
    
    def test_inf_nu_rejected(self):
        """Infinite nu in grid should raise ValueError."""
        with pytest.raises(ValueError, match="All nu values must be finite"):
            StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, np.inf, 5])
    
    def test_nan_nu_rejected(self):
        """NaN nu in grid should raise ValueError."""
        with pytest.raises(ValueError, match="All nu values must be finite"):
            StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, np.nan, 5])
    
    def test_prior_length_mismatch_rejected(self):
        """Prior length must match grid length."""
        with pytest.raises(ValueError, match="nu_prior length .* must match nu_grid length"):
            StudentTNG(
                mu0=0, kappa0=1, alpha0=1, beta0=1,
                nu=[2, 3, 5],
                nu_prior=[0.5, 0.5]  # Wrong length
            )
    
    def test_negative_prior_rejected(self):
        """Negative prior weights should raise ValueError."""
        with pytest.raises(ValueError, match="All nu_prior values must be >= 0"):
            StudentTNG(
                mu0=0, kappa0=1, alpha0=1, beta0=1,
                nu=[2, 3, 5],
                nu_prior=[0.5, -0.3, 0.8]
            )
    
    def test_all_zero_prior_rejected(self):
        """All-zero prior should raise ValueError."""
        with pytest.raises(ValueError, match="nu_prior must have at least one positive value"):
            StudentTNG(
                mu0=0, kappa0=1, alpha0=1, beta0=1,
                nu=[2, 3, 5],
                nu_prior=[0, 0, 0]
            )
    
    def test_inf_prior_rejected(self):
        """Infinite prior weight should raise ValueError."""
        with pytest.raises(ValueError, match="All nu_prior values must be finite"):
            StudentTNG(
                mu0=0, kappa0=1, alpha0=1, beta0=1,
                nu=[2, 3, 5],
                nu_prior=[0.5, np.inf, 0.5]
            )
    
    def test_multidim_prior_rejected(self):
        """Multi-dimensional prior should raise ValueError."""
        with pytest.raises(ValueError, match="nu_prior must be 1-D"):
            StudentTNG(
                mu0=0, kappa0=1, alpha0=1, beta0=1,
                nu=[2, 3, 5],
                nu_prior=[[0.3, 0.4, 0.3]]  # 2-D
            )


class TestStudentTNGGridContiguity:
    """Test that arrays are stored contiguously (for C interface)."""
    
    def test_nu_grid_is_contiguous(self):
        """nu_grid should be C-contiguous."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        
        assert model.nu_grid.flags['C_CONTIGUOUS']
        assert model.nu_grid.dtype == np.float64
    
    def test_nu_prior_is_contiguous(self):
        """nu_prior should be C-contiguous."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[2, 3, 5])
        
        assert model.nu_prior.flags['C_CONTIGUOUS']
        assert model.nu_prior.dtype == np.float64
    
    def test_non_contiguous_input_converted(self):
        """Non-contiguous input should be converted."""
        arr = np.array([2, 3, 5, 10, 20])
        non_contiguous = arr[::2]  # Every other element (non-contiguous)
        
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=non_contiguous)
        
        assert model.nu_grid.flags['C_CONTIGUOUS']
        assert np.allclose(model.nu_grid, [2, 5, 20])


class TestStudentTNGGridEdgeCases:
    """Test edge cases for grid mode."""
    
    def test_k_equals_1(self):
        """K=1 grid is valid (degenerate case)."""
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=[3.0])
        
        assert model.is_grid is True
        assert model.K == 1
        assert np.isclose(model.nu_prior[0], 1.0)
    
    def test_large_k(self):
        """Large K should work."""
        large_grid = np.linspace(2, 20, 100)
        model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=large_grid)
        
        assert model.is_grid is True
        assert model.K == 100
        assert np.isclose(np.sum(model.nu_prior), 1.0)
    
    def test_zero_weight_in_prior_allowed(self):
        """Zero weight for some components is valid."""
        model = StudentTNG(
            mu0=0, kappa0=1, alpha0=1, beta0=1,
            nu=[2, 3, 5],
            nu_prior=[0, 0.6, 0.4]  # First component has zero weight
        )
        
        assert model.is_grid is True
        assert model.nu_prior[0] == 0.0
        assert np.isclose(np.sum(model.nu_prior), 1.0)
