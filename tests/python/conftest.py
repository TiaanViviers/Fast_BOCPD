"""
Shared pytest fixtures for Fast-BOCPD tests.
"""
import pytest
import numpy as np
from fast_bocpd import GaussianNIG, StudentTNG, PoissonGamma, ConstantHazard


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset numpy random seed before each test for determinism."""
    np.random.seed(42)
    yield


@pytest.fixture
def gaussian_nig_params():
    """Standard Gaussian NIG parameters for testing."""
    return {
        'mu0': 0.0,
        'kappa0': 1.0,
        'alpha0': 1.0,
        'beta0': 1.0
    }


@pytest.fixture
def student_t_ng_params():
    """Standard Student-t NG parameters for testing."""
    return {
        'mu0': 0.0,
        'kappa0': 1.0,
        'alpha0': 1.0,
        'beta0': 1.0,
        'nu': 3.0
    }


@pytest.fixture
def gaussian_model(gaussian_nig_params):
    """Gaussian NIG model instance."""
    return GaussianNIG(**gaussian_nig_params)


@pytest.fixture
def student_t_model(student_t_ng_params):
    """Student-t NG model instance."""
    return StudentTNG(**student_t_ng_params)


@pytest.fixture
def poisson_gamma_params():
    """Standard Poisson-Gamma parameters for testing."""
    return {
        'alpha0': 2.0,
        'beta0': 1.0
    }


@pytest.fixture
def poisson_model(poisson_gamma_params):
    """Poisson-Gamma model instance."""
    return PoissonGamma(**poisson_gamma_params)


@pytest.fixture
def constant_hazard():
    """Standard constant hazard function."""
    return ConstantHazard(lambda_=100)


@pytest.fixture
def test_data_simple():
    """Simple deterministic test data."""
    np.random.seed(42)
    return np.random.randn(100)


@pytest.fixture
def test_data_with_changepoint():
    """Test data with a clear changepoint at t=100."""
    np.random.seed(42)
    segment1 = np.random.randn(100)  # mean=0
    segment2 = np.random.randn(100) + 5.0  # mean=5
    return np.concatenate([segment1, segment2])


@pytest.fixture
def test_count_data_simple():
    """Simple deterministic count data (Poisson)."""
    np.random.seed(42)
    return np.random.poisson(lam=3.0, size=100)


@pytest.fixture
def test_count_data_with_changepoint():
    """Count data with changepoint at t=100 (rate change)."""
    np.random.seed(42)
    segment1 = np.random.poisson(lam=2.0, size=100)  # low rate
    segment2 = np.random.poisson(lam=10.0, size=100)  # high rate
    return np.concatenate([segment1, segment2])

