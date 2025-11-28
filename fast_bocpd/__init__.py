"""
Fast BOCPD - Bayesian Online Changepoint Detection with C acceleration.
"""
from .bocpd_accelerated import BOCPD, is_available
from .hazard import ConstantHazard
from .models import GaussianNIG

__version__ = "0.1.0"
__all__ = ["BOCPD", "ConstantHazard", "GaussianNIG", "is_available"]
