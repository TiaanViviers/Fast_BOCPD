"""
Fast BOCPD - Bayesian Online Changepoint Detection with C acceleration.
"""
from .core import BOCPD, is_available
from .hazard import ConstantHazard
from .models import GaussianNIG
from .utils import OnlineChangeDetector, Changepoint

__version__ = "0.1.0"
__all__ = ["BOCPD", "ConstantHazard", "GaussianNIG", "is_available", 
           "OnlineChangeDetector", "Changepoint"]
