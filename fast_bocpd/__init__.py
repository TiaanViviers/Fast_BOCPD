"""
Fast BOCPD - Bayesian Online Changepoint Detection with C acceleration.
"""
from .core import BOCPD, is_available
from .hazard import ConstantHazard
from .models import GaussianNIG, StudentTNG, PoissonGamma, BernoulliBeta, BinomialBeta, GammaGamma
from .utils import OnlineChangeDetector, Changepoint

__version__ = "1.0.0"
__all__ = ["BOCPD", "ConstantHazard", "GaussianNIG", "StudentTNG", "PoissonGamma", 
           "BernoulliBeta", "BinomialBeta", "GammaGamma", "is_available", "OnlineChangeDetector", "Changepoint"]
