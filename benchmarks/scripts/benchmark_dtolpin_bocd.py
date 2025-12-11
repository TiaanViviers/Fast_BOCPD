"""
Benchmark module for dtolpin/bocd implementation.

Repository: https://github.com/dtolpin/bocd
Installation: Clone repo to benchmarks/competitors/dtolpin_bocd/
Type: Python/NumPy implementation with fixed-space optimization

This module exposes a simple function to benchmark the dtolpin BOCD
implementation. It should be called from a benchmark orchestrator script.
"""

import sys
import time
import numpy as np
from functools import partial
from pathlib import Path

# Add dtolpin's bocd to path
DTOLPIN_PATH = Path(__file__).parent.parent / "competitors" / "dtolpin_bocd"
sys.path.insert(0, str(DTOLPIN_PATH))

# Import bocd library
from bocd.bocd import BOCD, constant_hazard, StudentT


def benchmark_dtolpin_bocd(data: np.ndarray, distribution: str = "student_t",
    mode: str = "online", lambda_: float = 150, runs: int = 10, warmup: int = 2
) -> dict:
    """Benchmark dtolpin/bocd on the given data.

    Args:
        data: Input time series (numpy array)
        distribution: Distribution type - currently only "student_t" supported
        mode: Only "online" supported (dtolpin/bocd is online-only)
        lambda_: Expected run length (inverse of hazard rate)
        runs: Number of benchmark runs
        warmup: Number of warmup runs (not counted)

    Returns:
        dict: Benchmark results with keys:
            - 'n_obs': Number of observations
            - 'mode': Always 'online'
            - 'median': Median execution time (seconds)
            - 'mean': Mean execution time (seconds)
            - 'std': Standard deviation of times
            - 'cv_percent': Coefficient of variation (%)
            - 'throughput': Observations per second
    """
    # Validate inputs
    if distribution != "student_t":
        raise ValueError(f"Distribution '{distribution}' not supported. Only 'student_t' is available.")
    
    if mode != "online":
        raise ValueError(f"Only 'online' mode supported for dtolpin/bocd, got '{mode}'")
    
    n = len(data)
    
    # Set up model parameters (match Fast-BOCPD defaults)
    alpha = 1.0   # alpha0 in StudentTNG
    beta = 1.0    # beta0 in StudentTNG  
    kappa = 1.0   # kappa0 in StudentTNG
    mu = 0.0      # mu0 in StudentTNG
    
    # Warmup runs
    for _ in range(warmup):
        _run_online(data, lambda_, alpha, beta, kappa, mu)
    
    # Timed runs
    times = []
    for _ in range(runs):
        elapsed = _run_online(data, lambda_, alpha, beta, kappa, mu)
        times.append(elapsed)
    
    # Calculate statistics
    return _compute_stats(times, n, mode="online")


def _run_online(data: np.ndarray, lambda_: float, alpha: float, beta: float, 
                kappa: float, mu: float) -> float:
    """Execute a single online BOCD run and return elapsed time."""
    # Create fresh detector for each run
    hazard_fn = partial(constant_hazard, lambda_)
    likelihood = StudentT(alpha, beta, kappa, mu)
    detector = BOCD(hazard_fn, likelihood)
    
    # Time the execution
    start = time.perf_counter()
    for x in data:
        detector.update(x)
    elapsed = time.perf_counter() - start
    
    return elapsed


def _compute_stats(times: list, n_obs: int, mode: str) -> dict:
    """Calculate benchmark statistics from timing runs."""
    times_array = np.array(times)
    median_time = np.median(times_array)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array, ddof=1)
    
    results = {
        'n_obs': n_obs,
        'mode': mode,
        'median': median_time,
        'mean': mean_time,
        'std': std_time,
        'cv_percent': (std_time / mean_time) * 100,
        'throughput': n_obs / median_time
    }
    
    return results
