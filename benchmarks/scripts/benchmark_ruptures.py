"""
Benchmark module for ruptures library.

Repository: https://github.com/deepcharles/ruptures
Installation: pip install ruptures
Type: Python/Cython/C optimized offline change point detection

This module exposes a simple function to benchmark the ruptures
implementation. It should be called from a benchmark orchestrator script.
"""

import time
import numpy as np

# Import ruptures library
import ruptures as rpt


def benchmark_ruptures(
    data: np.ndarray,
    distribution: str = "gaussian",
    mode: str = "offline",
    lambda_: float = 150,
    runs: int = 10,
    warmup: int = 2
) -> dict:
    """Benchmark ruptures on the given data.

    Args:
        data: Input time series (numpy array)
        distribution: Distribution type - currently only "gaussian" supported
        mode: Only "offline" supported (ruptures is offline-only)
        lambda_: Expected run length (used to calculate penalty parameter)
        runs: Number of benchmark runs
        warmup: Number of warmup runs (not counted)

    Returns:
        dict: Benchmark results with keys:
            - 'n_obs': Number of observations
            - 'mode': Always 'offline'
            - 'median': Median execution time (seconds)
            - 'mean': Mean execution time (seconds)
            - 'std': Standard deviation of times
            - 'cv_percent': Coefficient of variation (%)
            - 'throughput': Observations per second
    """
    # Validate inputs
    if distribution != "gaussian":
        raise ValueError(f"Distribution '{distribution}' not supported. Only 'gaussian' is available.")
    
    if mode != "offline":
        raise ValueError(f"Only 'offline' mode supported for ruptures, got '{mode}'")
    
    n = len(data)
    
    # Calculate penalty parameter from lambda
    # Heuristic: pen should be proportional to the expected run length
    # Typical values are in the range [1, 20] depending on noise level
    # We use a simple heuristic: pen = log(lambda) to get reasonable values
    pen = np.log(lambda_) if lambda_ > 1 else 3.0
    
    # Warmup runs
    for _ in range(warmup):
        _run_offline(data, pen)
    
    # Timed runs
    times = []
    for _ in range(runs):
        elapsed = _run_offline(data, pen)
        times.append(elapsed)
    
    # Calculate statistics
    return _compute_stats(times, n, mode="offline")


def _run_offline(data: np.ndarray, pen: float) -> float:
    """Execute a single offline ruptures run and return elapsed time."""
    # Reshape data if needed (ruptures expects 2D: n_samples x n_dims)
    if data.ndim == 1:
        signal = data.reshape(-1, 1)
    else:
        signal = data
    
    # Time the execution
    start = time.perf_counter()
    
    # Use Pelt algorithm with normal (Gaussian) cost function
    # model="normal" detects changes in mean and variance of Gaussian distribution
    algo = rpt.Pelt(model="normal").fit(signal)
    
    # Predict change points with penalty parameter
    # Note: ruptures returns changepoint indices including the final index (n)
    result = algo.predict(pen=pen)
    
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
