"""
Benchmark module for promised-ai/changepoint library.

Repository: https://github.com/promised-ai/changepoint
Installation: pip install changepoint
Type: Rust-based implementation with Python bindings (fast!)

This module benchmarks the Rust-based changepoint library against Fast-BOCPD.
It supports online-only changepoint detection with multiple conjugate priors.

Note: This library only supports online mode (no offline batch processing).
"""

import time
import numpy as np

# Import changepoint library
import changepoint as cpt


def benchmark_promised_ai(data, distribution="gaussian", lambda_=150,
                          runs=10, warmup=2) -> dict:
    """Benchmark promised-ai/changepoint on the given data.

    Args:
        data: Input time series (numpy array)
        distribution: Distribution type - supports:
            - 'gaussian': NormalGamma prior (matches Fast-BOCPD's GaussianNIG)
            - 'bernoulli': BetaBernoulli prior (matches Fast-BOCPD's Bernoulli)
            - 'poisson': PoissonGamma prior (matches Fast-BOCPD's Poisson)
            - 'binomial': Not supported (uses Bernoulli instead)
            - 'gamma': Not directly supported
            - 'student_t': Not directly supported (uses NormalGamma)
        lambda_: Expected run length (inverse of hazard rate)
        runs: Number of benchmark runs
        warmup: Number of warmup runs (not counted)

    Returns:
        dict: Benchmark results with keys:
            - 'n_obs': Number of observations
            - 'mode': Always 'online' (offline not supported)
            - 'median': Median execution time (seconds)
            - 'mean': Mean execution time (seconds)
            - 'std': Standard deviation of times
            - 'cv_percent': Coefficient of variation (%)
            - 'throughput': Observations per second
            - 'prior': Name of the prior used
    """
    n = len(data)
    
    # Map distribution to appropriate prior
    prior, prior_name = _get_prior(distribution)
    
    # Warmup runs
    for i in range(warmup):
        print(f" Warmup run {i+1}/{warmup} for prior {prior_name} {n} observations...")
        _run_online(data, prior, lambda_)
    
    # Timed runs
    times = []
    for i in range(runs):
        print(f" Timed run {i+1}/{runs} for prior {prior_name} {n} observations...")
        elapsed = _run_online(data, prior, lambda_)
        times.append(elapsed)
    
    # Calculate statistics
    return _compute_stats(times, n, prior_name)


def _get_prior(distribution: str):
    """Map distribution name to changepoint prior and get descriptive name.
    
    Returns:
        tuple: (prior_object, prior_name)
    """
    if distribution == "gaussian":
        return cpt.NormalGamma(m=0.0, r=1.0, s=1.0, v=1.0), "NormalGamma"
    
    elif distribution == "bernoulli":
        return cpt.BetaBernoulli(alpha=0.5, beta=0.5), "BetaBernoulli"
    
    elif distribution == "poisson":
        return cpt.PoissonGamma(shape=1.0, rate=1.0), "PoissonGamma"
    
    else:
        raise ValueError(f"Distribution '{distribution}' not supported. "
                        f"Supported: gaussian, bernoulli, poisson")


def _run_online(data: np.ndarray, prior, lambda_: float) -> float:
    """Execute a single online BOCPD run and return elapsed time.
    
    Args:
        data: Input data array
        prior: Changepoint prior object (NormalGamma, BetaBernoulli, etc.)
        lambda_: Expected run length
        
    Returns:
        float: Elapsed time in seconds
    """
    # Create new Bocpd instance for this run
    cpd = cpt.Bocpd(prior=prior, lam=lambda_)
    
    # Time the online processing
    start = time.perf_counter()
    
    # Process each datum sequentially (online)
    for x in data:
        run_length_probs = cpd.step(x)  # Returns list of run length probabilities
    
    elapsed = time.perf_counter() - start
    
    return elapsed


def _compute_stats(times: list, n_obs: int, prior_name: str) -> dict:
    """Calculate benchmark statistics from timing runs."""
    times_array = np.array(times)
    median_time = np.median(times_array)
    mean_time = np.mean(times_array)
    std_time = np.std(times_array, ddof=1)
    
    results = {
        'n_obs': n_obs,
        'mode': 'online',
        'prior': prior_name,
        'median': median_time,
        'mean': mean_time,
        'std': std_time,
        'cv_percent': (std_time / mean_time) * 100,
        'throughput': n_obs / median_time
    }
    
    return results
