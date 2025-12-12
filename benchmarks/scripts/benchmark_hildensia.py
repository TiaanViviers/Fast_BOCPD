"""
Benchmark module for hildensia/bayesian_changepoint_detection library.

Repository: https://github.com/hildensia/bayesian_changepoint_detection
Installation: Clone from GitHub and pip install -e .
Type: PyTorch-based implementation with CPU/GPU support

This module supports both CPU and GPU benchmarking with proper CUDA synchronization.
"""

import time
import numpy as np
from functools import partial

# Import bayesian_changepoint_detection library
import torch
from bayesian_changepoint_detection import (
    online_changepoint_detection,
    offline_changepoint_detection,
    constant_hazard,
    const_prior
)
from bayesian_changepoint_detection.online_likelihoods import StudentT as OnlineStudentT
from bayesian_changepoint_detection.offline_likelihoods import StudentT as OfflineStudentT


def benchmark_hildensia(
    data: np.ndarray,
    distribution: str = "gaussian",
    mode: str = "online",
    lambda_: float = 150,
    runs: int = 10,
    warmup: int = 2,
    device: str = "cpu"
) -> dict:
    """Benchmark hildensia/bayesian_changepoint_detection on the given data.

    Args:
        data: Input time series (numpy array)
        distribution: Distribution type - currently only "gaussian" supported
        mode: "online" or "offline"
        lambda_: Expected run length (inverse of hazard rate)
        runs: Number of benchmark runs
        warmup: Number of warmup runs (not counted)
        device: Device to run on - "cpu" or "cuda" (default: "cpu")

    Returns:
        dict: Benchmark results with keys:
            - 'n_obs': Number of observations
            - 'mode': 'online' or 'offline'
            - 'median': Median execution time (seconds)
            - 'mean': Mean execution time (seconds)
            - 'std': Standard deviation of times
            - 'cv_percent': Coefficient of variation (%)
            - 'throughput': Observations per second
            - 'device': Device used for computation
    """
    # Validate inputs
    if distribution != "gaussian":
        raise ValueError(f"Distribution '{distribution}' not supported. Only 'gaussian' is available.")
    
    if mode not in ["online", "offline"]:
        raise ValueError(f"Mode must be 'online' or 'offline', got '{mode}'")
    
    # Validate and setup device
    if device == "cuda" and not torch.cuda.is_available():
        print(f"WARNING: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    # Keep both forms: torch.device for .to() and sync, string for library APIs
    torch_device = torch.device(device)
    device_str = "cuda" if torch_device.type == "cuda" else "cpu"
    n = len(data)
    
    # WARNING: Hildensia's online implementation has O(n²) complexity
    # CPU: only run <= 1000 (too slow)
    # GPU: allow up to 10000 (may OOM at 100k)
    if mode == "online":
        if device_str == "cpu" and n > 1000:
            print(f"WARNING: Hildensia online mode has O(n²) complexity on CPU.")
            print(f"         Benchmarking {n} observations may take extremely long.")
            print(f"         Estimated time: ~{(n/100)**2 * 0.6:.0f} seconds")
            print(f"         Consider using --device cuda or limiting to n=1000")
        elif device_str == "cuda" and n > 10000:
            print(f"WARNING: Hildensia online mode may run out of GPU memory at n={n}")
            print(f"         O(n²) memory growth can exceed GPU capacity.")
    
    # Convert data to PyTorch tensor and move to device
    data_tensor = torch.from_numpy(data).float().to(torch_device)
    
    # Student-T hyperparameters (matching Fast-BOCPD's GaussianNIG predictive)
    alpha = 1.0   # Shape parameter
    beta = 1.0    # Scale parameter
    kappa = 1.0   # Precision parameter
    mu = 0.0      # Mean parameter
    
    print(f"Benchmarking Hildensia: {mode} mode, {n} observations, device={device_str}")
    
    # Warmup runs
    for i in range(warmup):
        if mode == "online":
            _run_online(data_tensor, lambda_, alpha, beta, kappa, mu, torch_device, device_str)
        else:
            _run_offline(data_tensor, lambda_, alpha, beta, kappa, mu, torch_device, device_str)
    
    # Timed runs
    times = []
    for i in range(runs):
        if mode == "online":
            elapsed = _run_online(data_tensor, lambda_, alpha, beta, kappa, mu, torch_device, device_str)
        else:
            elapsed = _run_offline(data_tensor, lambda_, alpha, beta, kappa, mu, torch_device, device_str)
        times.append(elapsed)
    
    # Calculate statistics
    return _compute_stats(times, n, mode, device_str)


def _run_online(data: torch.Tensor, lambda_: float, alpha: float, beta: float,
                kappa: float, mu: float, device: torch.device, device_str: str) -> float:
    """Execute a single online BOCPD run and return elapsed time.
    
    Uses proper CUDA synchronization for accurate GPU timing.
    
    Args:
        data: Input tensor (already on correct device)
        lambda_: Expected run length
        alpha, beta, kappa, mu: StudentT hyperparameters
        device: torch.device for synchronization
        device_str: "cpu" or "cuda" string for library API
    """
    # Create hazard function (constant hazard with expected run length lambda_)
    hazard_func = partial(constant_hazard, lambda_)
    
    # Create likelihood model - pass string to library API
    likelihood = OnlineStudentT(alpha=alpha, beta=beta, kappa=kappa, mu=mu, device=device_str)
    
    # Synchronize before timing (critical for GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time the execution - pass string to library API
    start = time.perf_counter()
    run_length_probs, changepoint_probs = online_changepoint_detection(
        data, hazard_func, likelihood, device=device_str
    )
    
    # Synchronize after computation (critical for GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return elapsed


def _run_offline(data: torch.Tensor, lambda_: float, alpha: float, beta: float,
                 kappa: float, mu: float, device: torch.device, device_str: str) -> float:
    """Execute a single offline BOCPD run and return elapsed time.
    
    Uses proper CUDA synchronization for accurate GPU timing.
    
    Args:
        data: Input tensor (already on correct device)
        lambda_: Expected run length (unused in offline, kept for consistency)
        alpha, beta, kappa, mu: StudentT hyperparameters (unused in offline StudentT)
        device: torch.device for synchronization
        device_str: "cpu" or "cuda" string for library API
    """
    n = len(data)
    
    # Create prior function (constant prior)
    prior_func = partial(const_prior, p=1/(n+1))
    
    # Create likelihood model - pass string to library API
    likelihood = OfflineStudentT(device=device_str)
    
    # Synchronize before timing (critical for GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time the execution - pass device_str to library API
    start = time.perf_counter()
    Q, P, changepoint_log_probs = offline_changepoint_detection(
        data, prior_func, likelihood, device=device_str
    )
    
    # Synchronize after computation (critical for GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return elapsed


def _compute_stats(times: list, n_obs: int, mode: str, device: str) -> dict:
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
        'throughput': n_obs / median_time,
        'device': device
    }
    
    return results
