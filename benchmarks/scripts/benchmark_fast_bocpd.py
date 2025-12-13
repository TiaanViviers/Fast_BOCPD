"""
Benchmarking script for Fast-BOCPD internal performance evaluation.

This module provides comprehensive performance benchmarking for all Fast-BOCPD
models across multiple dataset sizes. It measures both online (sequential) and
offline (batch) processing modes, tracking median runtime, throughput, and
statistical stability.

Purpose
-------
1. Track iterative performance improvements across development versions
2. Validate O(n) scaling behavior for all models
3. Quantify batch processing speedup (offline vs online mode)
4. Ensure consistent, predictable performance (low CV%)

Usage
-----
Benchmark a specific distribution:
    python benchmark_fast_bocpd.py --distribution gaussian --runs 10
    
Benchmark all sizes for a distribution:
    python benchmark_fast_bocpd.py --distribution student_t_fixed
    
Benchmark specific size only:
    python benchmark_fast_bocpd.py --distribution poisson --size 10000

Or use the convenience wrapper:
    ../../benchmark.sh gaussian
    ../../benchmark.sh Fbocpd  # All distributions

Supported Distributions
-----------------------
- gaussian: GaussianNIG (Normal-Inverse-Gamma prior)
- student_t_fixed: StudentTNG with fixed degrees of freedom
- student_t_grid: StudentTNG with grid search over degrees of freedom
- bernoulli: BernoulliBeta for binary data
- binomial: BinomialBeta for proportion data
- poisson: PoissonGamma for count data
- gamma: GammaGamma for positive continuous data

Output Metrics
--------------
- Median runtime (s): Typical execution time (robust to outliers)
- Throughput (obs/sec): Processing speed
- CV%: Coefficient of variation (std/mean × 100), measures stability
- Batch speedup: Offline vs online performance ratio

Notes
-----
- All benchmarks use λ=150 (expected run length)
- Datasets are pre-generated in ../data/
- Results are logged in ../Benchmark_tracking.md for version comparison
- Low CV% (<2%) indicates consistent, predictable performance

See Also
--------
benchmark_competitors.py : Benchmark against other libraries
generate_data.py : Generate synthetic benchmark datasets
../README.md : Full benchmarking methodology and results
"""

import sys
import argparse
from pathlib import Path
import time
import numpy as np

# Add parent directory to path to import fast_bocpd
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from fast_bocpd import (
    BOCPD, ConstantHazard, OnlineChangeDetector,
    GaussianNIG, StudentTNG, PoissonGamma, BernoulliBeta,
    BinomialBeta, GammaGamma,
)


DISTRIBUTIONS = ['gaussian', "student_t_fixed", "student_t_grid", "bernoulli",
                 "binomial", "poisson", "gamma"]


def main():
    args = get_args()
    data_files = get_files(args.distribution, args.size)
    
    # Set up benchmarking model
    model = get_model(args.distribution)
    hazard = ConstantHazard(lambda_=args.lambda_)
    max_run_length = 3 * args.lambda_
    detector = BOCPD(model, hazard, max_run_length=max_run_length)
    
    print_start(args.distribution, args.lambda_, args.runs, args.warmup_runs, data_files)
    
    all_results = {}
    for file in data_files:
        data = np.load(file)
        
        # Run both modes
        online_results = benchmark_dataset(data, detector, args.runs, args.warmup_runs, 'online')
        offline_results = benchmark_dataset(data, detector, args.runs, args.warmup_runs, 'offline')

        name = get_file_size(file)
        all_results[name] = {'online': online_results, 'offline': offline_results}
        print_results(online_results, offline_results, name)

    if len(all_results) > 1:
        print_summary(all_results)
        
    
#===============================================================================
# Benchmarking Logic
#===============================================================================
    
def benchmark_dataset(data, detector, n_runs, warmup_runs, mode):
    """Benchmark BOCPD on dataset with multiple runs."""
    
    # Warmup
    for _ in range(warmup_runs):
        _ = benchmark_single_run(data, detector, mode)

    # Timed runs
    times = []
    for _ in range(n_runs):
        elapsed = benchmark_single_run(data, detector, mode)
        times.append(elapsed)
    
    # Statistics
    median_time = np.median(times)
    results = {
        'n_obs': len(data),
        'mode': mode,
        'median': median_time,
        'mean': np.mean(times),
        'std': np.std(times, ddof=1),
        'cv_percent': (np.std(times, ddof=1) / np.mean(times)) * 100,
        'throughput': len(data) / median_time
    }
    
    return results


def benchmark_single_run(data, detector, mode):
    """Run BOCPD detection once and return elapsed time."""
    
    if mode == 'online':
        online_detector = OnlineChangeDetector(detector)
        start = time.perf_counter()
        for x in data:
            _ = online_detector.update(x)
        elapsed = time.perf_counter() - start
        
    else:  # batch/offline mode
        start = time.perf_counter()
        _ = detector.batch_update(data)
        elapsed = time.perf_counter() - start
    
    # Reset detector for next run
    detector.reset()
    
    return elapsed
    
#===============================================================================
# Setup Helper Functions
#===============================================================================

def get_model(distribution):
    """Instantiate model based on distribution type."""
    if distribution == 'gaussian':
        return GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    elif distribution == "student_t_fixed":
        return StudentTNG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0, nu=5.0)
    elif distribution == "student_t_grid":
        return StudentTNG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0, 
                          nu=[5, 10, 15, 20, 25])
    elif distribution == 'bernoulli':
        return BernoulliBeta(alpha0=1.0, beta0=1.0)
    elif distribution == 'binomial':
        return BinomialBeta(alpha0=1.0, beta0=1.0, n_trials=10)
    elif distribution == 'poisson':
        return PoissonGamma(alpha0=1.0, beta0=1.0) 
    elif distribution == 'gamma':
        return GammaGamma(alpha0=1.0, beta0=1.0)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def get_files(distribution, size=None):
    """Get list of data files for given distribution."""
    data_dir = Path(__file__).parent.parent / "data"
    pattern = f"data_{distribution}_n*.npy"
    files = sorted(data_dir.glob(pattern), reverse=True)
    if size is not None:
        files = [f for f in files if f"_n{size}_" in f.name]
    return files


def get_file_size(file_path):
    """Extract size from filename."""
    name = file_path.stem
    parts = name.split('_')
    for part in parts:
        if part.startswith('n'):
            return f"{int(part[1:]):,}"
    raise ValueError(f"Cannot extract size from filename: {file_path}, \
                     expected format 'data_<distr>_n<size>_seg<lambda>.npy'.")


#===============================================================================
# Printing Functions
#===============================================================================

def print_start(distribution, lambda_, runs, warmup, data_files):
    """Print benchmark header."""
    print()
    print("="*75)
    print(f"Fast-BOCPD Benchmark for {distribution}")
    print("="*75)
    print(f"Lambda: {lambda_} | Runs: {runs} | Warmup: {warmup}")
    print("Benchmarking on datasets:")
    for f in data_files:
        print(f" - {f.name}")
    print("="*75)
    print()
    

def print_results(online_results, batch_results, dataset_name):
    """Print compact results for both modes."""
    print(f"\n{dataset_name} ({online_results['n_obs']:,} obs):")
    print(f"  {'Mode':<10} {'Median':>10} {'Throughput':>14} {'CV%':>6}")
    print(f"  {'-'*45}")
    
    for results in [online_results, batch_results]:
        mode = results['mode'].capitalize()
        print(f"  {mode:<10} {results['median']:>9.4f}s {results['throughput']:>13,.0f}/s {results['cv_percent']:>5.1f}%")
    
    speedup = online_results['median'] / batch_results['median']
    print(f"  Batch speedup: {speedup:.2f}x")


def print_summary(all_results):
    """Print final summary table."""
    print(f"\n{'='*75}")
    print(f"SUMMARY")
    print(f"{'='*75}")
    print(f"{'Size':<10} {'Mode':<10} {'Median (s)':>12} {'Throughput':>15} {'CV%':>8}")
    print(f"{'-'*75}")
    
    sizes = sorted(all_results.keys(), key=lambda x: all_results[x]['online']['n_obs'])
    
    for size in sizes:
        online = all_results[size]['online']
        offline = all_results[size]['offline']
        
        print(f"{size:<10} {'Online':<10} {online['median']:>12.4f} {online['throughput']:>15,.0f} {online['cv_percent']:>7.1f}%")
        print(f"{'':<10} {'Offline':<10} {offline['median']:>12.4f} {offline['throughput']:>15,.0f} {offline['cv_percent']:>7.1f}%")

    print(f"{'='*75}")


#===============================================================================
# Argument Parsing
#===============================================================================

def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Fast-BOCPD library"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        choices=DISTRIBUTIONS,
        required=True,
        help="Type of observation model to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs to perform (default: 10).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warmup runs before timing (default: 2).",
    )
    parser.add_argument(
        "--lambda",
        type=int,
        default=150,
        dest="lambda_",
        help="Hazard function parameter (default: 150).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['online', 'batch'],
        default='online',
        help="Benchmark mode: 'online' for OnlineChangeDetector, \
             'batch' for BOCPD batch update (default: 'online').",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[1000, 10000, 100000],
        help="Run benchmark for specific dataset size only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()