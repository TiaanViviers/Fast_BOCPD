"""
Univariate Gaussian Benchmark Script for Fast-BOCPD

Benchmarks both online and batch modes on Gaussian data with varying sizes.

Usage:
    python test_uv_gaussian.py [--runs N] [--warmup N] [--lambda L]
"""

import sys
import argparse
import time
from pathlib import Path
from statistics import median
import numpy as np

# Add parent directory to path to import fast_bocpd
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard, OnlineChangeDetector


def benchmark_single_run(data, lambda_=150, max_run_length=450, min_confidence=0.3, mode='online'):
    """Run BOCPD detection once and return elapsed time."""
    model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
    hazard = ConstantHazard(lambda_=lambda_)
    bocpd = BOCPD(model, hazard, max_run_length=max_run_length)
    
    start = time.perf_counter()
    
    if mode == 'online':
        detector = OnlineChangeDetector(bocpd, min_confidence=min_confidence)
        for x in data:
            detector.update(x)
        _ = detector.get_changepoints()
    else:  # batch / offline mode
        _ = bocpd.batch_update(data)
    
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_dataset(data, lambda_=150, n_runs=10, warmup_runs=2, mode='online'):
    """Benchmark BOCPD on dataset with multiple runs."""
    max_run_length = 3 * lambda_
    min_confidence = 0.3
    
    # Warmup
    for _ in range(warmup_runs):
        _ = benchmark_single_run(data, lambda_, max_run_length, min_confidence, mode)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        elapsed = benchmark_single_run(data, lambda_, max_run_length, min_confidence, mode)
        times.append(elapsed)
    
    # Statistics
    results = {
        'n_obs': len(data),
        'lambda': lambda_,
        'mode': mode,
        'median': median(times),
        'mean': np.mean(times),
        'std': np.std(times, ddof=1),
        'cv_percent': (np.std(times, ddof=1) / np.mean(times)) * 100,
        'throughput': len(data) / median(times)
    }
    
    return results


def print_results(online_results, batch_results, dataset_name):
    """Print compact results for both modes."""
    print(f"\n{dataset_name} ({online_results['n_obs']:,} obs):")
    print(f"  {'Mode':<10} {'Median':>10} {'Throughput':>14} {'CV%':>6}")
    print(f"  {'-'*45}")
    
    for results in [online_results, batch_results]:
        mode = results['mode'].capitalize()
        print(f"  {mode:<10} {results['median']:>9.4f}s {results['throughput']:>13,.0f}/s {results['cv_percent']:>5.1f}%")
    
    # Speedup
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
        batch = all_results[size]['batch']
        
        print(f"{size:<10} {'Online':<10} {online['median']:>12.4f} {online['throughput']:>15,.0f} {online['cv_percent']:>7.1f}%")
        print(f"{'':<10} {'Batch':<10} {batch['median']:>12.4f} {batch['throughput']:>15,.0f} {batch['cv_percent']:>7.1f}%")
    
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Fast-BOCPD on Gaussian data')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs (default: 10)')
    parser.add_argument('--warmup', type=int, default=2, help='Warmup runs (default: 2)')
    parser.add_argument('--lambda', type=int, default=150, dest='lambda_', help='Expected run length (default: 150)')
    parser.add_argument('--size', type=int, choices=[1000, 10000, 100000], help='Run specific size only')
    args = parser.parse_args()
    
    # Use absolute path to avoid cProfile issues
    data_dir = Path(__file__).resolve().parent.parent / 'data'
    datasets = {
        '1k':    data_dir / 'data_uv_gaussian_n1000_seg150.npy',
        '10k':   data_dir / 'data_uv_gaussian_n10000_seg150.npy',
        '100k':  data_dir / 'data_uv_gaussian_n100000_seg150.npy',
    }
    
    if args.size:
        size_map = {1000: '1k', 10000: '10k', 100000: '100k'}
        key = size_map[args.size]
        datasets = {key: datasets[key]}
    
    
    print("="*75)
    print("Fast-BOCPD Benchmark: Univariate Gaussian")
    print("="*75)
    print(f"Lambda: {args.lambda_} | Runs: {args.runs} | Warmup: {args.warmup}")
    print(f"Datasets: {', '.join(datasets.keys())}")
    print("="*75)
    
    
    all_results = {}
    for name, filepath in datasets.items():
        if not filepath.exists():
            print(f"\nERROR: {filepath.name} not found")
            print(f"Generate with: python generate_data.py uv gaussian <n> 150")
            continue
        
        data = np.load(filepath)
        print(f"\nBenchmarking {name}...")
        
        # Run both modes
        online_results = benchmark_dataset(data, args.lambda_, args.runs, args.warmup, 'online')
        batch_results = benchmark_dataset(data, args.lambda_, args.runs, args.warmup, 'batch')
        
        all_results[name] = {'online': online_results, 'batch': batch_results}
        print_results(online_results, batch_results, name)
    
    
    if len(all_results) > 1:
        print_summary(all_results)


if __name__ == '__main__':
    main()
