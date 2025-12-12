import argparse
import numpy as np
from pathlib import Path

from benchmark_dtolpin_bocd import benchmark_dtolpin_bocd
from benchmark_ruptures import benchmark_ruptures
from benchmark_hildensia import benchmark_hildensia
from benchmark_promised_ai import benchmark_promised_ai


def main():
    args = parse_args()

    if args.lib == 'dtolpin':
        print_start("dtolpin/bocd", args.lambda_, args.runs, args.warmup_runs)
        dtolpin_results = run_dtolpin_benchmark(args)
        print_summary(dtolpin_results)

    elif args.lib == 'ruptures':
        print_start("ruptures", args.lambda_, args.runs, args.warmup_runs)
        ruptures_results = run_ruptures_benchmark(args)
        print_summary(ruptures_results)
    
    elif args.lib == 'hildensia':
        print_start("hildensia/bayesian_changepoint_detection", args.lambda_, args.runs, args.warmup_runs, args.device)
        hildensia_results = run_hildensia_benchmark(args)
        print_summary(hildensia_results)
    
    elif args.lib == 'promised-ai':
        print_start("promised-ai/changepoint", args.lambda_, args.runs, args.warmup_runs)
        promised_ai_results = run_promised_ai_benchmark(args)
        print_summary(promised_ai_results)
    
    else:
        print_start("dtolpin/bocd", args.lambda_, args.runs, args.warmup_runs)
        dtolpin_results = run_dtolpin_benchmark(args)
        print_summary(dtolpin_results)
        
        print_start("ruptures", args.lambda_, args.runs, args.warmup_runs)
        ruptures_results = run_ruptures_benchmark(args)
        print_summary(ruptures_results)
        
        print_start("hildensia/bayesian_changepoint_detection", args.lambda_, args.runs, args.warmup_runs, args.device)
        hildensia_results = run_hildensia_benchmark(args)
        print_summary(hildensia_results)
        
        print_start("promised-ai/changepoint", args.lambda_, args.runs, args.warmup_runs)
        promised_ai_results = run_promised_ai_benchmark(args)
        print_summary(promised_ai_results)


#===============================================================================
# Benchmarking Logic
#===============================================================================
def run_dtolpin_benchmark(args):
    """Run benchmark for dtolpin/bocd."""
    files = get_files('student_t_fixed', args.size)
    
    results = {}
    for file in files:
        data = np.load(file)
        online_results = benchmark_dtolpin_bocd(data)
        name = get_file_size(file)
        results[name] = {'online': online_results}
        print("done with size: ", name)

    return results


def run_ruptures_benchmark(args):
    """Run benchmark for ruptures."""
    files = get_files('gaussian', args.size)
    
    results = {}
    for file in files:
        data = np.load(file)
        offline_results = benchmark_ruptures(
            data, 
            distribution='gaussian',
            mode='offline',
            lambda_=args.lambda_,
            runs=args.runs,
            warmup=args.warmup_runs
        )
        name = get_file_size(file)
        results[name] = {'offline': offline_results}
        print("done with size: ", name)

    return results


def run_hildensia_benchmark(args):
    """Run benchmark for hildensia/bayesian_changepoint_detection.
    
    Note: Online mode has O(n²) complexity on CPU - only runs on n=1000.
          GPU mode can handle larger sizes. Offline mode runs on all sizes.
    """
    files = get_files('gaussian', args.size)
    device = args.device
    
    results = {}
    for file in files:
        data = np.load(file)
        name = get_file_size(file)
        n_obs = len(data)
        
        results[name] = {}
        
        # Run online mode: CPU-only for n <= 1000, GPU for all sizes
        if n_obs > 1000:
            print(f"Skipping Hildensia for n={name} (O(n²) complexity, would take ~{(n_obs/100)**2 * 0.6/60:.1f} minutes)")
            results[name]['online'] = None
        else:
            print(f"Running Hildensia online mode for n={name} on {device}...")
            online_results = benchmark_hildensia(
                data,
                distribution='gaussian',
                mode='online',
                lambda_=args.lambda_,
                runs=args.runs,
                warmup=args.warmup_runs,
                device=device
            )
            results[name]['online'] = online_results
       
            print(f"Running Hildensia offline mode for n={name} on {device}...")
            offline_results = benchmark_hildensia(
                data,
                distribution='gaussian',
                mode='offline',
                lambda_=args.lambda_,
                runs=args.runs,
                warmup=args.warmup_runs,
                device=device
            )
            results[name]['offline'] = offline_results
            
            print(f"Completed benchmarking n={name}")

    return results


def run_promised_ai_benchmark(args):
    """Run benchmark for promised-ai/changepoint.
    
    Note: This Rust-based library only supports online mode (no offline).
          It should be very fast due to Rust implementation.
          
    Supported distributions:
        - Gaussian (NormalGamma prior)
        - Bernoulli (BetaBernoulli prior)
        - Poisson (PoissonGamma prior)
    """
    # Only benchmark distributions that promised-ai actually supports
    # Don't try to fit Gaussian models to Student-t data - that's unfair!
    supported_distributions = ['gaussian', 'bernoulli', 'poisson']
    
    results = {}
    
    # Benchmark each supported distribution
    for distribution in supported_distributions:
        files = get_files(distribution, args.size)
        if not files:
            continue
            
        for file in files:
            data = np.load(file)
            name = f"{distribution}_{get_file_size(file)}"
            
            online_results = benchmark_promised_ai(
                data,
                distribution=distribution,  # Use distribution name directly
                lambda_=args.lambda_,
                runs=args.runs,
                warmup=args.warmup_runs
            )
            
            results[name] = {'online': online_results, 'offline': None}
    
    return results


#===============================================================================
# Setup Helper Functions
#===============================================================================

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

def print_start(lib, lambda_, runs, warmup):
    """Print benchmark header."""
    print()
    print("="*75)
    print(f"Benchmark for the {lib} library")
    print("="*75)
    print(f"Lambda: {lambda_} | Runs: {runs} | Warmup: {warmup}")
    print("="*75)
    print()


def print_summary(all_results):
    """Print final summary table."""
    # Extract device if available (for Hildensia)
    device_info = ""
    for size_results in all_results.values():
        for mode_result in size_results.values():
            if mode_result and 'device' in mode_result:
                device_info = f" [Device: {mode_result['device'].upper()}]"
                break
        if device_info:
            break
    
    print(f"\n{'='*95}")
    print(f"SUMMARY{device_info}")
    print(f"{'='*95}")
    print(f"{'Dataset':<30} {'Mode':<10} {'Median (s)':>12} {'Throughput':>15} {'CV%':>8}")
    print(f"{'-'*95}")
    
    # Sort by size (extract n_obs from first available mode)
    def get_n_obs(size_key):
        result = all_results[size_key]
        # Handle None values for skipped benchmarks
        if result.get('online') is not None:
            return result['online']['n_obs']
        elif result.get('offline') is not None:
            return result['offline']['n_obs']
        return 0
    
    sizes = sorted(all_results.keys(), key=get_n_obs)
    
    for size in sizes:
        result = all_results[size]
        
        # Print online results if available and not None
        if 'online' in result and result['online'] is not None:
            online = result['online']
            print(f"{size:<30} {'Online':<10} {online['median']:>12.4f} {online['throughput']:>15,.0f} {online['cv_percent']:>7.1f}%")
        elif 'online' in result and result['online'] is None:
            print(f"{size:<30} {'Online':<10} {'SKIPPED':>12} {'(O(n²) too slow)':>15} {'-':>8}")
        
        # Print offline results if available and not None
        if 'offline' in result and result['offline'] is not None:
            offline = result['offline']
            size_label = '' if ('online' in result and result['online'] is not None) else size  # Only show dataset once
            print(f"{size_label:<30} {'Offline':<10} {offline['median']:>12.4f} {offline['throughput']:>15,.0f} {offline['cv_percent']:>7.1f}%")

    print(f"{'='*95}")


#===============================================================================
# Argument Parsing
#===============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark competitor implementations"
    )
    parser.add_argument(
        "--lib",
        choices=['dtolpin', 'ruptures', 'hildensia', 'promised-ai'],
        help="Competitor library to benchmark.",
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
        help="Number of warmup runs to perform (default: 2).",
    )
    parser.add_argument(
        "--lambda",
        type=int,
        default=150,
        dest="lambda_",
        help="Expected run length parameter (default: 150).",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[1000, 10000, 100000],
        help="Run benchmark for specific dataset size only.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on: 'cpu' or 'cuda' (default: cpu). Only applies to Hildensia.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
