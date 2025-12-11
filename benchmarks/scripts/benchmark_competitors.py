import argparse
import numpy as np
from pathlib import Path

from benchmark_dtolpin_bocd import benchmark_dtolpin_bocd
from benchmark_ruptures import benchmark_ruptures


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
    print(f"\n{'='*75}")
    print(f"SUMMARY")
    print(f"{'='*75}")
    print(f"{'Size':<10} {'Mode':<10} {'Median (s)':>12} {'Throughput':>15} {'CV%':>8}")
    print(f"{'-'*75}")
    
    # Sort by size (extract n_obs from first available mode)
    def get_n_obs(size_key):
        result = all_results[size_key]
        if 'online' in result:
            return result['online']['n_obs']
        elif 'offline' in result:
            return result['offline']['n_obs']
        return 0
    
    sizes = sorted(all_results.keys(), key=get_n_obs)
    
    for size in sizes:
        result = all_results[size]
        
        # Print online results if available
        if 'online' in result:
            online = result['online']
            print(f"{size:<10} {'Online':<10} {online['median']:>12.4f} {online['throughput']:>15,.0f} {online['cv_percent']:>7.1f}%")
        
        # Print offline results if available
        if 'offline' in result:
            offline = result['offline']
            size_label = '' if 'online' in result else size  # Only show size once
            print(f"{size_label:<10} {'Offline':<10} {offline['median']:>12.4f} {offline['throughput']:>15,.0f} {offline['cv_percent']:>7.1f}%")

    print(f"{'='*75}")


#===============================================================================
# Argument Parsing
#===============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark competitor implementations"
    )
    parser.add_argument(
        "--lib",
        choices=['dtolpin', 'ruptures'],
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
    return parser.parse_args()


if __name__ == "__main__":
    main()