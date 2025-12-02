"""
Benchmark Data Generator for Fast-BOCPD

Generates synthetic time series data with known changepoints for benchmarking.
Supports both univariate and multivariate data with various distributions.

Usage:
    python generate_data.py <dim> <distr> <n> <seg_len>
    
    dim:     'uv' (univariate) or 'mv' (multivariate)
    distr:   'gaussian' (more distributions coming soon)
    n:       Total number of observations
    seg_len: Length of each segment between changepoints
    
Examples:
    python generate_data.py uv gaussian 10000 150
    python generate_data.py uv gaussian 100000 200
"""

import sys
import numpy as np
from pathlib import Path


def main():
    """Main entry point - parse args and generate data."""
    dim, distr, n, seg_len = parse_args(sys.argv[1:])
    
    if dim == 'uv':
        data = generate_univariate(distr, n, seg_len)
    elif dim == 'mv':
        print("Multivariate data generation not yet implemented")
        sys.exit(1)
    
    write_data(data, dim, distr, n, seg_len)


def generate_univariate(distr, n, seg_len):
    """
    Generate univariate time series with changepoints.
    
    Parameters:
    -----------
    distr : str
        Distribution type ('gaussian')
    n : int
        Total number of observations
    seg_len : int
        Target length of each segment
        
    Returns:
    --------
    np.ndarray : Generated time series data
    """
    num_segments = n // seg_len
    
    if distr == 'gaussian':
        return generate_gaussian(n, seg_len, num_segments)
    else:
        raise ValueError(f"Unknown distribution: {distr}")


def generate_gaussian(n, seg_len, num_segments):
    """
    Generate Gaussian data with shifting means.
    
    Creates segments with incrementing means (0, 5, 10, ..., 40, then wraps).
    Variance is randomized for each segment to add realism.
    
    Parameters:
    -----------
    n : int
        Total observations
    seg_len : int
        Segment length
    num_segments : int
        Number of segments
        
    Returns:
    --------
    np.ndarray : Time series with shape (n,)
    """
    np.random.seed(42)  # Reproducible data
    data = []
    mu = 0
    
    # Generate full segments
    for _ in range(num_segments):
        mu = (mu + 5) if mu <= 40 else 0
        sigma = np.random.uniform(0.5, 2.5)
        segment = np.random.normal(loc=mu, scale=sigma, size=seg_len)
        data.extend(segment)
    
    # Handle remaining observations
    remaining = n % seg_len
    if remaining > 0:
        mu = (mu + 5) if mu <= 40 else 0
        sigma = np.random.uniform(0.5, 2.5)
        segment = np.random.normal(loc=mu, scale=sigma, size=remaining)
        data.extend(segment)
    
    return np.array(data)


def write_data(data, dim, distr, n, seg_len):
    """
    Save generated data to .npy file.
    
    Parameters:
    -----------
    data : np.ndarray
        Generated time series
    dim : str
        Dimension type ('uv' or 'mv')
    distr : str
        Distribution type
    n : int
        Number of observations
    seg_len : int
        Segment length
    """
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    filename = f"data_{dim}_{distr}_n{n}_seg{seg_len}.npy"
    filepath = data_dir / filename
    
    np.save(filepath, data)
    print(f"Generated {len(data):,} observations")
    print(f"Saved to: {filepath}")


def parse_args(args):
    """
    Parse and validate command-line arguments.
    
    Parameters:
    -----------
    args : list
        Command-line arguments (excluding script name)
        
    Returns:
    --------
    tuple : (dim, distr, n, seg_len)
    """
    if len(args) != 4:
        print(__doc__)
        sys.exit(1)
    
    dim = args[0]
    if dim not in ['uv', 'mv']:
        print("Error: dimension must be 'uv' (univariate) or 'mv' (multivariate)")
        sys.exit(1)
    
    distr = args[1]
    if distr not in ['gaussian']:
        print("Error: distribution must be 'gaussian'")
        print("(More distributions coming soon)")
        sys.exit(1)
    
    try:
        n = int(args[2])
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Error: number of samples must be a positive integer")
        sys.exit(1)
    
    try:
        seg_len = int(args[3])
        if seg_len <= 0:
            raise ValueError
    except ValueError:
        print("Error: segment length must be a positive integer")
        sys.exit(1)
    
    return dim, distr, n, seg_len


if __name__ == "__main__":
    main()
