"""
Benchmark Data Generator for Fast-BOCPD

Generates synthetic time series data with known changepoints for benchmarking.
Supports both univariate and multivariate data with various distributions.

Usage:
    python generate_data.py <distr> <n> <seg_len>

    distr:   gaussian, student_t_fixed, student_t_grid, bernoulli, binomial, poisson, gamma
    n:       Total number of observations
    seg_len: Length of each segment between changepoints
    
Examples:
    python generate_data.py gaussian 10000 150
    python generate_data.py gaussian 100000 200
"""

import sys
import numpy as np
from pathlib import Path

DISTRIBUTIONS = ['gaussian', "student_t_fixed", "student_t_grid", "bernoulli",
                 "binomial", "poisson", "gamma"]


def main():
    """Main entry point - parse args and generate data."""
    distr, n, seg_len = parse_args(sys.argv[1:])
    num_segments = n // seg_len
    
    if distr == 'gaussian':
        data = generate_gaussian(n, seg_len, num_segments)
    elif distr == "student_t_fixed":
        data = generate_student_t(n, seg_len, num_segments, df=5)
    elif distr == "student_t_grid":
        data = generate_student_t(n, seg_len, num_segments)
    elif distr == 'bernoulli':
        data = generate_binomial(n, seg_len, num_segments, trials=1)
    elif distr == 'binomial':
        data = generate_binomial(n, seg_len, num_segments, trials=10)
    elif distr == 'poisson':
        data = generate_poisson(n, seg_len, num_segments)
    elif distr == 'gamma':
        data = generate_gamma(n, seg_len, num_segments)
    else:
        raise ValueError(f"Unknown distribution: {distr}")
    
    write_data(data, distr, n, seg_len)

# ==============================================================================
# Data Generation Functions
# ==============================================================================

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
    np.random.seed(42)
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


def generate_student_t(n, seg_len, num_segments, df=None):
    """
    Generate Student-t data with shifting means.
    Creates segments with incrementing means (0, 5, 10, ..., 40, then wraps).
    Degrees of freedom and scale are randomized for each segment to add realism.
    
    Parameters:
    -----------
    n : int
        Total observations
    seg_len : int
        Segment length
    num_segments : int
        Number of segments
    df : int or None
        Degrees of freedom for Student-t distribution.
        If None, random df between 3 and 30 is chosen for each segment.
        
    Returns:
    --------
    np.ndarray : Time series with shape (n,)
    """
    np.random.seed(42)
    data = []
    mu = 0
    
    rand_df = False
    if df is None: rand_df = True

    for _ in range(num_segments):
        mu = (mu + 5) if mu <= 40 else 0
        df = np.random.randint(3, 30) if rand_df else df
        sigma = np.random.uniform(0.5, 2.5)
        standard_seg = np.random.standard_t(df=df, size=seg_len)
        segment = mu + sigma * standard_seg
        data.extend(segment)
        
    # Handle remaining observations
    remaining = n % seg_len
    if remaining > 0:
        mu = (mu + 5) if mu <= 40 else 0
        df = np.random.randint(3, 30) if rand_df else df
        sigma = np.random.uniform(0.5, 2.5)
        standard_seg = np.random.standard_t(df=df, size=remaining)
        segment = mu + sigma * standard_seg
        data.extend(segment)

    return np.array(data)


def generate_binomial(n, seg_len, num_segments, trials=10):
    """
    Generate Binomial data with shifting success probabilities.
    
    Parameters:
    -----------
    n : int
        Total observations
    seg_len : int
        Segment length
    num_segments : int
        Number of segments
    trials : int or None
        Number of trials for Binomial distribution. 
        If None, defaults to 10.
        If 1, behaves like Bernoulli.
    
    Returns:
    --------
    np.ndarray : Time series with shape (n,)
    """
    np.random.seed(42)
    data = []
    p = 0.1
    
    for _ in range(num_segments):
        p = p + 0.4 if p <= 0.5 else 0.1
        segment = np.random.binomial(n=trials, p=p, size=seg_len)
        data.extend(segment)
    
    # Handle remaining observations
    remaining = n % seg_len
    if remaining > 0:
        p = p + 0.4 if p <= 0.5 else 0.1
        segment = np.random.binomial(n=trials, p=p, size=remaining)
        data.extend(segment)
    
    return np.array(data)
        

def generate_poisson(n, seg_len, num_segments):
    """
    Generate Poisson data with shifting rates.
    
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
    np.random.seed(42)
    data = []
    lam = 2.0
    
    for _ in range(num_segments):
        lam = lam + 4.0 if lam <= 12.0 else 2.0
        segment = np.random.poisson(lam=lam, size=seg_len)
        data.extend(segment)
    
    # Handle remaining observations
    remaining = n % seg_len
    if remaining > 0:
        lam = lam + 4.0 if lam <= 12.0 else 2.0
        segment = np.random.poisson(lam=lam, size=remaining)
        data.extend(segment)

    return np.array(data)


def generate_gamma(n, seg_len, num_segments, shape=1.0):
    """
    Generate Gamma data with shifting scale/rate parameters.
    
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
    np.random.seed(42)
    data = []
    scale = 1.0
    
    for i in range(num_segments):
        scale = scale + 3.0 if scale <= 13.0 else 1.0
        segment = np.random.gamma(shape=shape, scale=scale, size=seg_len)
        data.extend(segment)

    # Handle remaining observations
    remaining = n % seg_len
    if remaining > 0:
        scale = scale + 3.0 if scale <= 13.0 else 1.0
        segment = np.random.gamma(shape=shape, scale=scale, size=remaining)
        data.extend(segment)

    return np.array(data)
    

# ==============================================================================
# Data Saving
# ==============================================================================

def write_data(data, distr, n, seg_len):
    """
    Save generated data to .npy file.
    
    Parameters:
    -----------
    data : np.ndarray
        Generated time series
    distr : str
        Distribution type
    n : int
        Number of observations
    seg_len : int
        Segment length
    """
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    filename = f"data_{distr}_n{n}_seg{seg_len}.npy"
    filepath = data_dir / filename
    
    np.save(filepath, data)
    print(f"Generated {len(data):,} observations")
    print(f"Saved to: {filepath}")


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args(args):
    """
    Parse and validate command-line arguments.
    
    Parameters:
    -----------
    args : list
        Command-line arguments (excluding script name)
        
    Returns:
    --------
    tuple : (distr, n, seg_len)
    """
    if len(args) != 3:
        print(__doc__)
        sys.exit(1)
    
    distr = args[0]
    if distr not in DISTRIBUTIONS:
        print(f"Error: distribution must be one of {DISTRIBUTIONS}")
        sys.exit(1)
    
    try:
        n = int(args[1])
        if n <= 0:
            raise ValueError
    except ValueError:
        print("Error: number of samples must be a positive integer")
        sys.exit(1)
    
    try:
        seg_len = int(args[2])
        if seg_len <= 0:
            raise ValueError
    except ValueError:
        print("Error: segment length must be a positive integer")
        sys.exit(1)

    return distr, n, seg_len


if __name__ == "__main__":
    main()
