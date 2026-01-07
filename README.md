# Fast BOCPD

[![PyPI version](https://badge.fury.io/py/fast-bocpd.svg)](https://pypi.org/project/fast-bocpd/)
[![Python](https://img.shields.io/pypi/pyversions/fast-bocpd.svg)](https://pypi.org/project/fast-bocpd/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue.svg)](https://fast-bocpd.readthedocs.io)

High-performance Bayesian Online Changepoint Detection (BOCPD) with a pure C backend and clean Python API. Achieve **10-1500x speedup** over existing implementations while maintaining numerical accuracy.

## Key Features

- **Exceptional Performance**: 10-1500x faster than competing implementations
- **Production Ready**: Extensively tested with 98 C unit tests and 359 Python test cases
- **Comprehensive Model Library**: 7 observation models covering continuous, discrete, and count data
- **Dual Processing Modes**: Online streaming and batch processing (Offline mode)
- **Minimal Dependencies**: Only requires NumPy
- **Well Documented**: Complete Sphinx documentation with examples and theory
- **Benchmarked**: Rigorous performance comparisons against 5 major implementations

## Performance Benchmarks

Comparative performance on 100,000 observations:

| Library | Language | Throughput | vs Fast-BOCPD |
|---------|----------|------------|---------------|
| **Fast-BOCPD** | **C** | **25,952 obs/s** | **1.0x (baseline)** |
| promised-ai | Rust | 915 obs/s | 28.3x slower |
| ruptures | Cython/C | 2,564 obs/s* | 10.1x slower* |
| dtolpin | Python | 163 obs/s | 159.2x slower |
| hildensia | PyTorch | 17 obs/s** | 1,525x slower** |

See [benchmarks/](benchmarks/) for detailed methodology and results.

## Installation

```bash
pip install fast-bocpd
```

The C extension is automatically compiled during installation. Requires a C compiler and NumPy.

### From Source

```bash
git clone https://github.com/TiaanViviers/Fast_BOCPD.git
cd Fast_BOCPD
pip install -e .
```

## Quick Start

### Basic Usage

```python
import numpy as np
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard

# Generate synthetic data with a changepoint
np.random.seed(42)
data = np.concatenate([
    np.random.normal(0, 1, 100),
    np.random.normal(5, 1, 100)
])

# Configure the model
obs_model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
hazard = ConstantHazard(lambda_=100)  # Expected run length = 100
bocpd = BOCPD(obs_model, hazard, max_run_length=200)

# Process data and detect changepoints
for t, x in enumerate(data):
    posterior_r, cp_prob = bocpd.update(x)
    if cp_prob > 0.5:
        print(f"Changepoint detected at t={t} (probability: {cp_prob:.3f})")
```

### Streaming Detection with OnlineChangeDetector

```python
from fast_bocpd import BOCPD, StudentTNG, ConstantHazard, OnlineChangeDetector

# Setup detector
obs_model = StudentTNG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
hazard = ConstantHazard(lambda_=100)
bocpd = BOCPD(obs_model, hazard)
detector = OnlineChangeDetector(bocpd, min_confidence=0.3)

# Process streaming data
for t, observation in enumerate(data_stream):
    cp = detector.update(observation)
    
    if cp:
        print(f"Changepoint at t={cp.index}")
        print(f"Previous segment: {cp.prev_run_length} observations")
        print(f"Confidence: {cp.confidence:.3f}")

# Retrieve all detected changepoints
changepoints = detector.get_changepoints()
segments = detector.get_segments()
```

### Batch Processing

```python
# Process entire dataset at once
cp_probs = bocpd.batch_update(data)

# Find changepoint locations
changepoints = np.where(cp_probs > 0.5)[0]
print(f"Changepoints detected at: {changepoints}")
```

## Available Models

### Observation Models

**Continuous Data:**
- `GaussianNIG`: Gaussian likelihood with Normal-Inverse-Gamma conjugate prior
- `StudentTNG`: Student-t likelihood with Normal-Gamma conjugate prior (robust to outliers)
- `StudentTNGGrid`: Grid-based Student-t for faster inference with controlled precision

**Discrete Data:**
- `BernoulliBeta`: Bernoulli likelihood (binary outcomes) with Beta conjugate prior
- `BinomialBeta`: Binomial likelihood (count of successes) with Beta conjugate prior

**Count Data:**
- `PoissonGamma`: Poisson likelihood (rare events) with Gamma conjugate prior
- `GammaGamma`: Gamma likelihood with Gamma conjugate prior (positive continuous data)

### Hazard Functions

- `ConstantHazard`: Constant changepoint probability (geometric prior on run lengths)

All models feature efficient conjugate Bayesian updates implemented in optimized C code.

## Documentation

Complete documentation is available at [https://fast-bocpd.readthedocs.io](https://fast-bocpd.readthedocs.io)

**Documentation includes:**
- Getting Started Guide
- User Guide with detailed model descriptions
- API Reference
- Mathematical Theory and Derivations
- Example Notebooks
- Architecture and Implementation Details
- Benchmark Results and Methodology

**Example Notebooks:**
- [01_quickstart.ipynb](examples/01_quickstart.ipynb) - Basic usage and concepts
- [02_online_vs_batch.ipynb](examples/02_online_vs_batch.ipynb) - Processing mode comparison
- [03_understanding_outputs.ipynb](examples/03_understanding_outputs.ipynb) - Interpreting results
- [04_1_univariate_models.ipynb](examples/04_1_univariate_models.ipynb) - Continuous data models
- [04_2_multivariate_models.ipynb](examples/04_2_multivariate_models.ipynb) - Discrete and count models
- [04_3_hazard_and_run_length.ipynb](examples/04_3_hazard_and_run_length.ipynb) - Prior configuration
- [05_advanced_features.ipynb](examples/05_advanced_features.ipynb) - Advanced usage patterns
- [06_real_world_example.ipynb](examples/06_real_world_example.ipynb) - Financial time series analysis

## Testing

Fast-BOCPD is extensively tested to ensure correctness and reliability:

- **98 C unit tests**: Core algorithm and model implementations
- **359 Python tests**: API, integration, and end-to-end workflows
- **Numerical validation**: Results verified against reference implementations
- **Edge case coverage**: Boundary conditions and error handling

Run the test suite:

```bash
# Run all tests
make test

# C unit tests only
make test-c

# Python tests only
make test-python
```

## Development

### Build from Source

```bash
git clone https://github.com/TiaanViviers/Fast_BOCPD.git
cd Fast_BOCPD

# Install in development mode
pip install -e .

# Build C library manually (optional)
make lib

# Run tests
make test
```

### Project Structure

```
Fast_BOCPD/
├── fast_bocpd/           # Python package
│   ├── core.py           # Main BOCPD class
│   ├── models.py         # Observation models
│   ├── hazard.py         # Hazard functions
│   ├── utils.py          # Helper utilities
│   ├── _bindings.py      # C extension bindings
│   └── _c/               # C implementation
│       ├── bocpd_core.c
│       ├── gaussian_nig.c
│       ├── student_t_ng.c
│       └── ...
├── tests/                # Test suite
│   ├── python/           # Python integration tests
│   └── c_tests/          # C unit tests
├── examples/             # Jupyter notebooks
├── docs/                 # Sphinx documentation
├── benchmarks/           # Performance comparisons
└── Makefile              # Build system
```

## Algorithm

Fast-BOCPD implements the Bayesian Online Changepoint Detection algorithm (Adams & MacKay, 2007). The algorithm:

1. Maintains a distribution over run lengths (time since last changepoint)
2. Updates beliefs online as new data arrives
3. Provides probabilistic changepoint detection without threshold tuning
4. Supports arbitrary observation models via conjugate Bayesian updates

The implementation uses dynamic programming with efficient log-space computations to maintain numerical stability.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Links

- **PyPI**: https://pypi.org/project/fast-bocpd/
- **Documentation**: https://fast-bocpd.readthedocs.io
- **Source Code**: https://github.com/TiaanViviers/Fast_BOCPD
- **Issue Tracker**: https://github.com/TiaanViviers/Fast_BOCPD/issues

## Acknowledgments

This implementation is based on the foundational work by Adams and MacKay (2007). Performance optimizations and the C backend were developed to enable real-time changepoint detection in production systems
