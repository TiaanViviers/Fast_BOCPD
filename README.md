# Fast BOCPD

Fast, C-based Bayesian Online Changepoint Detection for Python.

## Features

- **Pure C implementation** for maximum performance
- **Clean Python API** via ctypes (zero dependencies beyond NumPy)
- **Online and batch processing** modes
- **Model-agnostic architecture** - easy to extend with new models

## Installation

### From PyPI (when published)

```bash
pip install fast-bocpd
```

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/Fast_BOCPD.git
cd Fast_BOCPD

# Install with pip (automatically compiles C code)
pip install -e .
```

### Development mode

If you're developing and want to manually rebuild the C library:

```bash
cd fast_bocpd/_c
make clean
make lib
```

**Note:** The C library will be automatically compiled during `pip install`. You only need manual compilation if you're actively developing the C code.

## Quick Start

```python
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard
import numpy as np

# Set up the model
obs_model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
hazard = ConstantHazard(lambda_=100)  # Expected run length = 100
bocpd = BOCPD(obs_model, hazard, max_run_length=200)

# Online mode - process one observation at a time
for x in data_stream:
    posterior_r, cp_prob = bocpd.update(x)
    if cp_prob > 0.5:
        print(f"Changepoint detected! Probability: {cp_prob:.3f}")

# Offline mode - process all data at once
cp_probs = bocpd.batch_update(data_array)
```

### Using OnlineChangeDetector (Recommended for Streaming)

```python
from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard, OnlineChangeDetector

# Setup
bocpd = BOCPD(GaussianNIG(...), ConstantHazard(100))
detector = OnlineChangeDetector(bocpd, min_confidence=0.3)

# Process streaming data
for t, observation in enumerate(data_stream):
    cp = detector.update(observation, metadata=f"sample_{t}")
    
    if cp:
        print(f"Changepoint at t={cp.index}: previous segment lasted {cp.prev_run_length} steps")
    
    # Check current run length
    run_length = detector.get_current_run_length()
    
# Get all detected changepoints and segments
changepoints = detector.get_changepoints()
segments = detector.get_segments()
```

## Project Structure

```
Fast_BOCPD/
├── fast_bocpd/
│   ├── __init__.py              # Public API
│   ├── bocpd_accelerated.py     # Python wrapper for C implementation
│   ├── _bindings.py             # ctypes bindings to C library
│   ├── hazard.py                # Hazard function parameter wrappers
│   ├── obs_models/              # Observation model parameter wrappers
│   │   ├── __init__.py
│   │   └── gaussian_nig.py
│   └── _c/                      # C implementation
│       ├── bocpd_core.c/h       # Main BOCPD algorithm
│       ├── gaussian_nig.c/h     # GaussianNIG model
│       ├── hazard.c/h           # Hazard functions
│       ├── test_modules.c       # C unit tests
│       ├── Makefile
│       └── libbocpd.so          # Compiled shared library
└── test_bocpd.ipynb             # Example notebook

```

## Available Models

### Observation Models
- **GaussianNIG**: 1D Gaussian with Normal-Inverse-Gamma prior

### Hazard Functions
- **ConstantHazard**: Constant changepoint probability

## Development

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Fast_BOCPD.git
cd Fast_BOCPD

# Run all tests
make test

# Or run separately
make test-c       # C unit tests
make test-python  # Python integration tests

# Build shared library for development
make lib

# Clean build artifacts
make clean
```

### Development Workflow

1. **Edit C code**: Modify files in `fast_bocpd/_c/`
2. **Test changes**: Run `make test`
3. **Install for Python**: Run `pip install -e .`

### Test Structure

```
tests/
├── python/                    # Python integration tests
│   ├── test_api.py            # Public API & user interface
│   ├── test_models.py         # Model parameter validation
│   ├── test_hazard.py         # Hazard parameter validation
│   └── test_integration.py    # End-to-end integration tests
└── c_tests/                   # C unit tests
    ├── test_utils.h           # Test utilities & macros
    ├── test_gaussian_nig.c    # GaussianNIG tests
    ├── test_hazard.c          # Hazard function tests
    ├── test_bocpd_core.c      # BOCPD algorithm tests
    └── test_runner.c          # Test suite runner
```

### Build System

All build artifacts are placed in `build/`:
```
build/
├── lib/          # Compiled libraries
└── obj/          # Object files
```

The root `Makefile` provides:
- `make lib` - Build shared library
- `make test` - Run all tests
- `make clean` - Remove all artifacts

### Project Structure

```
Fast_BOCPD/
├── fast_bocpd/                  # Implementation (shipped to users)
│   ├── __init__.py              # Public API
│   ├── bocpd_accelerated.py     # Python wrapper
│   ├── _bindings.py             # ctypes bindings
│   ├── hazard.py                # Hazard function wrappers
│   ├── models.py                # Observation model wrappers
│   └── _c/                      # C implementation
│       ├── bocpd_core.c/h       # Main BOCPD algorithm
│       ├── gaussian_nig.c/h     # GaussianNIG model
│       ├── hazard.c/h           # Hazard functions
│       └── Makefile             # For dev builds only
├── tests/                       # Testing (not shipped)
│   ├── test_bocpd.py            # Python unit tests
│   └── c_tests/
│       ├── test_modules.c       # C unit tests
│       └── Makefile
├── setup.py
├── pyproject.toml
└── README.md
```

## Performance

The C implementation provides significant speedup over pure Python implementations while maintaining numerical accuracy.

## License

MIT License

## Citation

If you use this library, please cite:
```
Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection.
arXiv preprint arXiv:0710.3742.
```
