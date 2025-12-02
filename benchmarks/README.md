# Fast-BOCPD Benchmarks

This directory contains benchmarking scripts and datasets for evaluating the performance of Fast-BOCPD.

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── data/                        # Generated benchmark datasets
│   ├── data_uv_gaussian_n1000_seg150.npy
│   ├── data_uv_gaussian_n10000_seg150.npy
│   └── data_uv_gaussian_n100000_seg150.npy
└── scripts/
    ├── generate_data.py         # Data generation utility
    └── test_uv_gaussian.py      # Univariate Gaussian benchmark
```

## Quick Start

### 1. Generate Data (if not already present)

```bash
cd benchmarks/scripts

# Generate 1k observations
python generate_data.py uv gaussian 1000 150

# Generate 10k observations
python generate_data.py uv gaussian 10000 150

# Generate 100k observations
python generate_data.py uv gaussian 100000 150
```

### 2. Run Benchmark

```bash
# Run all datasets with default settings (10 runs, 2 warmup)
python test_uv_gaussian.py

# More rigorous (20 runs, 3 warmup)
python test_uv_gaussian.py --runs 20 --warmup 3

# Benchmark specific size only
python test_uv_gaussian.py --size 10000

# Test with different lambda
python test_uv_gaussian.py --lambda 100

# Quiet mode (less output)
python test_uv_gaussian.py --quiet
```

## Benchmark Methodology

### Statistical Approach

The benchmark uses proper statistical methodology to ensure reliable measurements:

1. **Warmup Runs (default: 2)**: Discarded runs to warm up CPU cache and JIT compilation
2. **Timed Runs (default: 10)**: Actual measurements used for statistics
3. **High-Resolution Timing**: Uses `time.perf_counter()` for accurate measurements
4. **Multiple Statistics**: Reports median, mean, std, min, max, and coefficient of variation

### Why These Choices?

- **Median over Mean**: More robust to outliers (e.g., OS interrupts)
- **Warmup Runs**: First runs are always slower due to cold cache
- **Multiple Runs**: Statistical confidence requires repeated measurements
- **Coefficient of Variation (CV)**: Indicates measurement reliability
  - CV < 5%: Excellent consistency
  - CV 5-10%: Good consistency
  - CV > 10%: High variability (warning displayed)

### Interpreting Results

Example output:
```
======================================================================
RESULTS: 10k
======================================================================
  Dataset size:         10,000 observations
  Lambda:                  150
  Median time:          1.2345 s
  Mean time:            1.2367 s
  Std deviation:        0.0123 s
  Min time:             1.2201 s
  Max time:             1.2589 s
  Throughput:           8,097 obs/sec
  Variability (CV):      1.00 %
======================================================================
```

**Key metrics:**
- **Median time**: Primary performance indicator (robust to outliers)
- **Throughput**: Observations processed per second
- **Variability (CV)**: Lower is better (<5% ideal)

### Scaling Analysis

When benchmarking multiple sizes, the script automatically analyzes algorithmic complexity:

```
SCALING ANALYSIS:
  1k → 10k:
    Size increase:        10.0x
    Time increase:        10.23x
    Expected (O(n)):      10.0x
    Scaling:             ✓ Near-linear
```

**Expected behavior:**
- **O(n)**: Time increases linearly with size (ideal)
- **O(n log n)**: Time increases slightly faster than linear
- **O(n²)**: Time increases quadratically (poor scaling)

## Performance Tracking

### Baseline (Current Implementation)

Record your baseline performance here before making optimizations:

| Date | Version | 1k (median) | 10k (median) | 100k (median) | Notes |
|------|---------|-------------|--------------|---------------|-------|
| 2025-01-XX | v0.1.0 | X.XXXXs | X.XXXXs | XX.XXXXs | Initial baseline |

### Optimization History

Track improvements over time:

| Date | Version | 1k | 10k | 100k | Speedup | Changes |
|------|---------|----|----|------|---------|---------|
| TBD | v0.2.0 | - | - | - | - | Compiler flags (-O3, -march=native) |
| TBD | v0.3.0 | - | - | - | - | SIMD vectorization |

## Best Practices

### For Reliable Measurements

1. **Close other programs**: Minimize background processes
2. **Use AC power**: Laptop power management can affect results
3. **Consistent environment**: Run benchmarks under similar conditions
4. **Multiple runs**: Use `--runs 20` for production measurements
5. **Check variability**: CV > 10% indicates unreliable results

### For Fair Comparisons

1. **Same data**: Always use the same generated datasets
2. **Same parameters**: Keep lambda and other settings constant
3. **Same hardware**: Don't compare across different machines
4. **Record everything**: Note CPU model, RAM, OS, Python version

### Before/After Optimization

```bash
# 1. Establish baseline
python test_uv_gaussian.py --runs 20 > baseline_results.txt

# 2. Make optimization changes
# ... edit C code, update setup.py, etc. ...

# 3. Rebuild
cd ../../
pip install -e . --force-reinstall

# 4. Re-benchmark
cd benchmarks/scripts
python test_uv_gaussian.py --runs 20 > optimized_results.txt

# 5. Compare
diff baseline_results.txt optimized_results.txt
```

## Advanced Usage

### Custom Dataset

```python
import numpy as np

# Create your own test data
data = np.random.randn(50000)
np.save('../data/custom_data.npy', data)

# Modify test_uv_gaussian.py to include it
```

### Profiling Integration

```bash
# Run with profiler
python -m cProfile -o profile.stats test_uv_gaussian.py --size 10000

# Analyze with snakeviz
snakeviz profile.stats
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory_profiler

# Profile memory usage
python -m memory_profiler test_uv_gaussian.py --size 100000
```

## Troubleshooting

### High Variability (CV > 10%)

**Problem**: Inconsistent timing measurements

**Solutions**:
- Close browser, IDEs, and other heavy programs
- Disable CPU frequency scaling (if possible)
- Increase `--runs` to 20 or 30
- Run benchmark multiple times and compare

### Unexpectedly Slow

**Problem**: Performance worse than expected

**Check**:
- Python interpreter (CPython vs PyPy)
- NumPy version and BLAS backend
- Compiler used for C extension (GCC, Clang, MSVC)
- Debug vs Release build

### Dataset Not Found

**Problem**: `ERROR: Dataset not found`

**Solution**:
```bash
cd scripts
python generate_data.py uv gaussian <n> 150
```

## Future Enhancements

- [ ] Multivariate benchmarks
- [ ] External library comparisons (bayesian-online-changepoint, ruptures)
- [ ] Memory usage tracking
- [ ] Automatic visualization (plots)
- [ ] CI/CD integration for regression testing
- [ ] Different distributions (Student-t, Exponential)

## Contributing

When adding new benchmarks:

1. Follow existing naming conventions
2. Include proper docstrings
3. Use statistical best practices (warmup, multiple runs)
4. Document methodology in this README
5. Update performance tracking tables

---

**Questions?** Check the main project README or open an issue.
