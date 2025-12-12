# Fast-BOCPD Benchmarks

Comprehensive performance benchmarking suite for Fast-BOCPD.

## Contents
- [Overview](#1-overview)
- [Quick Start](#2-quick-start)
- [Internal Benchmarks](#3-internal-benchmarks)
- [Competitor Benchmarks](#4-competitor-benchmarks)

---

## 1. Overview

This benchmarking suite serves three critical purposes:

### 1.1 Why Performance Matters

**Changepoint detection is computationally expensive.** Traditional methods often exhibit polynomial complexity (O(n²) or worse). While the original Bayesian Online Changepoint Detection (BOCPD) algorithm by Adams and MacKay (2007) achieves linear O(n) complexity, pure Python or R implementations suffer from significant loop overhead. Fast-BOCPD addresses this by implementing the core algorithm in C, making performance a first-class priority.

**Iterative optimization requires rigorous measurement.** As we continuously improve Fast-BOCPD through compiler optimizations, algorithmic refinements, and implementation enhancements, we need reliable historical benchmarks to quantify the impact of each change.

**Performance is relative.** Fast-BOCPD aims to be not just easy to use and dependency-light, but also demonstrably faster than competing implementations. Comparative benchmarks validate this claim.

### 1.2 Benchmarking Methodology

#### Data Generation
The `generate_data.py` script creates synthetic time series with known changepoint locations. Users can specify:
- **Distribution type**: Gaussian, Student-t(fixed or grid df), Poisson, Gamma, Bernoulli, Binomial
- **Segment length**: Expected run length between changepoints (λ = 150)
- **Sample size**: 1,000, 10,000, or 100,000 observations

All benchmark datasets are pre-generated and stored in `benchmarks/data/` for reproducibility.

#### Execution Protocol
Each benchmark follows a standardized procedure:
1. **Warm-up runs** (`--warmup-runs`): Initial executions to warm instruction caches and minimize JIT compilation effects
2. **Timed runs** (`--runs`): Subsequent executions with precise timing measurements
3. **Statistical aggregation**: Results are summarized across all timed runs

#### Performance Metrics
We report three key metrics:

- **Median runtime (seconds)**: More robust to outliers than mean, represents typical performance
- **Throughput (obs/sec)**: Observations processed per second; higher is better
- **Coefficient of variation (CV%)**: `(std/mean) × 100`; measures run-to-run stability

Low CV% indicates consistent, predictable performance. High CV% suggests external factors (system load, thermal throttling) affecting results.


### 1.3 Performance tracking.
The performance evolution of fast-bocpd is tracked inside 
`Benchmark_tracking.md`. All available models are benchmarked in each "version"
of development. This .md file thus allows us to reflect and compare against 
previous versions of performance and explains what steps where taken to improve
performance.


### 1.4 Directory structure
```
├── benchmark.sh                      # Main benchmarking runner
├── Benchmark_tracking.md             # Log of past and current performance
├── competitors
│   ├── dtolpin_bocd/                 # Competitor library, needs git clone
│   │   
│   └── requirements.txt              # Installation instructions for competitors
│
├── data/                             # Synthetic benchmarking data, *.npy
│
├── README.md
└── scripts
    ├── benchmark_competitors.py      # Main competitor benchmarking script
    ├── benchmark_dtolpin_bocd.py     # Benchmarking for dtolpin's library
    ├── benchmark_fast_bocpd.py       # Main benchmarking script for our library
    ├── benchmark_ruptures.py         # Benchmarking for ruptures library
    ├── generate_data.py              # Data generation script
```

---

## 2. Quick Start

### 2.1 Running Benchmarks with `benchmark.sh`

The simplest way to run benchmarks is using the provided shell script. It automatically generates missing data files and executes benchmarks.

**Run all Fast-BOCPD benchmarks:**
```bash
./benchmark.sh Fbocpd
```

**Run a specific fast-bocpd distribution:**
```bash
./benchmark.sh gaussian
./benchmark.sh student_t_fixed
./benchmark.sh poisson
# ... etc
```

**Run competitor benchmarks:**
```bash
./benchmark.sh competitors
```

**Run everything:**
```bash
./benchmark.sh .
```

The script will:
1. Check for required data files in `benchmarks/data/`
2. Generate any missing datasets automatically
3. Execute the appropriate benchmark scripts
4. Display results in a formatted table

### 2.2 Setting Up Competitor Libraries

Before running competitor benchmarks, install the required libraries:

**Clone and set up dtolpin/bocd:**
```bash
cd benchmarks/competitors
git clone https://github.com/dtolpin/bocd.git dtolpin_bocd
# Dependencies (numpy, scipy) should already be installed
```

**Install all competitors at once:**
```bash
cd benchmarks/competitors
pip install -r requirements.txt
git clone https://github.com/dtolpin/bocd.git dtolpin_bocd
```

### 2.3 Custom Benchmark Runs

For more control over benchmark parameters, use the Python scripts directly:

**Fast-BOCPD benchmarks:**
```bash
cd benchmarks/scripts
python benchmark_fast_bocpd.py --distribution gaussian --runs 20 --warmup-runs 3
python benchmark_fast_bocpd.py --distribution poisson --size 10000
```

**Competitor benchmarks:**
```bash
cd benchmarks/scripts
python benchmark_competitors.py --lib ruptures --runs 10
python benchmark_competitors.py --lib dtolpin --size 1000
```

Run `python <script>.py --help` for complete documentation of available options.

---

## 3. Internal Benchmarks

This section presents the **current performance** of all Fast-BOCPD models. All benchmarks use:
- **Test environment**: Standard laptop/workstation (results may vary by hardware)
- **Compiler flags**: `-O3 -march=native -fomit-frame-pointer`
- **Benchmark settings**: λ=150, 10 runs, 2 warmup runs
- **Dataset sizes**: 1,000 / 10,000 / 100,000 observations

For historical performance tracking and optimization notes, see [`Benchmark_tracking.md`](Benchmark_tracking.md).

### 3.1 Performance Summary

**Throughput comparison (100k observations, online mode):**

| Model | Obs/sec | Description | Performance Notes |
|-------|---------|-------------|-------------------|
| **Bernoulli-Beta** | 33,573 | Binary data (0/1 outcomes) | Simplest likelihood, fastest |
| **Gaussian-NIG** | 25,063 | Continuous data (Normal distribution) | Excellent all-around performance |
| **Gamma-Gamma** | 24,290 | Positive continuous data (durations, amounts) | Very stable, good throughput |
| **Student-t (Fixed ν)** | 21,796 | Robust to outliers, known degrees of freedom | Slightly slower than Gaussian |
| **Poisson-Gamma** | 21,402 | Count data (events per interval) | Good performance, excellent batch speedup |
| **Binomial-Beta** | 14,599 | Count/proportion data (k successes in n trials) | Slower due to binomial coefficients |
| **Student-t (Grid ν)** | 3,471 | Robust to outliers, unknown degrees of freedom | Grid search overhead (~7x slower) |

**Key insights:**
- **Most models achieve 20-35k obs/sec** - sufficient for real-time applications
- **Offline mode is 1.3-1.5x faster** than online mode (reduced Python overhead)
- **All models are highly stable** - CV% typically < 1%
- **Choose Student-t Grid only when necessary** - accuracy vs. speed trade-off

### 3.2 Detailed Results by Model

#### Gaussian-NIG (Normal-Inverse-Gamma)
**Use case**: General-purpose continuous data with unknown mean and variance

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0322 | 31,009/s | 0.1% |
| | Offline | 0.0228 | 43,835/s | 1.7% |
| 10k | Online | 0.3838 | 26,057/s | 0.1% |
| | Offline | 0.2882 | 34,699/s | 0.1% |
| 100k | Online | 3.9899 | 25,063/s | 0.9% |
| | Offline | 3.0314 | 32,988/s | 0.5% |

#### Student-t (Fixed ν)
**Use case**: Robust changepoint detection when degrees of freedom are known

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0377 | 26,504/s | 0.5% |
| | Offline | 0.0280 | 35,714/s | 0.2% |
| 10k | Online | 0.4507 | 22,189/s | 0.3% |
| | Offline | 0.3520 | 28,409/s | 0.5% |
| 100k | Online | 4.5880 | 21,796/s | 0.2% |
| | Offline | 3.6040 | 27,747/s | 0.1% |

Performance comparable to Gaussian; ~20% slower due to additional parameter.

#### Student-t (Grid ν)
**Use case**: Robust detection when degrees of freedom are unknown (searches over grid)

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.2261 | 4,422/s | 0.2% |
| | Offline | 0.2150 | 4,652/s | 0.4% |
| 10k | Online | 2.8238 | 3,541/s | 0.1% |
| | Offline | 2.7136 | 3,685/s | 0.1% |
| 100k | Online | 28.8063 | 3,471/s | 0.5% |
| | Offline | 27.7352 | 3,606/s | 0.1% |

**6-7x slower** than fixed ν due to grid search. Minimal batch speedup (computation-dominated). Use when robustness to unknown ν is critical.

#### Bernoulli-Beta
**Use case**: Binary data (success/failure, on/off, yes/no)

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0281 | 35,608/s | 0.5% |
| | Offline | 0.0222 | 45,013/s | 0.2% |
| 10k | Online | 0.2952 | 33,871/s | 1.8% |
| | Offline | 0.2468 | 40,521/s | 0.1% |
| 100k | Online | 2.9786 | 33,573/s | 1.0% |
| | Offline | 2.5797 | 38,765/s | 1.0% |

**Fastest model** - simple beta conjugate updates, no expensive computations.

#### Binomial-Beta
**Use case**: Proportion data (k successes out of n trials)

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0574 | 17,422/s | 0.3% |
| | Offline | 0.0429 | 23,314/s | 0.2% |
| 10k | Online | 0.6762 | 14,788/s | 0.2% |
| | Offline | 0.5307 | 18,843/s | 0.2% |
| 100k | Online | 6.8500 | 14,599/s | 0.1% |
| | Offline | 5.3848 | 18,571/s | 0.1% |

Slower than Bernoulli due to binomial coefficient calculations. Extremely stable (CV% < 0.3%).

#### Poisson-Gamma
**Use case**: Count data (number of events per time interval)

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0399 | 25,039/s | 0.5% |
| | Offline | 0.0261 | 38,249/s | 8.0% |
| 10k | Online | 0.4592 | 21,777/s | 0.4% |
| | Offline | 0.3212 | 31,131/s | 0.1% |
| 100k | Online | 4.6725 | 21,402/s | 0.1% |
| | Offline | 3.2575 | 30,698/s | 0.1% |

**Best batch speedup** (1.43-1.53x). Note: 1k offline shows 8% CV% (likely warmup artifact on small data).

#### Gamma-Gamma (Fixed Shape)
**Use case**: Positive continuous data (waiting times, transaction amounts, lifetimes)

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0340 | 29,448/s | 0.2% |
| | Offline | 0.0243 | 41,103/s | 0.2% |
| 10k | Online | 0.3908 | 25,586/s | 0.1% |
| | Offline | 0.2950 | 33,894/s | 1.0% |
| 100k | Online | 4.1170 | 24,290/s | 0.1% |
| | Offline | 3.1020 | 32,237/s | 0.1% |

Excellent stability (CV% ≤ 1%) and solid throughput. Good all-around performer for positive data.

---

## 4. Competitor Benchmarks




