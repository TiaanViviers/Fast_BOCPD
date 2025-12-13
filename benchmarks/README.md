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

To validate Fast-BOCPD's performance claims, we benchmark against four established Python implementations of Bayesian changepoint detection. This section presents head-to-head comparisons on identical synthetic datasets.

### 4.1 Competitor Overview

| Library | Language | Mode Support | Models | Repository |
|---------|----------|-------------|---------|------------|
| **Fast-BOCPD** | C + Python | Online + Offline | 7 conjugate priors | [github.com/TiaanViviers/Fast_BOCPD](https://github.com/TiaanViviers/Fast_BOCPD) |
| **dtolpin/bocd** | Pure Python | Online only | Student-t | [github.com/dtolpin/bocd](https://github.com/dtolpin/bocd) |
| **ruptures** | Python/Cython/C | Offline only | Gaussian (CostNormal) | [github.com/deepcharles/ruptures](https://github.com/deepcharles/ruptures) |
| **promised-ai/changepoint** | Rust + Python bindings | Online only | 6 conjugate priors | [github.com/promised-ai/changepoint](https://github.com/promised-ai/changepoint) |
| **hildensia/bayesian_changepoint_detection** | PyTorch (Python) | Online + Offline | Student-t | [github.com/hildensia/bayesian_changepoint_detection](https://github.com/hildensia/bayesian_changepoint_detection) |

---

### 4.2 Benchmark Results

#### 4.2.1 dtolpin/bocd - Pure Python Online BOCPD

**Implementation:** Pure Python implementation of the original Adams & MacKay (2007) algorithm with numpy and scipy optimisation. Uses Student-t conjugate prior.

**Key characteristics:**
- Educational reference implementation
- No compiled optimizations
- Single-threaded Python loops
- Online only, no offline batching

**Performance:**

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.1660 | 6,023/s | 0.3% |
| 10k | Online | 6.1030 | 1,639/s | 1.0% |
| 100k | Online | 614.8018 | 163/s | 0.6% |

**Analysis:**
- **Linear scaling preserved:** O(n) complexity maintained, but with high Python overhead
- **Degrading efficiency at scale:** Throughput drops from 6k obs/s (1k) to 163 obs/s (100k)
- **163x slower than Fast-BOCPD** at 100k observations (163 vs 27,000 obs/s for Student-t fixed)
- Excellent stability (CV% < 1%) indicates deterministic performance

**Verdict: Not production-ready for large datasets.** While correct, the pure Python implementation becomes prohibitively slow beyond 10k observations. Fast-BOCPD's C implementation provides **40-165x speedup** across all dataset sizes.

---

#### 4.2.2 ruptures - Offline Segmentation (Cython/C)

**Implementation:** Optimized offline changepoint detection using dynamic programming. Uses Gaussian likelihood with CostNormal (mean+variance changes).
*Important note:* Ruptures is not bayesian, but it is the gold standard for
offline changepoint detection, used in many production systems by many 
companies, including NASA.

**Key characteristics:**
- Compiled Cython/C backend
- Optimized for batch processing
- Industry-standard segmentation library

**Performance:**

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Offline | 0.0441 | 22,676/s | 4.8% |
| 10k | Offline | 0.7634 | 13,099/s | 0.6% |
| 100k | Offline | 38.9962 | 2,564/s | 1.0% |

**Analysis:**
- **Sublinear scaling issue:** Throughput degrades significantly from 22k → 2.5k obs/s
- **10x slower than Fast-BOCPD at 100k** (2,564 vs 27,747 obs/s for Gaussian offline)
- Despite Cython/C optimizations, our C implementation maintains superior scaling
- Note: 4.8% CV% on 1k data suggests warmup artifacts on small datasets

**Verdict: Strong competitor for offline use cases**, but Fast-BOCPD offers **3-11x better throughput** with more consistent scaling. Ruptures excels in its rich ecosystem (16+ cost functions), but for pure Gaussian changepoint 
detection, Fast-BOCPD dominates.

---

#### 4.2.3 promised-ai/changepoint - Rust Implementation

**Implementation:** Rust-based BOCPD with Python bindings via PyO3. Supports multiple conjugate priors (NormalGamma, BetaBernoulli, PoissonGamma).

**Key characteristics:**
- Compiled Rust (zero-cost abstractions)
- Memory-safe systems language
- Online-only (no offline mode)

**Performance:**

**Gaussian (NormalGamma prior):**
| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0367 | 27,227/s | 0.2% |
| 10k | Online | 1.4060 | 7,112/s | 1.8% |
| 100k | Online | 109.2458 | 915/s | 0.7% |

**Bernoulli (BetaBernoulli prior):**
| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0180 | 55,595/s | 0.7% |
| 10k | Online | 1.4018 | 7,134/s | 0.3% |
| 100k | Online | 83.3343 | 1,200/s | 0.5% |

**Poisson (PoissonGamma prior):**
| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 0.0316 | 31,643/s | 0.3% |
| 10k | Online | 1.1186 | 8,940/s | 1.4% |
| 100k | Online | 86.6610 | 1,154/s | 0.9% |

**Analysis:**
- **Excellent small-scale performance:** Matches or exceeds Fast-BOCPD at n=1k
- **Severe scaling degradation:** Throughput collapses from 27k → 915 obs/s (Gaussian, 30x slowdown)
- **Similar patterns across all models:** Bernoulli shows 46x slowdown, Poisson shows 27x slowdown
- **Hypothesis: Memory allocation overhead?** Rust's safety guarantees may introduce per-datum allocation costs
- **24-29x slower than Fast-BOCPD at 100k** observations across all comparable models

**Comparison with Fast-BOCPD (100k observations):**
| Model | promised-ai | Fast-BOCPD | Speedup |
|-------|-------------|------------|---------|
| Gaussian | 915 obs/s | 25,952 obs/s | **28.3x** |
| Bernoulli | 1,200 obs/s | 33,573 obs/s | **28.0x** |
| Poisson | 1,154 obs/s | 21,402 obs/s | **18.5x** |

**Verdict: Competitive at small scale, poor at large scale.** Despite Rust's performance reputation, the implementation exhibits unexpected O(n) throughput degradation. Fast-BOCPD's **18-28x advantage at 100k observations** is particularly striking given Rust's theoretical performance advantages. This suggests algorithmic differences (possible reallocation patterns) rather than language limitations.

---

#### 4.2.4 hildensia/bayesian_changepoint_detection - PyTorch Implementation

**Implementation:** PyTorch-based BOCPD with GPU support. Intended for deep learning integration and GPU acceleration.


*Important Note:* I was under the impression that the PyTorch implementation
is a optimisation mechanism and that this library is fast, espesially on gpu.
Results did not indicate this and it might be that I am making crucial 
mistakes in the benchmarking process. If anyone wants to double check the 
implementation of this benchmarking please see 
`scripts/benchmark_competitors.py`, `scripts/benchmark_hildensia.py` and my gpu testing notebook that i ran on google colab with a T4 GPU at `scripts/hildensia_gpu_gcolab.ipynb`.


**Key characteristics:**
- PyTorch backend (dynamic computation graphs)
- CPU and GPU support
- **O(n²) complexity** due to maintaining full run-length distributions without truncation

**Performance (CPU only, n=1k):**

| Size | Mode | Median (s) | Throughput | CV% |
|------|------|------------|------------|-----|
| 1k | Online | 58.1504 | 17 obs/s | 0.2% |
| 1k | Offline | 340.0730 | 3 obs/s | 0.1% |

**Larger sizes:** Not benchmarked due to quadratic complexity (estimated 100 minutes for 10k, 169 hours for 100k).

**Analysis:**
- **Catastrophic performance:** 17 obs/s vs Fast-BOCPD's 26,504 obs/s (Student-t, **1,559x slower**)
- **Offline mode even worse:** 3 obs/s vs 35,714 obs/s (**11,905x slower!**)
- **O(n²) scaling:** Creates ~500,000 PyTorch distributions for 1000 observations
- PyTorch validation overhead dominates (Chi2 → Gamma distribution creation in tight loop)
- **No truncation/pruning:** Maintains all run-length hypotheses, leading to memory explosion

**GPU Attempt:** Even on Google Colab T4 GPU, performance remained unusably slow due to:
1. Algorithmic complexity (GPU can't fix O(n²) → O(n))
2. CPU-GPU transfer overhead for sequential online processing
3. PyTorch distribution creation not GPU-optimized for this use case

**Verdict: Fundamentally unscalable.** The O(n²) complexity makes this implementation academic-only. Fast-BOCPD's truncation strategy and C implementation provide **1,500-12,000x speedup**. While Hildensia may excel in GPU-accelerated batch scenarios with custom kernels, for standard BOCPD it cannot compete.

---

### 4.3 Cross-Library Comparison

**Gaussian Online Detection (100k observations):**

| Library | Language | Throughput | vs Fast-BOCPD |
|---------|----------|------------|---------------|
| **Fast-BOCPD** | **C** | **25,952 obs/s** | **1.0x (baseline)** |
| promised-ai | Rust | 915 obs/s | 28.3x slower |
| ruptures | Cython/C | 2,564 obs/s* | 10.1x slower* |
| dtolpin | Python | 163 obs/s | 159.2x slower |
| hildensia | PyTorch | 17 obs/s** | 1,525x slower** |

\* Offline mode comparison  
\** Extrapolated from n=1k (actual run would take hours)

---

### 4.4 Key Takeaways

1. **Language matters, but implementation matters more:** 
   - Rust (promised-ai) starts strong but scales poorly
   - Cython/C (ruptures) is fast but still 10x behind our C implementation
   - Pure Python (dtolpin) is predictably slow (~160x)
   - PyTorch (hildensia) suffers from O(n²) algorithmic issues

2. **Fast-BOCPD's advantages:**
   - **Consistent O(n) scaling:** Maintains high throughput even at 100k observations
   - **Minimal overhead:** Direct C implementation without interpreter/VM layers
   - **Smart truncation:** Prunes unlikely run lengths, avoiding quadratic explosion
   - **Batch optimizations:** Offline mode leverages cache locality

3. **When to use competitors:**
   - **ruptures:** Rich ecosystem for exploratory analysis (16+ cost functions, visualization tools)
   - **promised-ai:** If already in Rust ecosystem and staying below 10k observations
   - **dtolpin:** Educational purposes, understanding BOCPD internals
   - **hildensia:** GPU-accelerated batch scenarios with custom implementation (not stock)

4. **When to use Fast-BOCPD:**
   - Production workloads
   - Real-time streaming applications
   - Embedded systems with limited resources
   - Any scenario where throughput matters

**Bottom line:** Fast-BOCPD achieves **10-1,500x better throughput** than competing Python/Rust implementations while maintaining the full algorithmic correctness of the original Adams & MacKay (2007) BOCPD algorithm.




