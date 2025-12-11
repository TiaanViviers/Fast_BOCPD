# Overview:
This file aims to track the performance of our implementation over different itterations. For each version we track, we will display the performance of our implementation for both online/production mode and offline/Offline mode. 

We also track the performance of each distribution type we support in each respective version of this library.

---

## Version 1: Baseline

**Fast-BOCPD Benchmark:** Univariate Gaussian <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.0888 | 11,260/s | 0.2% |
| | Offline | 0.0687 | 14,561/s | 0.4% |
| 10k | Online | 0.8445 | 11,841/s | 32.6% |
| | Offline | 0.4269 | 23,423/s | 0.4% |
| 100k | Online | 5.2937 | 18,891/s | 1.4% |
| | Offline | 4.3223 | 23,136/s | 0.1% |

**Key Observations:**
- Offline is 1.2-2x faster (Python loop overhead)
- Throughput increases with size (better amortization)
- 10k online has high variability (GC/cache issue)
- Overall: 11-23k obs/sec baseline

---

## Version 2: Compiler Optimizations

**Changes:**
- Added compiler flags: `-O3 -march=native -fomit-frame-pointer`
- Experimented with `-ffast-math` and `-funroll-loops` but removed them:
  - `-ffast-math`: Hurt performance (likely due to math library optimizations)
  - `-funroll-loops`: No performance gain, increased CV% (variability)
- No code changes, purely compilation optimizations

**Fast-BOCPD Benchmark:** Univariate Gaussian <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% | Change |
|---|---|---|---|---|---|
| 1k | Online | 0.0322 | 31,009/s | 0.1% | **+175%** |
| | Offline | 0.0228 | 43,835/s | 1.7% | **+201%** |
| 10k | Online | 0.3838 | 26,057/s | 0.1% | **+120%** |
| | Offline | 0.2882 | 34,699/s | 0.1% | **+48%** |
| 100k | Online | 3.9899 | 25,063/s | 0.9% | **+33%** |
| | Offline | 3.0314 | 32,988/s | 0.5% | **+43%** |

**Key Observations:**
- **Massive gains on small datasets (1k):** 3x faster across the board.
- **Strong gains on medium datasets (10k online):** 2x faster for online use case
- **Large datasets (100k):** 1.3-1.4x faster
- **10k online stability FIXED:** CV% dropped from 32.6% to 0.1%
- **All measurements highly stable:** CV% ≤ 2% across the board

**Performance Summary:**
- **Small data (1k):** Excellent - ~3x speedup
- **Medium data (10k):** Good - ~2x speedup 
- **Large data (100k):** Good - ~1.35x speedup

#### New Models:
During the development cycle of version 2 some more models have been added to
the fast-bocpd library and their performance was measured to be as follows: <br />

---

**Fast-BOCPD Benchmark:** Student-t (Fixed ν) <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.0377 | 26,504/s | 0.5% |
| | Offline | 0.0280 | 35,714/s | 0.2% |
| 10k | Online | 0.4507 | 22,189/s | 0.3% |
| | Offline | 0.3520 | 28,409/s | 0.5% |
| 100k | Online | 4.5880 | 21,796/s | 0.2% |
| | Offline | 3.6040 | 27,747/s | 0.1% |

**Key Observations:**
- Performance similar to Gaussian model (~20-35k obs/sec)
- Batch speedup 1.27-1.35x (consistent across dataset sizes)
- Very stable: CV% ≤ 0.5% across all configurations
- Slightly slower than Gaussian due to extra degrees of freedom parameter

---

**Fast-BOCPD Benchmark:** Student-t (Grid ν) <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.2261 | 4,422/s | 0.2% |
| | Offline | 0.2150 | 4,652/s | 0.4% |
| 10k | Online | 2.8238 | 3,541/s | 0.1% |
| | Offline | 2.7136 | 3,685/s | 0.1% |
| 100k | Online | 28.8063 | 3,471/s | 0.5% |
| | Offline | 27.7352 | 3,606/s | 0.1% |

**Key Observations:**
- **Significantly slower:** ~3.5-4.5k obs/sec (vs 20-35k for fixed ν)
- Grid search over degrees of freedom adds ~6-7x overhead
- Minimal batch speedup (1.04-1.05x) - computation dominates
- Very stable despite complexity: CV% ≤ 0.5%
- Trade-off: robustness to unknown ν vs. computational cost

---

**Fast-BOCPD Benchmark:** Bernoulli-Beta <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.0281 | 35,608/s | 0.5% |
| | Offline | 0.0222 | 45,013/s | 0.2% |
| 10k | Online | 0.2952 | 33,871/s | 1.8% |
| | Offline | 0.2468 | 40,521/s | 0.1% |
| 100k | Online | 2.9786 | 33,573/s | 1.0% |
| | Offline | 2.5797 | 38,765/s | 1.0% |

**Key Observations:**
- **Excellent performance:** 33-45k obs/sec (faster than Gaussian!)
- Bernoulli likelihood is computationally simple (beta updates)
- Batch speedup 1.15-1.26x (Python overhead reduction)
- Very stable: CV% ≤ 1.8% (10k online slightly higher)
- Best performer for binary data

---

**Fast-BOCPD Benchmark:** Binomial-Beta <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.0574 | 17,422/s | 0.3% |
| | Offline | 0.0429 | 23,314/s | 0.2% |
| 10k | Online | 0.6762 | 14,788/s | 0.2% |
| | Offline | 0.5307 | 18,843/s | 0.2% |
| 100k | Online | 6.8500 | 14,599/s | 0.1% |
| | Offline | 5.3848 | 18,571/s | 0.1% |

**Key Observations:**
- Moderate performance: 14-23k obs/sec
- Slower than Bernoulli (binomial coefficient computations)
- Batch speedup 1.27-1.34x (consistent)
- Extremely stable: CV% ≤ 0.3% across all configurations
- Good choice for count/proportion data with fixed trials

---

**Fast-BOCPD Benchmark:** Poisson-Gamma <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.0399 | 25,039/s | 0.5% |
| | Offline | 0.0261 | 38,249/s | 8.0% |
| 10k | Online | 0.4592 | 21,777/s | 0.4% |
| | Offline | 0.3212 | 31,131/s | 0.1% |
| 100k | Online | 4.6725 | 21,402/s | 0.1% |
| | Offline | 3.2575 | 30,698/s | 0.1% |

**Key Observations:**
- Good performance: 21-38k obs/sec
- Comparable to Student-t (fixed ν) performance
- Batch speedup 1.43-1.53x (excellent, highest across all models!)
- One outlier: 1k offline has 8% CV% (likely warmup artifact on small data)
- Excellent for count/rate data

---

**Fast-BOCPD Benchmark:** Gamma-Gamma (Fixed Shape) <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.0340 | 29,448/s | 0.2% |
| | Offline | 0.0243 | 41,103/s | 0.2% |
| 10k | Online | 0.3908 | 25,586/s | 0.1% |
| | Offline | 0.2950 | 33,894/s | 1.0% |
| 100k | Online | 4.1170 | 24,290/s | 0.1% |
| | Offline | 3.1020 | 32,237/s | 0.1% |

**Key Observations:**
- Very good performance: 24-41k obs/sec
- Batch speedup 1.32-1.40x (solid)
- Exceptionally stable: CV% ≤ 1% across all configurations
- Excellent for positive continuous data (e.g., durations, amounts)

---

### Model Performance Comparison (Version 2)

**Throughput Rankings (100k dataset, online mode):**
1. **Bernoulli-Beta:** 33,573 obs/sec ⭐ (simplest likelihood)
2. **Gaussian-NIG:** 25,063 obs/sec
3. **Gamma-Gamma:** 24,290 obs/sec
4. **Student-t (Fixed ν):** 21,796 obs/sec
5. **Poisson-Gamma:** 21,402 obs/sec
6. **Binomial-Beta:** 14,599 obs/sec (binomial coefficients)
7. **Student-t (Grid ν):** 3,471 obs/sec (grid search overhead)

**Key Takeaways:**
- **Simple likelihoods win:** Bernoulli > Gaussian > Gamma
- **Grid search is expensive:** 6-7x slower than fixed parameters
- **All models highly stable:** CV% typically < 1%
- **Batch mode consistent benefit:** 1.04-1.53x speedup across all models
- **Production ready:** 15k-34k obs/sec for most models (except grid)
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

