# Overview:
This file aims to track the performance of our implementation over different itterations. For each version we track, we will display the performance of our implementation for both online/production mode and offline/batch mode. 

We also track the performance of each distribution type we support in each respective version of this library.

---

## Version 1: Baseline

**Fast-BOCPD Benchmark:** Univariate Gaussian <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.0888 | 11,260/s | 0.2% |
| | Batch | 0.0687 | 14,561/s | 0.4% |
| 10k | Online | 0.8445 | 11,841/s | 32.6% |
| | Batch | 0.4269 | 23,423/s | 0.4% |
| 100k | Online | 5.2937 | 18,891/s | 1.4% |
| | Batch | 4.3223 | 23,136/s | 0.1% |

**Key Observations:**
- Batch is 1.2-2x faster (Python loop overhead)
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
| 1k | Online | 0.0423 | 23,637/s | 0.2% | **+110%** |
| | Batch | 0.0331 | 30,245/s | 0.1% | **+108%** |
| 10k | Online | 0.5135 | 19,473/s | 0.1% | **+64%** |
| | Batch | 0.4198 | 23,819/s | 0.2% | **+2%** |
| 100k | Online | 5.2430 | 19,073/s | 0.1% | **+1%** |
| | Batch | 4.2921 | 23,299/s | 0.0% | **+1%** |

**Key Observations:**
- **Massive gains on small datasets (1k):** 2.1x faster across the board.
- **Strong gains on medium datasets (10k online):** 1.64x faster for online use case
- **Large datasets (100k):** Essentially unchanged (within measurement variance)
- **10k online stability FIXED:** CV% dropped from 32.6% to 0.1%
- **All measurements highly stable:** CV% ≤ 0.2% across the board

**Performance Summary:**
- **Small data (1k):** Excellent - 2.1x speedup
- **Medium data (10k):** Good - 1.6x speedup (online), stable (batch)
- **Large data (100k):** Neutral - No regression, maintains baseline

