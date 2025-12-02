# Overview:
This file aims to track the performance of our implementation over different itterations. For each version we track, we will display the performance of our implementation for both online/production mode and offline/batch mode. 

We also track the performance of each distribution type we support in each respective version of this library.

---

## Version 1:

**Fast-BOCPD Benchmark:** Univariate Gaussian <br />
**Lambda:** 150 | **Runs:** 10 | **Warmup:** 2 <br />
**Datasets**: 1k, 10k, 100k <br />

| Size | Mode | Median (s) | Throughput | CV% |
|---|---|---|---|---|
| 1k | Online | 0.0888 | 11,260 | 0.2% |
| | Batch | 0.0687 | 14,561 | 0.4% |
| 10k | Online | 0.8445 | 11,841 | 32.6% |
| | Batch | 0.4269 | 23,423 | 0.4% |
| 100k | Online | 5.2937 | 18,891 | 1.4% |
| | Batch | 4.3223 | 23,136 | 0.1% |

**Key Observations** <br />
- Batch is 1.2-2x faster (Python loop overhead)
- Throughput increases with size (better amortization)
- 10k online has high variability (GC/cache issue)
- Overall: 11-23k obs/sec baseline