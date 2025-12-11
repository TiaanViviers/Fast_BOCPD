# Fast-BOCPD Benchmarks

Performance benchmarking suite for Fast-BOCPD.

## Content
 - 1. Overview of our benchmarking suite
 - 2. Quick start
 - 3. Benchmarks for this library
 - 4. Benchmarks for other changepoint libraries

---

## 1. Overview
The goal of this part of the project is to benchmark the performance of our 
implementation for 3 reasons: <br />
1. **Changepoint detection algorithms are inherently slow:** <br />
Traditional methods often have high polynomial complexity, such as O(n²). 
The original Bayesian Online Changepoint Detection algorithm 
introduced by Adams and MacKay (2007), has a time complexity of O(n).
Many of these algorithms are implemented in languages like Python or R with
terrible loop overhead. In order for this library to be helpful in the space 
of changepoint detection, we need to be very concious of performance.

2. **Itterative performance improvement:** <br />
To improve the performance of this library multiple speedup experiments
will need to be conducted. We thus need strong past benchmarks to reflect on
the significance and effect size of new implementations.

3. **Competitors:** <br />
The goal of this project is not just to build a easy-to-use and low dependency
changepoint detection library, but also to be really fast. Speed and 
performance is relative to that of other implementations.


### 1.1. Benchmarking Process


