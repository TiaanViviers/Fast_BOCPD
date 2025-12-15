Competitor Comparison
=====================

To validate Fast-BOCPD’s claims we benchmark against other Bayesian
changepoint libraries using the exact same datasets and methodology.

Competitor Overview
-------------------

.. csv-table::
   :header: Library, Language, Mode Support, Models, Repository

   Fast-BOCPD, C + Python, Online + Offline, 7 conjugate priors, https://github.com/TiaanViviers/Fast_BOCPD
   dtolpin/bocd, Pure Python, Online, Student-t, https://github.com/dtolpin/bocd
   ruptures, Python/Cython/C, Offline, Gaussian (CostNormal), https://github.com/deepcharles/ruptures
   promised-ai/changepoint, Rust + Python, Online, 6 conjugate priors, https://github.com/promised-ai/changepoint
   hildensia/bayesian_changepoint_detection, PyTorch, Online + Offline, Student-t, https://github.com/hildensia/bayesian_changepoint_detection

dtolpin/bocd (Pure Python)
--------------------------

Reference implementation of Adams & MacKay (2007); valuable for education,
but limited by interpreter overhead.

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 0.1660, 6,023, 0.3%
   10k, Online, 6.1030, 1,639, 1.0%
   100k, Online, 614.8018, 163, 0.6%

* 40–165× slower than Fast-BOCPD (Student-t fixed).
* Predictable scaling (O(n)) but untenable runtimes beyond ~10k points.

ruptures (Offline Gaussian)
---------------------------

Industry-standard offline segmentation library (dynamic programming plus
Gaussian cost). Included to show how a highly optimized Cython/C system
fares against our offline mode.

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Offline, 0.0441, 22,676, 4.8%
   10k, Offline, 0.7634, 13,099, 0.6%
   100k, Offline, 38.9962, 2,564, 1.0%

* 3–11× slower than Fast-BOCPD (Gaussian offline).
* Throughput drops sharply at scale (22k → 2.5k obs/s).

promised-ai/changepoint (Rust)
------------------------------

Rust implementation with PyO3 bindings. Supports NormalGamma, BetaBernoulli,
and PoissonGamma priors.

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%
   :name: promised-gaussian

   1k, Gaussian, 0.0367, 27,227, 0.2%
   10k, Gaussian, 1.4060, 7,112, 1.8%
   100k, Gaussian, 109.2458, 915, 0.7%

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%
   :name: promised-bernoulli

   1k, Bernoulli, 0.0180, 55,595, 0.7%
   10k, Bernoulli, 1.4018, 7,134, 0.3%
   100k, Bernoulli, 83.3343, 1,200, 0.5%

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%
   :name: promised-poisson

   1k, Poisson, 0.0316, 31,643, 0.3%
   10k, Poisson, 1.1186, 8,940, 1.4%
   100k, Poisson, 86.6610, 1,154, 0.9%

* Excellent performance for n ≤ 1k.
* Severe throughput collapse (≈20–30× slowdown) by 100k observations.
* Fast-BOCPD remains 18–28× faster at large scale.

hildensia/bayesian_changepoint_detection (PyTorch)
---------------------------------------------------

PyTorch implementation intended for GPU acceleration. Maintains the full
run-length distribution ⇒ O(n²) complexity.

CPU (n = 1k):

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 58.1504, 17, 0.2%
   1k, Offline, 340.0730, 3, 0.1%

GPU (T4, n = 1k):

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 317.39, 3, 0.3%
   1k, Offline, 1808.03, 1, 0.5%

* 1,500–12,000× slower than Fast-BOCPD (Student-t) even on GPU.
* Scaling beyond 1k observations is impractical (estimated 169 hours for 100k).

Cross-Library Snapshot (100k Observations)
------------------------------------------

.. csv-table::
   :header: Library, Throughput (obs/s), Relative to Fast-BOCPD

   Fast-BOCPD (Gaussian offline), 25,952, 1.0×
   promised-ai (Gaussian online), 915, 28.3× slower
   ruptures (offline), 2,564, 10.1× slower
   dtolpin (online), 163, 159× slower
   hildensia (online, extrapolated), 17, 1,500× slower

Key Takeaways
-------------

* **Implementation dominates language.** Even though Rust/Cython/PyTorch are
  capable languages, algorithmic details (memory allocation, truncation,
  O(n) vs O(n²)) determine real-world throughput.
* **Fast-BOCPD maintains O(n) scaling** across all models while sustaining
  20k–35k obs/s at 100k observations.
* **Competitor sweet spots:**

  - ``ruptures`` for its rich offline ecosystem.
  - ``promised-ai`` when already invested in Rust and working with ≤10k
    samples.
  - ``dtolpin`` for educational/reference purposes.
  - ``hildensia`` for experimental GPU research (not production).
* **When performance matters**—production streaming, large offline batches,
  embedded deployments—Fast-BOCPD delivers **10–1,500× speedups** while
  staying dependency-light and API-compatible with standard Python workflows.
