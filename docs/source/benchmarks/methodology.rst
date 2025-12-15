Benchmark Methodology
=====================

Fast-BOCPD treats performance as a first-class feature. The benchmarking
suite under ``benchmarks/`` is designed to answer three questions:

1. *How fast is each observation model in both online and offline modes?*
2. *How does performance evolve as we optimize the code base?*
3. *How do we compare against other open-source BOCPD implementations?*

This page summarizes the methodology documented in ``benchmarks/README.md``.

Data Generation
---------------

Synthetic datasets are produced by ``benchmarks/scripts/generate_data.py``.
Each file encodes known changepoint locations so we can validate both
accuracy and speed. Key parameters:

* **Distribution** – ``gaussian``, ``student_t_fixed``, ``student_t_grid``,
  ``poisson``, ``gamma``, ``bernoulli``, ``binomial``
* **Expected run length** – λ = 150 observations between changepoints
* **Sizes** – 1,000 / 10,000 / 100,000 observations

Generated arrays are versioned under ``benchmarks/data/``. The benchmark
runner checks for missing files and regenerates them automatically.

Execution Protocol
------------------

All benchmark scripts (``benchmark_fast_bocpd.py``,
``benchmark_competitors.py``) follow the same protocol:

1. **Warm-up runs** – ``--warmup-runs`` (default 2) to prime instruction
   caches and avoid measuring import/initialization overhead.
2. **Timed runs** – ``--runs`` (default 10) executed back-to-back using
   ``time.perf_counter`` with microsecond precision.
3. **Aggregation** – We record the per-run median runtime, throughput, and
   coefficient of variation (CV%) to characterise both performance and
   stability.

Benchmarks are compiled with ``-O3 -march=native -fomit-frame-pointer`` and
are typically executed on a modern laptop/workstation. Results may vary,
but relative differences remain consistent.

Performance Metrics
-------------------

Three metrics are reported everywhere (internal and competitor benchmarks):

* **Median runtime (seconds)** – robust central tendency (less sensitive to
  outliers than mean).
* **Throughput (obs/sec)** – number of observations processed per second;
  higher is better.
* **Coefficient of variation (CV%)** – ``(std / mean) × 100`` indicating
  run-to-run stability. CV% < 1% means results are highly reproducible.

Quick Start
-----------

``benchmarks/benchmark.sh`` orchestrates the entire suite:

.. code-block:: bash

   # All Fast-BOCPD models
   ./benchmark.sh fbocpd

   # Specific model
   ./benchmark.sh gaussian

   # Competitors
   ./benchmark.sh competitors

   # Everything
   ./benchmark.sh .

The script handles data generation, invokes the appropriate Python runner,
and prints formatted summaries.

Also see ``benchmarks/competitors/requirements.txt`` for installation
instructions for third-party libraries.

For more control, call the Python scripts directly:

.. code-block:: bash

   # Fast-BOCPD
   cd benchmarks/scripts
   python benchmark_fast_bocpd.py --distribution gaussian --runs 20 --warmup-runs 3
   python benchmark_fast_bocpd.py --distribution poisson --size 10000

   # Competitors
   python benchmark_competitors.py --lib ruptures --runs 10
   python benchmark_competitors.py --lib dtolpin --size 1000

Historical Tracking
-------------------

Optimization progress is documented in ``benchmarks/Benchmark_tracking.md``.
Every major iteration captures:

* Raw benchmark outputs for all models
* Compiler flags and environment notes
* Narrative explaining observed regressions or improvements

Refer to that log when evaluating long-term trends or validating that new
optimizations maintain performance guarantees.
