Performance Tips
================

The C backend is already highly optimised, but a few practical choices can
deliver substantial speedups in real workloads.

Choose the Right Mode
---------------------

* **Online (``BOCPD.update``)** – minimal latency, one observation at a
  time. Python call overhead dominates when data arrive slowly.
* **Batch (``BOCPD.batch_update``)** – processes contiguous NumPy arrays
  using a single FFI call. Expect 30–50% higher throughput because the C
  loop runs uninterrupted.

Feed Contiguous NumPy Arrays
----------------------------

When calling ``batch_update`` (or passing grid parameters to Student-t),
use ``np.ascontiguousarray`` to avoid implicit copies. The bindings
already do this for internal buffers, but pre-allocating contiguous arrays
lets you reuse memory and avoid repeated validation.

Set ``max_run_length`` Appropriately
------------------------------------

* Large values increase memory and per-update work (more run lengths to
  propagate). Only track run lengths you care about.
* Rule of thumb: ``max_run_length`` ≈ 3× expected regime duration.
* If you need longer history but can tolerate approximation, consider
  downsampling the input series.

Avoid Expensive Models When Not Needed
--------------------------------------

* Student-t grid is ~6× slower than fixed-ν. Reserve it for critical
  robustness requirements.
* Binomial-Beta overhead grows with ``n_trials`` because of binomial
  coefficients. If ``n_trials`` is large and stable, rescale to a smaller
  effective sample per time step or approximate with Gaussian.

Leverage Offline Warm-Up
------------------------

Before entering a strict real-time loop, call ``batch_update`` with a
historical window. This initialises sufficient statistics so the online
phase runs at steady-state speed (avoiding the transient when all
run-length probabilities start at zero).

Disabling Strict Validation (Advanced)
--------------------------------------

Discrete models validate inputs (integers, binary) on the Python side to
prevent invalid data from reaching C. When you trust upstream data and
benchmarking shows validation overhead matters (~5–10% for small batches),
set ``strict=False`` on ``PoissonGamma``/``BernoulliBeta``/``BinomialBeta``.
Be careful: invalid values will then propagate undefined behaviour.

Profiling Tips
--------------

* Use ``benchmark_fast_bocpd.py`` under ``benchmarks/scripts`` to test new
  configurations consistently.
* Enable ``BOCPD_DEBUG_CHECKS`` (compile-time flag) only while debugging.
  It zeroes buffers for safety but reduces throughput.
* If you need to profile the Python layer, wrap updates inside
  ``numpy.errstate`` / ``time.perf_counter`` loops and measure several
  thousand iterations to minimise timer noise.

Following these guidelines keeps the BOCPD loop fast and predictable,
allowing the C implementation to remain the bottleneck rather than Python
bookkeeping.
