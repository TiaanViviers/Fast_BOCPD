Adding New Models
=================

Fast-BOCPD supports multiple conjugate observation models. Each model is
encapsulated in its own .c and .h file and plugged into the BOCPD
engine via a virtual table. This guide walks through the steps required to
add another model (for example, a Negative-Binomial variant).

1. Design the Sufficient Statistics
-----------------------------------

Decide which minimal statistics you need to maintain per run length.
Typical patterns include:

* ``n`` (count), ``sum_x``, ``sum_x2`` for Gaussian-like models.
* ``sum_k`` or ``sum_log_x`` for discrete count models.

Define the ``struct`` in a new header (e.g.,
``fast_bocpd/_c/negative_binomial_gamma.h``) and provide a matching
``typedef`` for the parameter struct.

2. Implement the Model in C
---------------------------

Each model must expose the following functions:

* ``size_t <model>_stats_size(void);``
* ``void <model>_prior_stats(<Stats>* stats);``
* ``void <model>_update_stats(<Stats>* stats,
  const <Params>* params, double x);``
* ``double <model>_predictive_logpdf(const <Params>* params,
  const <Stats>* stats, double x);``
* ``void <model>_copy_stats(void* dst, const void* src);``

Follow the established style: validate inputs aggressively, document edge
cases, and keep the predictive calculation numerically stable (use
``lgamma``, ``log1p``, etc.). Refer to existing files such as
``poisson_gamma.c`` for patterns.

3. Register the Model with BOCPD
--------------------------------

* Add an enum entry to ``ObsModelType`` (``bocpd_core.h``).
* Extend ``ObsModelParams`` and the ``copy_obs_params`` helper in
  ``bocpd_core.c``.
* Update ``init_obs_vtable`` so the new model’s functions are wired into
  the virtual table.
* Add Python ``ctypes`` definitions in ``fast_bocpd/_bindings.py`` and
  expose the model through the high-level API (usually by extending
  ``fast_bocpd/models.py``).

4. Write Tests
--------------

* Add a focused C test in ``tests/c_tests`` to cover prior stats, update
  logic, predictive values, and validation.
* Mirror that test coverage in ``tests/python/models`` and
  ``tests/python/integration`` so the Python surface area is also checked.

5. Update the Documentation
---------------------------

* Document the new model in :doc:`../api/index`.
* Mention any new parameters in the user guides (installation, tuning,
  interpreting results).
* If the model requires special setup (e.g., grids, priors), describe it
  here so future contributors know how it fits into the architecture.

Tips and Best Practices
-----------------------

* Keep statistics structs **plain old data**—no pointers or heap
  allocations. Their size should be small so copying between run lengths
  is cheap.
* Use the ``BOCPD_DEBUG_CHECKS`` macro to guard invariants during
  development. Add ``#ifdef`` blocks if you need extra diagnostics that
  should stay out of release builds.
* Benchmark early. Even small allocations inside ``predictive_logpdf`` can
  negate the benefits of the C implementation.

With these steps the new model will behave like any built-in option,
benefiting from the same high-level Python API while leveraging the C
runtime for speed.
