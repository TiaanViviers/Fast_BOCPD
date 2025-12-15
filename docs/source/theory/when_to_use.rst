When to Use BOCPD
=================

Bayesian Online Changepoint Detection is most effective when you need
probabilistic, real-time detection of regime shifts with calibrated
uncertainty. This section contrasts BOCPD with other popular approaches.

Suitable Scenarios
------------------

* **Streaming/online monitoring** – data arrive sequentially and decisions
  must be made on the fly (manufacturing sensors, telemetry, trading).
* **Piecewise-stationary processes** – each regime can be modelled by a
  conjugate likelihood–prior pair from :doc:`conjugate_priors`.
* **Low-latency alerts** – changepoint probabilities
  :math:`P(r_t=0 \mid x_{1:t})` provide interpretable confidence scores.
* **Limited data per segment** – the Bayesian update works even with very
  few samples because the prior supplies regularisation.

Less Suitable Scenarios
-----------------------

* **Gradual drift** – if changes are smooth rather than abrupt, methods
  such as Kalman filters or Prophet’s trend components may be preferable.
  There are extensions to BOCPD that handle gradual changes, but they are not
  implemented in Fast-BOCPD yet (e.g. different prior/hazard models).
* **Very high-dimensional observations** – the current observation models
  are univariate; multivariate extensions demand custom conjugate pairs.

Comparison to Other Methods
---------------------------

.. list-table::
   :header-rows: 1

   * - Method
     - Strengths
     - Limitations vs BOCPD
   * - CUSUM / Page-Hinkley
     - Simple, few parameters, O(1) memory
     - Tailored to mean shifts, requires handcrafted thresholds, no posterior
   * - Prophet / ARIMA
     - Captures seasonality, trend, regressors
     - Not changepoint detectors per se; changepoints are model components, not probabilistic events
   * - ruptures
     - Rich offline algorithms, many cost functions, visual tools
     - Batch only; recomputes from scratch when new data arrive
   * - promised-ai / dtolpin BOCD
     - Bayesian online detection in other languages
     - Slower throughput (see :doc:`../benchmarks/comparison`)
   * - HMMs / Switching Kalman filters
     - Explicit latent-state models, handle dependence
     - Require predefined number of states and transition matrices

Implementation Mapping
----------------------

* **Observation models** correspond to the likelihood/prior choices in the
  conjugate derivation. Users pick a class such as
  ``GaussianNIG`` or ``PoissonGamma`` rather than specifying raw pdfs.
* **Hazard functions** encapsulate the gap distribution; the current
  ``ConstantHazard`` class matches the geometric prior used in the theory.
* **Outputs** – ``BOCPD.update`` returns both the full posterior
  :math:`P(r_t \mid x_{1:t})` and :math:`P(r_t=0 \mid x_{1:t})`. These
  probabilities can drive downstream alerting or feed into tools like
  ``OnlineChangeDetector`` (see :doc:`../api/index`).

If your application matches the assumptions above and you need calibrated,
online decisions, Fast-BOCPD provides an efficient, theoretically grounded
solution. For purely offline retrospective analysis or models with strong
seasonality, use BOCPD in combination with other techniques (e.g. detrend
with ARIMA, then apply BOCPD to residuals).
