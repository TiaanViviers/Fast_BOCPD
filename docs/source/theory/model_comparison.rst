Model Comparison
================

Choosing an observation model amounts to picking the likelihood–prior pair
that best matches your data. The following table summarises the trade-offs.

.. list-table::
   :header-rows: 1
   :widths: 18 25 18 18 21

   * - Model
     - Data assumptions
     - Robustness
     - Computational cost
     - Recommended use cases
   * - Gaussian-NIG
     - Continuous, approximately Gaussian
     - Moderate (sensitive to outliers)
     - Fast (baseline)
     - Sensor drift, financial spreads, streaming KPIs
   * - Student-t (fixed ν)
     - Continuous with occasional outliers
     - High (ν controls tail weight)
     - ~20% slower than Gaussian
     - Industrial monitoring, heavy-tailed returns
   * - Student-t (ν grid)
     - Continuous, unknown tail weight
     - Very high (adapts ν online)
     - 6–7× slower (mixture update)
     - Critical systems where robustness outweighs speed
   * - Poisson-Gamma
     - Non-negative integer counts
     - Moderate (handles overdispersion)
     - Comparable to Student-t
     - Event counts, telemetry hits, arrivals
   * - Bernoulli-Beta
     - Binary indicators
     - High (exact conjugacy)
     - Fastest model
     - Success/failure streams, anomaly flags
   * - Binomial-Beta
     - Counts out of N trials per time step
     - High
     - Midrange (binomial coefficients)
     - Conversion rates, aggregated proportions
   * - Gamma-Gamma
     - Positive continuous values (rates/durations)
     - Moderate (depends on fixed shape k)
     - Slightly slower than Gaussian
     - Transaction sizes, waiting times

Hazard Interaction
------------------

The hazard controls how aggressively we expect changepoints. A low
:math:`\lambda` => frequent changes => shorter regimes. Observation models
rely on sufficient statistics whose variance scales with run length.
Consequently:

* Heavy-tailed models (Student-t) tolerate higher :math:`\lambda`
  (longer regimes) without false alarms.
* Discrete models (Bernoulli/Binomial/Poisson) are less sensitive to the
  hazard choice because their statistics saturate quickly (counts).

Implementation Notes
--------------------

* Each Python class in ``fast_bocpd.models`` mirrors the C implementation
  to keep parameter names consistent between theory and code.
* Hazard functions live in ``fast_bocpd.hazard``; currently only the
  constant hazard is exposed, but the recursion supports any
  :math:`P(r_t \mid r_{t-1})`.
* Switching models requires only substituting the Python wrapper; the C
  engine handles the rest via virtual dispatch.
