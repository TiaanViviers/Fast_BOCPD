Tuning Parameters
=================

Observation models expose statistical hyperparameters, while the hazard
controls changepoint frequency. This guide explains how each knob maps to
the mathematics in :doc:`../theory/conjugate_priors` and how to pick
reasonable defaults in practice.

General Strategy
----------------

1. **Standardise your data** (subtract mean, divide by scale) whenever
   possible. This keeps prior hyperparameters near order-of-one values.
2. **Start with weak priors** so the model learns primarily from data.
3. **Tighten priors** only after you understand typical regimes. Strong
   priors accelerate convergence but can mask real shifts if mis-specified.

Hazard Parameter (``lambda_``)
------------------------------

``fast_bocpd.hazard.ConstantHazard`` uses a single number:

.. math::
   H = 1 / \lambda_.

* ``lambda_=50`` means “expect a changepoint roughly every 50 samples”.
* Small ``lambda_`` ⇒ aggressive reset behaviour.
* Large ``lambda_`` ⇒ long memory; use robust observation models to avoid
  spurious resets when outliers appear.

Observation Model Hyperparameters
---------------------------------

GaussianNIG
~~~~~~~~~~~

.. csv-table::
   :header: Parameter, Controls, Heuristic

   ``mu0``, Prior mean, Set to historical average or 0 if data are centred
   ``kappa0``, Confidence in ``mu0``, Use 1.0 for weak prior; scale up if you know the baseline
   ``alpha0``/``beta0``, Prior on variance, Start with 1.0; larger values lock variance near ``beta0 / (alpha0 - 1)``

StudentTNG
~~~~~~~~~~

Shares ``mu0``, ``kappa0``, ``alpha0``, ``beta0`` with GaussianNIG. The
additional ``nu`` or ``nu`` grid controls tail heaviness:

* ``nu`` between 2 and 5 covers most heavy-tailed scenarios.
* Grid mode: choose 3–5 values spanning plausible tails
  (e.g., ``[2, 3, 5, 10, 20]``) and default to a uniform ``nu_prior``.
* If your data rarely contain outliers, set ``nu`` ≥ 10 to approach the
  Gaussian limit while preserving some robustness.

PoissonGamma
~~~~~~~~~~~~

Hyperparameters have a direct interpretation:

* ``alpha0`` – pseudo-count of events.
* ``beta0`` – pseudo-time (exposure).
* Prior mean rate = ``alpha0 / beta0``.

Guidelines:

* If you expect ~5 events per interval, set ``alpha0=5``, ``beta0=1``.
* For vague priors use ``alpha0=beta0=1``.
* Lower ``beta0`` makes the prior rate higher; increasing both by the same
  factor keeps the mean but adds more “pseudo-observations”.

Bernoulli/Binomial Beta
~~~~~~~~~~~~~~~~~~~~~~~

* ``alpha0`` = prior successes + 1.
* ``beta0`` = prior failures + 1.

Examples:

* Uniform prior (no preference) ⇒ ``alpha0 = beta0 = 1``.
* Historical conversion rate 20% with strong belief ⇒
  ``alpha0=20``, ``beta0=80`` (behaves as if you observed 100 trials).

Binomial-Beta additionally requires ``n_trials`` (renamed ``N`` in the C
layer). Keep it small (≤ 1,000) to avoid extreme binomial coefficients; if
you aggregate more trials, consider rescaling or using multiple time steps.

Gamma-Gamma (Fixed Shape)
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``shape`` (k) – known shape of the Gamma likelihood. ``shape=1`` reduces
  to the exponential distribution; higher values tighten the distribution.
* ``alpha0`` / ``beta0`` – same interpretation as PoissonGamma but applied
  to the rate parameter of the Gamma likelihood.

Practical Workflow
------------------

1. **Prototype with weak priors** (`mu0=0`, `kappa0=1`, `alpha0=beta0=1`,
   `lambda_=100`). Inspect changepoint probabilities via
   ``BOCPD.update`` or ``OnlineChangeDetector``.
2. **Adjust hazard** so changepoint probability matches intuition (e.g.,
   if you expect weekly shifts in hourly data, ``lambda_=24*7``).
3. **Tune observation priors** if the model is too sensitive (increase
   ``kappa0`` or ``alpha0``/``beta0``) or too sluggish (decrease them).
4. **Validate** by replaying historical data and checking that known
   events trigger spikes in ``P(r_t = 0)``.

For a deeper statistical explanation of each hyperparameter, refer back to
:doc:`../theory/conjugate_priors`.
