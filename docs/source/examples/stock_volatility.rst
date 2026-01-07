Stock Volatility Detection
==========================

Real-world example based on the :doc:`../examples-notebooks <../index>`
notebook ``examples/06_real_world_example.ipynb``. We use Fast-BOCPD to
monitor the US30 (Dow Jones Industrial Average) daily realized volatility
from 2008 to 2025 and automatically detect major regime shifts such as the
2008 financial crisis and COVID-19 crash.

Why volatility?
---------------

* Data is naturally **positive and skewed**, making it a good showcase for the
  :class:`~fast_bocpd.models.GammaGamma` observation model.
* Regimes last from months to years, which stresses the hazard / run-length
  configuration.
* Finance teams care about both historical back-testing and live alerting.

Dataset
-------

* ``examples/us30_vol.csv`` contains normalized daily volatility (already scaled
  by price) from Jan-2008 to Jan-2025.
* We split chronologically: **2008–2020** for calibration and **2021–2025** for
  live simulation—no lookahead.

Training phase (2008–2020)
--------------------------

1. **Estimate Gamma parameters.**
   Method-of-moments gives a shape ≈ 1.06 and rate ≈ 314.3. We wrap those into
   ``GammaGamma(alpha0=2, beta0=alpha0/rate, shape=shape)``—a weak prior centred
   on the empirical rate.
2. **Hazard exploration.**
   Compare ``lambda`` in {80, 120, 150, 200}. ``lambda = 150`` balances
   sensitivity and stability; ``max_run_length = 3 × lambda`` handles long calm
   stretches.
3. **Visualization.**
   The notebook plots segment means and MAP run length to verify that large
   spikes (Lehman, COVID) trigger resets while the long 2010–2019 calm period is
   kept as a single regime.

Streaming deployment (2021–2025)
--------------------------------

* Fresh detector configured with the chosen hyperparameters. We call
  :class:`fast_bocpd.utils.OnlineChangeDetector` with ``min_cp_prob = 0.3`` so
  alerts fire when ``P(r_t = 0)`` crosses 30% or when the MAP run length drops
  sharply.
* Stream each day sequentially. Every detection logs the timestamp, posterior
  probability, and previous segment duration:

  .. code-block:: text

     CHANGEPOINT DETECTED
       Date: 2020-03-16
       Index: 271
       P(CP): 85.0%
       Previous segment: 522 days

* ``get_segments()`` and ``get_map_history()`` power dashboards that measure how
  long the market stayed calm vs. turbulent.

How to reuse
------------

1. Replace ``us30_vol.csv`` with your own positive-valued series (e.g. demand,
   latency, volatility of another symbol).
2. Redo the Gamma moment-fit and hazard sweep; the notebook exposes helper cells
   for re-fitting the model.
3. Plug your alerting backend into ``OnlineChangeDetector``’s stream loop—each
   :class:`fast_bocpd.utils.Changepoint` carries probability, segment length, and
   optional metadata for auditing.

Run the example
---------------

.. code-block:: bash

   jupyter lab examples/06_real_world_example.ipynb

Key takeaways
-------------

* **Model choice matters.** Positive, heavy-tailed volatility is better served
  by ``GammaGamma`` than Gaussian models.
* **Hazard tuning is crucial.** ``lambda`` encodes expected regime duration;
  long calm periods require higher ``max_run_length`` to avoid false resets.
* **Online detector = production ready.** ``min_cp_prob``, cooldown, and metadata
  handling make deployments straightforward once the parameters are calibrated.
