Your First Detection
====================

This tutorial mirrors  the ``examples/01_quickstart.ipynb`` notebook 
and shows the minimum set of steps required to
detect changepoints with Fast-BOCPD.

1. Create a toy dataset
-----------------------

.. code-block:: python

   import numpy as np
   np.random.seed(7)

   regime_lengths = [120, 150, 90]
   regime_means = [0.0, 1.5, -0.2]

   data = np.concatenate([
       np.random.normal(loc=mu, scale=0.2, size=n)
       for n, mu in zip(regime_lengths, regime_means)
   ])

2. Instantiate a model and hazard
---------------------------------

Fast-BOCPD provides conjugate observation models out of the box. For continuous,
roughly Gaussian data we use :class:`fast_bocpd.models.GaussianNIG`. The hazard
controls expected segment length.

.. code-block:: python

   from fast_bocpd import BOCPD, GaussianNIG, ConstantHazard

   obs_model = GaussianNIG(mu0=0.0, kappa0=1.0, alpha0=0.2, beta0=0.2)
   hazard = ConstantHazard(lambda_=150)  # Expect ≈150 samples per regime
   bocpd = BOCPD(obs_model=obs_model, hazard=hazard, max_run_length=600)

3. Stream observations
----------------------

``BOCPD.update(x)`` returns the posterior run-length distribution and the
probability that the current point is a changepoint.

.. code-block:: python

   cp_probs = []
   map_run_lengths = []

   for x in data:
       posterior_r, cp_prob = bocpd.update(float(x))
       cp_probs.append(cp_prob)
       map_run_lengths.append(bocpd.get_map_run_length())

4. Simple alerting
------------------

Raise an alert whenever ``cp_prob`` crosses a threshold from below or the MAP
run length resets to zero.

.. code-block:: python

   alerts = []
   THRESHOLD = 0.3

   for i, (cp_prob, map_r) in enumerate(zip(cp_probs, map_run_lengths)):
       if cp_prob >= THRESHOLD and map_r <= 2:
           alerts.append(i)

   print(f\"Detected {len(alerts)} changepoints: {alerts}\")

5. Visualize
------------

Plot the time series, ``P(r_t=0)`` and MAP run length to sanity check the
detector. See ``examples/_helpers.py`` for Plotly helpers or use Matplotlib:

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
   ax1.plot(data, color=\"#4E79A7\", linewidth=1.5)
   for idx in alerts:
       ax1.axvline(idx, color=\"red\", linestyle=\"--\", alpha=0.6)
   ax1.set_ylabel(\"Observation\")

   ax2.plot(cp_probs, color=\"#F28E2B\", linewidth=1.5, label=\"P(CP)\")
   ax2_twin = ax2.twinx()
   ax2_twin.plot(map_run_lengths, color=\"#59A14F\", linewidth=1.2, label=\"MAP run length\")
   ax2.set_xlabel(\"Time step\")
   ax2.set_ylabel(\"P(CP)\")
   ax2_twin.set_ylabel(\"MAP run length\")
   plt.tight_layout()

Next steps
----------

* Swap ``GaussianNIG`` with the model that matches your data type (see
  :doc:`../examples/stock_volatility` for Gamma data).
* Use :class:`fast_bocpd.utils.OnlineChangeDetector` to handle thresholds,
  cooldowns, and metadata automatically once you move beyond a prototype.
