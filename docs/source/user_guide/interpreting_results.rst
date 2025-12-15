Interpreting Results
====================

``BOCPD.update`` and ``BOCPD.batch_update`` return probabilistic summaries
that go beyond binary “changepoint/not changepoint” decisions. This guide
explains how to read those outputs and connect them to downstream logic.

Run-Length Posterior
--------------------

Calling ``posterior_r, cp_prob = bocpd.update(x)`` yields:

* ``posterior_r`` – a NumPy array of length ``max_run_length + 1`` where
  ``posterior_r[r] = P(r_t = r | x_{1:t})``. It represents our belief
  about how long the current regime has lasted.
* ``cp_prob`` – shorthand for ``posterior_r[0]`` (probability that a
  changepoint just occurred).

Understanding the Posterior
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Sharp peak near ``r=0``** – a likely changepoint.
* **Peak drifting rightwards** – the algorithm believes the current regime
  is continuing.
* **Broad posterior** – ambiguous evidence; consider increasing
  ``lambda_`` (hazard) or choosing a more appropriate observation model.

MAP Run Length and Confidence
-----------------------------

``bocpd.get_map_run_length()`` returns the most probable run length
``argmax_r posterior_r[r]``. Pair it with
``bocpd.get_map_confidence()`` (posterior mass at that index) to gauge
certainty. Typical patterns:

* **MAP drops from high value to ≤ 1** – strong signal for a changepoint.
* **MAP confidence < 0.2** – uncertain; consider waiting for more data or
  combining with ``cp_prob`` thresholds.

Batch Output
------------

``bocpd.batch_update(data)`` processes a contiguous array and returns the
sequence of changepoint probabilities. Use it for offline analysis or unit
tests. All internal state (run-length posterior, sufficient statistics) is
carried over so you can call ``batch_update`` repeatedly on streaming
chunks.

Practical Detection Rules
-------------------------

`fast_bocpd.utils.OnlineChangeDetector` demonstrates two common heuristics:

1. **Threshold on ``cp_prob``** – emit an alert whenever
   ``posterior_r[0]`` crosses ``min_cp_prob`` (with hysteresis to avoid
   duplicate alerts).
2. **MAP reset** – fire when the MAP run length collapses from a large
   value to near zero, even if ``cp_prob`` is modest. This captures subtle
   shifts where probability mass spreads across small run lengths.

You can mix and match:

.. code-block:: python

   posterior, cp_prob = bocpd.update(x)
   if cp_prob > 0.3 or bocpd.get_map_run_length() <= 1:
       handle_changepoint()

Visualising Results
-------------------

Plotting ``cp_prob`` over time shows spikes where changepoints likely
occur. Overlay ``posterior_r`` heatmaps or ``map_run_length`` to inspect
regime durations. Because ``posterior_r`` is a proper probability
distribution you can also compute summary statistics (expected run length,
variance) to quantify uncertainty.

Exporting State
---------------

* ``bocpd.get_posterior()`` – copy the current posterior into a NumPy
  array without triggering an update.
* ``bocpd.reset()`` – restart the process (useful between independent
  experiments).
* ``bocpd.close()`` or context manager – ensure C resources are freed when
  you are done.

Interpreting False Positives/Negatives
--------------------------------------

If you observe frequent false positives:

* Lower the hazard ``H=1/lambda_`` (longer expected regimes).
* Increase prior strength (larger ``kappa0``, ``alpha0``, ``beta0``).
* Choose a more robust observation model (Student-t instead of Gaussian).

If true changepoints are missed:

* Increase hazard (smaller ``lambda_``).
* Use grid Student-t if outliers mask shifts.
* Ensure ``max_run_length`` is large enough; otherwise longer regimes get
  truncated and you may lose context.

Use the probability outputs, not just binary decisions, to build nuanced
alerting logic tailored to your domain.
