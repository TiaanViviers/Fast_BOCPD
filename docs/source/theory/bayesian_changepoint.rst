Bayesian Online Changepoint Detection
=====================================

Fast-BOCPD implements the recursion introduced by Adams & MacKay (2007)
for estimating, in real time, the posterior distribution over the *run
length*—the number of observations since the last changepoint.

For a full derivation, including the assumptions that make the recursion
tractable (run-length Markov property, segment independence, iid within
segments, conjugacy), see ``examples/math_foundation.md``.

Notation
--------

.. list-table::
   :header-rows: 0

   * - :math:`x_{1:t}`
     - Observations up to time :math:`t`.
   * - :math:`r_t`
     - Run length at time :math:`t` (number of samples since the most recent changepoint).
   * - :math:`x_t^{(r)} = x_{t-r+1:t}`
     - Data belonging to the current segment when :math:`r_t = r`.
   * - :math:`p_\text{gap}(g)`
     - Prior over segment lengths (“gap distribution”). Implemented via the hazard function.
   * - :math:`\theta`
     - Latent parameters of the observation model.

Goal
----

Compute :math:`P(r_t \mid x_{1:t})` online. The posterior mass at
:math:`r_t = 0` is the changepoint probability; larger :math:`r_t`
values indicate longer regimes.

Run-Length Recursion
--------------------

Applying the chain rule and marginalising over :math:`r_{t-1}` yields

.. math::

   P(r_t, x_{1:t})
      = \sum_{r_{t-1}} P(r_t \mid r_{t-1}) \,
        P(x_t \mid r_{t-1}, x_t^{(r_{t-1})}) \,
        P(r_{t-1}, x_{1:t-1}),

where the three factors map directly to components in the code base:

1. **Change prior** :math:`P(r_t \mid r_{t-1})` – determined by the hazard
   function. In Fast-BOCPD the hazard object controls this term.
2. **Predictive likelihood**
   :math:`P(x_t \mid r_{t-1}, x_t^{(r_{t-1})})` – integrates out the
   latent parameters :math:`\theta`. This is provided by the observation
   model.
3. **Previous joint** :math:`P(r_{t-1}, x_{1:t-1})` – the recursion anchor
   stored inside the C state (`log_joint` in ``bocpd_core.c``).

After computing :math:`P(r_t, x_{1:t})` we normalise
:math:`P(r_t \mid x_{1:t}) = P(r_t, x_{1:t}) / \sum_{r_t} P(r_t, x_{1:t})`.

Hazard Function
---------------

The hazard encodes the *gap distribution* over segment lengths. Fast-BOCPD
currently ships a constant hazard:

.. math::
   :nowrap:

   \begin{align*}
   H(r) &= \Pr(\text{changepoint at time } t \mid r_{t-1} = r), \\
   P(r_t = 0 \mid r_{t-1} = r) &= H(r), \\
   P(r_t = r_{t-1}+1 \mid r_{t-1} = r) &= 1 - H(r),
   \end{align*}

with :math:`H(r) = 1/\lambda` for constant expected run length
:math:`\lambda`. Different hazard families (e.g. geometric, piecewise,
data-driven) can be added by extending ``hazard.c`` and the Python wrapper
``fast_bocpd.hazard.ConstantHazard``.

Predictive Likelihood
---------------------

For each observation model we use a conjugate prior so the predictive can
be computed in closed form:

.. math::

   P(x_t \mid r_{t-1}, x_t^{(r_{t-1})})
      = \int p(x_t \mid \theta) \, p(\theta \mid x_t^{(r_{t-1})}) \, \mathrm{d}\theta.

When :math:`r_{t-1}=0` (new segment), the prior predictive
:math:`\int p(x_t \mid \theta) p(\theta)\,\mathrm{d}\theta` is used
instead. The observation-model implementations in ``fast_bocpd/_c`` update
the sufficient statistics and evaluate these integrals exactly, avoiding
numerical quadrature.

Algorithm Skeleton
------------------

.. code-block:: text

   initialise run-length posterior: P(r_0 = 0) = 1
   for each observation x_t:
       # compute predictive probabilities
       for r_prev in 0..R:
           pred = obs_model.predictive_logpdf(stats[r_prev], x_t)
           log_joint_cp   = log_joint[r_prev] + log H(r_prev) + pred_using_prior
           log_joint_cont = log_joint[r_prev] + log(1-H(r_prev)) + pred
       # normalise and swap buffers
       posterior = softmax(log_joint_new)
       log_joint = log_joint_new
       stats     = stats_new

Implementation Mapping
----------------------

* **Observation models** – high-level Python classes
  (``fast_bocpd.models``) expose domain-specific parameters (e.g.
  Gaussian mean/precision, Binomial trials). They map directly to the C
  structs in ``fast_bocpd/_c`` and the ``ctypes`` definitions in
  ``fast_bocpd/_bindings.py``.
* **Hazard functions** – wrapped by ``fast_bocpd.hazard``. Each hazard
  struct provides the transition probabilities used in the recursion.
* **State machine** – ``fast_bocpd.core.BOCPD`` loads the shared library,
  allocates the aligned buffers, and calls into ``bocpd_core.c`` for
  updates. The changepoint probability returned to Python is simply
  :math:`P(r_t=0 \mid x_{1:t})`.
