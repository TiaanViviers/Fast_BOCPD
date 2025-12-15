Conjugate Priors
================

Fast-BOCPD encapsulates likelihood–prior pairs inside **observation
models**. Each model maintains sufficient statistics for the current run
length and exposes a predictive density used by the BOCPD recursion. The
table below summarises the conjugate pairs currently implemented (all
definitions follow ``examples/math_foundation.md``).

.. list-table::
   :header-rows: 1
   :widths: 18 18 22 22 20

   * - Observation model
     - Likelihood :math:`p(x \mid \theta)`
     - Prior :math:`p(\theta)`
     - Sufficient statistics (per run length)
     - Implementation (files)
   * - Gaussian-NIG
     - Normal with unknown mean :math:`\mu` and variance :math:`\sigma^2`
     - Normal–Inverse-Gamma :math:`(\mu,\sigma^2)`
     - :math:`n`, :math:`\sum x`, :math:`\sum x^2`
     - ``fast_bocpd/_c/gaussian_nig.{c,h}``
   * - Student-t (fixed :math:`\nu`)
     - Heavy-tailed Student-t (derived via Gaussian scale mixture)
     - Normal–Gamma on :math:`(\mu,\tau)`
     - Weighted sums :math:`S_0, S_1, S_2`
     - ``student_t_ng.{c,h}``
   * - Student-t Grid
     - Mixture of Student-t components indexed by :math:`\nu_k`
     - Same Normal–Gamma prior, discrete prior on :math:`\nu`
     - Component stats + log weights
     - ``student_t_ng_grid.{c,h}``
   * - Bernoulli-Beta
     - Bernoulli :math:`x \in \{0,1\}`
     - Beta :math:`\text{Beta}(\alpha_0,\beta_0)`
     - :math:`n`, :math:`\sum x`
     - ``bernoulli_beta.{c,h}``
   * - Binomial-Beta
     - Binomial with :math:`N` trials per observation
     - Beta on success probability
     - :math:`n`, :math:`\sum k`
     - ``binomial_beta.{c,h}``
   * - Poisson-Gamma
     - Poisson counts
     - Gamma on rate :math:`\lambda`
     - :math:`n`, :math:`\sum x`
     - ``poisson_gamma.{c,h}``
   * - Gamma-Gamma (fixed shape)
     - Gamma likelihood with known shape :math:`k`
     - Gamma prior on rate
     - :math:`n`, :math:`\sum x`
     - ``gamma_gamma_fixed_shape.{c,h}``

Why Conjugacy?
--------------

The BOCPD recursion requires evaluating the posterior predictive
:math:`P(x_t \mid x_t^{(r)})`. Conjugate priors yield closed-form updates:

.. math::

   p(\theta \mid x_t^{(r)}) \propto p(x_t^{(r)} \mid \theta) p(\theta),

and the predictive integrates parameters out analytically. In code terms:

* ``*_prior_stats`` initialises sufficient statistics for a new segment.
* ``*_update_stats`` increments the stats when a continuation branch is
  taken.
* ``*_predictive_logpdf`` returns :math:`\log P(x_t \mid x_t^{(r)})`.

The Python classes under ``fast_bocpd.models`` expose the same hyperparameters,
validate user input (finite/positive checks), and translate them to
``ctypes`` structures. The terminology “observation model” emphasises that
the user chooses a pre-packaged likelihood–prior pair instead of writing
custom probability code.

Adding new pairs requires defining the conjugate update, implementing the
five functions above, wiring the model into ``bocpd_core.c``’s virtual
table, and adding a Python wrapper (see :doc:`/architecture/adding_models`).
