Choosing the Right Observation Model
====================================

Fast-BOCPD packages each conjugate likelihood–prior pair as a Python class
(``fast_bocpd.models``). Selecting the right model ensures changepoint
probabilities reflect the structure of your data. This chapter ties the
statistical assumptions to concrete data characteristics and points out
how our implementation names these concepts.

Quick Decision Path
-------------------

1. **Identify the measurement type**

   * **Binary** (0/1 outcomes) → :class:`fast_bocpd.BernoulliBeta`
   * **Proportion** (successes out of known :math:`N`) →
     :class:`fast_bocpd.BinomialBeta`
   * **Counts** (integers ≥ 0) → :class:`fast_bocpd.PoissonGamma`
   * **Positive continuous** (strictly > 0) → :class:`fast_bocpd.GammaGamma`
   * **General continuous** →
     :class:`fast_bocpd.GaussianNIG` or :class:`fast_bocpd.StudentTNG`

2. **For continuous data, assess tail behaviour**

   * *Well-behaved / normal-like* → Gaussian-NIG
   * *Outliers or fat tails* → Student-t (fixed ``nu``)
   * *Unknown tail heaviness* → Student-t (grid ``nu``)

3. **For high-count Poisson data (λ > ~20)** you may approximate with
   Gaussian-NIG if speed is more critical than an exact count model.

Model Comparison Snapshot
-------------------------

Throughput numbers are taken from :doc:`../benchmarks/results` (100k
observations, online mode) to give a sense of relative cost.

.. list-table::
   :header-rows: 1
   :widths: 18 22 18 18 24

   * - Model
     - Data type
     - Robustness
     - Throughput (obs/s)
     - Typical use
   * - GaussianNIG
     - Continuous
     - Low
     - 25,063
     - Clean sensor data
   * - StudentTNG (fixed ν)
     - Continuous
     - High
     - 21,796
     - Financial returns
   * - StudentTNG (ν grid)
     - Continuous
     - Very high
     - 3,471
     - Unknown tail heaviness
   * - PoissonGamma
     - Counts
     - Medium
     - 21,402
     - Event rates
   * - BernoulliBeta
     - Binary
     - Exact
     - 33,573
     - Success/failure streams
   * - BinomialBeta
     - Proportions (k/N)
     - Exact
     - 14,599
     - Conversion rates
   * - GammaGamma
     - Positive continuous
     - Medium
     - 24,290
     - Durations / amounts

Observation Model Details
-------------------------

Each class mirrors the notation in :doc:`../theory/conjugate_priors` and
maps cleanly to a C implementation under ``fast_bocpd/_c``. Below are the
key considerations and tuning tips per model.

GaussianNIG (``GaussianNIG``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Assumptions:* iid Gaussian within each regime with unknown mean and
variance. Hyperparameters ``mu0``, ``kappa0``, ``alpha0``, ``beta0`` match
the Normal-Inverse-Gamma prior.

*Use when:* data are continuous, roughly bell-shaped, and you value
maximum throughput.

*Tips:*

* Center data around zero and set ``mu0`` accordingly.
* ``kappa0=1`` keeps the prior weak; increasing it enforces a tighter
  belief about the mean.
* Larger ``alpha0`` / ``beta0`` shrink the variance towards a prior guess.

StudentTNG (``StudentTNG``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Assumptions:* Student-t likelihood obtained via a Normal-Gamma prior.
Supports **fixed** ``nu`` or a **grid** of ν values. Our implementation
stores a flag ``is_grid`` so the C layer knows whether to dispatch to
``student_t_ng.c`` or ``student_t_ng_grid.c``.

*Use when:* data contain sporadic outliers or heavy tails.

*Choosing ν:*

* ``nu=1`` behaves like Cauchy (extreme robustness, slower adaptation).
* ``nu=3`` is a good default for financial/operational data.
* Grid mode allows ``nu=[2, 3, 5, 10, 20]`` with optional ``nu_prior`` if
  you want the algorithm to learn the right tail heaviness at runtime.

PoissonGamma (``PoissonGamma``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Assumptions:* Counts per unit time with a Gamma prior on the rate
``lambda``. Parameters ``alpha0``/``beta0`` correspond to prior event
counts and exposure.

*Use when:* you observe integer counts (clicks, failures, arrivals) and
need exact handling of discrete jumps. Offline mode is especially fast for
this model due to vectorised log-factorials in C.

BernoulliBeta (``BernoulliBeta``) & BinomialBeta (``BinomialBeta``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Assumptions:* Binary outcomes or aggregate successes out of ``n_trials``.
These models share identical sufficient statistics (counts of successes
and total trials). Binomial-Beta exposes ``n_trials`` (via
``n_trials`` attribute) and automatically caches ``log_N_factorial`` in the
C layer for numerical stability.

*Use when:* dealing with conversion rates, A/B tests, or thresholded
signals.

GammaGamma (``GammaGamma``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Assumptions:* Positive continuous data with fixed shape parameter ``shape``.
Useful for dwell times, monetary amounts, or any strictly-positive metric.

*Use when:* the distribution is skewed (right tail) but strictly positive.
Because the shape is fixed, choose ``shape=1`` for exponential-like data
or >1 for more symmetric positive data.

Hazard Function Interaction
---------------------------

Observation models specify *how* data behave between changepoints; the
hazard function (``fast_bocpd.hazard.ConstantHazard``) specifies *when*
changepoints occur on average. The two interact via the BOCPD recursion:

* High hazards (small ``lambda_``) expect frequent regime changes. Use
  them when your metrics fluctuate often (e.g., user traffic with daily
  resets).
* Low hazards (large ``lambda_``) assume long, stable runs. Pair them with
  robust models (Student-t) to avoid false positives when rare outliers
  appear.

Practical Recommendations
-------------------------

1. Start with either GaussianNIG or StudentTNG depending on outlier
   expectations. Switch to discrete models only when your data type
   demands it.
2. Keep priors weak (`kappa0 = alpha0 = beta0 = 1`) while prototyping.
   Tighten them only if you have domain knowledge.
3. Use grid Student-t sparingly; reserve it for critical applications
   where robustness matters more than throughput.
4. For proportions with a large denominator (``n_trials >= 50``), consider
   whether a Gaussian approximation is sufficient—Binomial-Beta is exact
   but slower.

For a full statistical comparison and benchmark numbers, refer to
:doc:`../theory/model_comparison` and :doc:`../benchmarks/results`.

**Example:**

.. code-block:: python

   model = PoissonGamma(
       alpha0=1.0,   # Prior shape
       beta0=1.0     # Prior rate
   )

**Real-world use cases:**

- Website clicks per hour
- Server errors per day
- Customer arrivals per minute
- Defects per product batch

**Tuning tips:**

- Prior mean is ``alpha0 / beta0``
- Set this to your expected event rate
- Higher α₀ and β₀ (with same ratio) = stronger prior

Bernoulli (Beta Prior)
~~~~~~~~~~~~~~~~~~~~~~~

**Statistical Model:**

.. math::

   x_t | p &\sim \text{Bernoulli}(p) \\\\
   p &\sim \text{Beta}(\alpha_0, \beta_0)

**When to use:**

- Binary outcomes (success/failure, yes/no)
- Probability estimation (coin flips, conversion)
- Data is 0 or 1

**Strengths:**

- Fastest model (~34,000 obs/sec)
- Perfect for A/B testing changepoint detection
- Simple, interpretable

**Weaknesses:**

Only for binary data

**Example:**

.. code-block:: python

   model = BernoulliBeta(
       alpha0=1.0,   # Prior successes
       beta0=1.0     # Prior failures
   )

**Real-world use cases:**

- Conversion rate changes (user clicked? yes/no)
- Manufacturing defects (pass/fail)
- Medical outcomes (recovered? yes/no)
- Coin fairness testing

**Tuning tips:**

- ``alpha0=beta0=1`` is uniform prior (no preference)
- ``alpha0=beta0=0.5`` is Jeffreys prior (uninformative)
- ``alpha0`` and ``beta0`` can be thought of as "pseudocounts"

Binomial (Beta Prior)
~~~~~~~~~~~~~~~~~~~~~~

**Statistical Model:**

.. math::

   x_t | p, N &\sim \text{Binomial}(N, p) \\\\
   p &\sim \text{Beta}(\alpha_0, \beta_0)

**When to use:**

- Proportion data (k successes out of N trials)
- Batch testing (10 out of 100 users converted)
- N is fixed and known

**Strengths:**

- Generalizes Bernoulli (Bernoulli is Binomial with N=1)
- Fast (~15,000 obs/sec)
- Natural for proportion changepoints

**Example:**

.. code-block:: python

   model = BinomialBeta(
       alpha0=1.0,   # Prior successes
       beta0=1.0,    # Prior failures
       n_trials=10   # Fixed N per observation
   )

**Real-world use cases:**

- Batch conversion rates (10 users, 3 converted → x=3)
- Clinical trials (20 patients, 12 responded → x=12)
- Quality control (sample 50 items, 2 defective → x=2)

**Tuning tips:**

- ``n_trials`` must match your data (every x_t is out of N trials)
- Same prior tuning as Bernoulli

Gamma (Gamma Prior)
~~~~~~~~~~~~~~~~~~~

**Statistical Model:**

.. math::

   x_t | k, \theta &\sim \text{Gamma}(k, \theta) \\\\
   \theta &\sim \text{Gamma}(\alpha_0, \beta_0)

**When to use:**

- Positive continuous data (x > 0)
- Right-skewed distributions
- Waiting times, durations, sizes

**Strengths:**

- Flexible (can model various shapes)
- Fast (~24,000 obs/sec)
- Conjugate prior (efficient updates)

**Weaknesses:**

Requires choosing fixed shape parameter k

**Example:**

.. code-block:: python

   model = GammaGamma(
       alpha0=1.0,   # Prior shape
       beta0=1.0     # Prior rate
   )

**Real-world use cases:**

- Customer lifetime value (always positive, skewed)
- Inter-arrival times (time between events)
- File sizes, transaction amounts
- Rainfall amounts (0 for no rain, positive otherwise)

Common Mistakes to Avoid
-------------------------

1. **Using Gaussian for count data**
   
   ``data = [1, 2, 3, ...]`` with ``GaussianNIG``
   Use ``PoissonGamma`` for counts

2. **Using Poisson for continuous data**

   ``data = [1.5, 2.3, 3.7, ...]`` with ``PoissonGamma``
   Use ``GaussianNIG`` or ``StudentTNG``

3. **Ignoring outliers**

   Financial data with ``GaussianNIG`` (will false alarm on every outlier)
   Use ``StudentTNG`` for robustness

4. **Grid ν without need**

   Using ``nu=[2,3,5,10,20]`` when fixed ``nu=3`` is fine
   Grid mode is 6x slower; use only if tail shape is very uncertain

When in Doubt
-------------

**Start with Student-t (fixed ν=3):**

.. code-block:: python

   model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=3)

It's:
- Robust to outliers
- Fast enough for most applications
- Works for most continuous data

Then experiment:
- If no outliers detected → Try ``GaussianNIG`` (faster)
- If heavy tails suspected → Try grid mode or lower ν
- If data is counts → Switch to ``PoissonGamma``

Next Steps
----------

- :doc:`tuning_parameters` - How to set hyperparameters
- :doc:`interpreting_results` - Understanding model outputs
- :doc:`../theory/conjugate_priors` - Mathematical details
- :doc:`../theory/model_comparison` - Statistical comparison
