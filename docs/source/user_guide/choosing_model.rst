Choosing the Right Model
=========================

Fast-BOCPD supports 7 different observation models. The right choice depends on your data type and characteristics.

Quick Decision Tree
-------------------

Use this flowchart to quickly choose a model:

1. **What type of data do you have?**
   
   - **Continuous** (prices, temperatures, measurements) → Go to step 2
   - **Count data** (events per hour, clicks, failures) → :class:`~fast_bocpd.PoissonGamma`
   - **Binary** (yes/no, success/fail, 0/1) → :class:`~fast_bocpd.BernoulliBeta`
   - **Proportions** (conversion rates, percentages with known N) → :class:`~fast_bocpd.BinomialBeta`
   - **Positive continuous** (durations, sizes, strictly > 0) → :class:`~fast_bocpd.GammaGamma`

2. **For continuous data: Does your data have outliers?**
   
   - **Yes** (or unsure) → :class:`~fast_bocpd.StudentTNG`
   - **No** (clean, normally distributed) → :class:`~fast_bocpd.GaussianNIG`

3. **For count data with large λ (mean > 20):**
   
   - Can approximate with :class:`~fast_bocpd.GaussianNIG` (faster)
   - Or use :class:`~fast_bocpd.PoissonGamma` (exact, slower)

Model Comparison Table
----------------------

+------------------+------------------+----------------+------------------+-------------+
| Model            | Data Type        | Robustness     | Speed (obs/sec)  | Use When    |
+==================+==================+================+==================+=============+
| GaussianNIG      | Continuous       | Low            | ~25,000          | Clean data  |
+------------------+------------------+----------------+------------------+-------------+
| StudentTNG       | Continuous       | **High**       | ~22,000          | Outliers    |
| (fixed ν)        |                  |                |                  |             |
+------------------+------------------+----------------+------------------+-------------+
| StudentTNG       | Continuous       | **Very High**  | ~3,500           | Unknown     |
| (grid ν)         |                  |                |                  | tail shape  |
+------------------+------------------+----------------+------------------+-------------+
| PoissonGamma     | Count (≥0)       | Medium         | ~21,000          | Small λ     |
+------------------+------------------+----------------+------------------+-------------+
| BernoulliBeta    | Binary (0/1)     | N/A            | ~34,000          | Coin flips  |
+------------------+------------------+----------------+------------------+-------------+
| BinomialBeta     | Proportion (k/N) | N/A            | ~15,000          | Conversion  |
|                  |                  |                |                  | rates       |
+------------------+------------------+----------------+------------------+-------------+
| GammaGamma       | Positive         | Medium         | ~24,000          | Durations   |
|                  | continuous       |                |                  |             |
+------------------+------------------+----------------+------------------+-------------+

Detailed Model Guide
--------------------

Gaussian (Normal-Inverse-Gamma)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Statistical Model:**

.. math::

   x_t | \mu, \sigma^2 &\sim \mathcal{N}(\mu, \sigma^2) \\\\
   \mu, \sigma^2 &\sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)

**When to use:**

- Continuous data with no outliers
- Data is approximately bell-shaped (normal distribution)
- You need maximum speed
- Variance is roughly constant within segments

**Strengths:**

✅ Fastest model (~25,000 obs/sec)
✅ Mathematically elegant (conjugate prior)
✅ Well-understood, widely used

**Weaknesses:**

❌ Very sensitive to outliers (even 1-2 outliers can trigger false alarms)
❌ Assumes constant variance (not great for volatility changes)

**Example:**

.. code-block:: python

   model = GaussianNIG(
       mu0=0.0,      # Prior mean (center data around 0)
       kappa0=1.0,   # Prior precision (1.0 = weak prior)
       alpha0=1.0,   # Prior shape for variance
       beta0=1.0     # Prior scale for variance
   )

**Real-world use cases:**

- Temperature sensor data
- Quality control measurements (stable process)
- Pre-processed financial returns (outliers removed)
- Any "well-behaved" continuous data

**Tuning tips:**

- Set ``mu0`` to the expected mean of your data
- Start with ``kappa0=1.0`` (weak prior), increase if you have strong prior knowledge
- ``alpha0`` and ``beta0`` control variance prior: higher values = stronger prior

Student-t (Normal-Gamma)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Statistical Model:**

.. math::

   x_t | \mu, \sigma^2, \nu &\sim \text{Student-t}(\mu, \sigma^2, \nu) \\\\
   \mu, \sigma^2 &\sim \text{NG}(\mu_0, \kappa_0, \alpha_0, \beta_0)

**When to use:**

- Continuous data that may have outliers
- Financial data (returns often have fat tails)
- Sensor data with occasional glitches
- When robustness is more important than speed

**Strengths:**

✅ Robust to outliers (heavy tails)
✅ Still very fast (~22,000 obs/sec for fixed ν)
✅ Can infer tail heaviness from data (grid mode)
✅ Degrades gracefully to Gaussian as ν→∞

**Weaknesses:**

❌ Grid mode is slower (~3,500 obs/sec)
❌ One more parameter to tune (ν)

**Two modes:**

1. **Fixed ν** (when you know tail shape):

.. code-block:: python

   model = StudentTNG(
       mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0,
       nu=3.0  # Fixed degrees of freedom
   )

2. **Grid ν** (let model choose):

.. code-block:: python

   model = StudentTNG(
       mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0,
       nu=[2, 3, 5, 10, 20]  # Try multiple ν values
   )

**Choosing ν (degrees of freedom):**

- ``ν = 1``: Cauchy distribution (extreme outliers, very heavy tails)
- ``ν = 2-3``: Financial returns, heavy-tailed data
- ``ν = 4-6``: Moderate outliers
- ``ν = 10-20``: Mild outliers, closer to Gaussian
- ``ν → ∞``: Approaches Gaussian (no extra robustness)

**Rule of thumb:** Start with ``nu=3`` or ``nu=[2, 3, 5, 10]`` for grid mode.

**Real-world use cases:**

- Stock returns (fat tails)
- Network latency (occasional spikes)
- User engagement metrics (power users create outliers)
- Any data where "rare but extreme" events occur

Poisson (Gamma Prior)
~~~~~~~~~~~~~~~~~~~~~~

**Statistical Model:**

.. math::

   x_t | \lambda &\sim \text{Poisson}(\lambda) \\\\
   \lambda &\sim \text{Gamma}(\alpha_0, \beta_0)

**When to use:**

- Count data (non-negative integers)
- Events per time period
- Rate estimation problems
- λ < 20 (for larger λ, Gaussian is faster and equivalent)

**Strengths:**

✅ Exact model for count data (no approximation)
✅ Fast (~21,000 obs/sec)
✅ Natural for event rate changes

**Weaknesses:**

❌ Only for count data (not continuous)
❌ Assumes events are independent

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

✅ Fastest model (~34,000 obs/sec)
✅ Perfect for A/B testing changepoint detection
✅ Simple, interpretable

**Weaknesses:**

❌ Only for binary data

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

✅ Generalizes Bernoulli (Bernoulli is Binomial with N=1)
✅ Fast (~15,000 obs/sec)
✅ Natural for proportion changepoints

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

✅ Flexible (can model various shapes)
✅ Fast (~24,000 obs/sec)
✅ Conjugate prior (efficient updates)

**Weaknesses:**

❌ Requires choosing fixed shape parameter k

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
   
   ❌ ``data = [1, 2, 3, ...]`` with ``GaussianNIG``
   ✅ Use ``PoissonGamma`` for counts

2. **Using Poisson for continuous data**
   
   ❌ ``data = [1.5, 2.3, 3.7, ...]`` with ``PoissonGamma``
   ✅ Use ``GaussianNIG`` or ``StudentTNG``

3. **Ignoring outliers**
   
   ❌ Financial data with ``GaussianNIG`` (will false alarm on every outlier)
   ✅ Use ``StudentTNG`` for robustness

4. **Grid ν without need**
   
   ❌ Using ``nu=[2,3,5,10,20]`` when fixed ``nu=3`` is fine
   ✅ Grid mode is 6x slower; use only if tail shape is very uncertain

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
