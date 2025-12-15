Benchmark Results
=================

All measurements below were captured using the methodology described in
:doc:`methodology`. Unless stated otherwise:

* Expected run length λ = 150
* 2 warm-up runs, 10 timed runs
* Dataset sizes: 1,000 / 10,000 / 100,000 observations

Summary (100k Observations, Online Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table::
   :header: Model, Throughput (obs/s), Description

   Bernoulli-Beta, 33,573, Binary outcomes (fastest model)
   Gaussian-NIG, 25,063, Continuous data (mean & variance unknown)
   Gamma-Gamma, 24,290, Positive continuous amounts/durations
   Student-t (Fixed ν), 21,796, Robust continuous w/ known ν
   Poisson-Gamma, 21,402, Event counts per interval
   Binomial-Beta, 14,599, k successes out of N trials
   Student-t (Grid ν), 3,471, Robust continuous w/ ν inference

Offline mode delivers ~1.3–1.5× higher throughput for every model due to
reduced Python call overhead and better cache locality.

Gaussian-NIG
------------

Use case: general-purpose continuous data with unknown mean and variance.

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 0.0322, 31,009, 0.1%
   1k, Offline, 0.0228, 43,835, 1.7%
   10k, Online, 0.3838, 26,057, 0.1%
   10k, Offline, 0.2882, 34,699, 0.1%
   100k, Online, 3.9899, 25,063, 0.9%
   100k, Offline, 3.0314, 32,988, 0.5%

Student-t (Fixed ν)
-------------------

Use case: robust changepoint detection when degrees of freedom are known.

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 0.0377, 26,504, 0.5%
   1k, Offline, 0.0280, 35,714, 0.2%
   10k, Online, 0.4507, 22,189, 0.3%
   10k, Offline, 0.3520, 28,409, 0.5%
   100k, Online, 4.5880, 21,796, 0.2%
   100k, Offline, 3.6040, 27,747, 0.1%

Student-t (Grid ν)
------------------

Use case: robust detection when ν is unknown and inferred from a grid.

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 0.2261, 4,422, 0.2%
   1k, Offline, 0.2150, 4,652, 0.4%
   10k, Online, 2.8238, 3,541, 0.1%
   10k, Offline, 2.7136, 3,685, 0.1%
   100k, Online, 28.8063, 3,471, 0.5%
   100k, Offline, 27.7352, 3,606, 0.1%

Bernoulli-Beta
--------------

Use case: binary observations (success/failure, on/off).

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 0.0281, 35,608, 0.5%
   1k, Offline, 0.0222, 45,013, 0.2%
   10k, Online, 0.2952, 33,871, 1.8%
   10k, Offline, 0.2468, 40,521, 0.1%
   100k, Online, 2.9786, 33,573, 1.0%
   100k, Offline, 2.5797, 38,765, 1.0%

Binomial-Beta
-------------

Use case: proportion data (k successes out of N trials).

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 0.0574, 17,422, 0.3%
   1k, Offline, 0.0429, 23,314, 0.2%
   10k, Online, 0.6762, 14,788, 0.2%
   10k, Offline, 0.5307, 18,843, 0.2%
   100k, Online, 6.8500, 14,599, 0.1%
   100k, Offline, 5.3848, 18,571, 0.1%

Poisson-Gamma
-------------

Use case: event counts per time interval.

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 0.0399, 25,039, 0.5%
   1k, Offline, 0.0261, 38,249, 8.0%
   10k, Online, 0.4592, 21,777, 0.4%
   10k, Offline, 0.3212, 31,131, 0.1%
   100k, Online, 4.6725, 21,402, 0.1%
   100k, Offline, 3.2575, 30,698, 0.1%

Gamma-Gamma (Fixed Shape)
-------------------------

Use case: positive continuous data (durations, lifetimes, transaction amounts).

.. csv-table::
   :header: Size, Mode, Median (s), Throughput (obs/s), CV%

   1k, Online, 0.0340, 29,448, 0.2%
   1k, Offline, 0.0243, 41,103, 0.2%
   10k, Online, 0.3908, 25,586, 0.1%
   10k, Offline, 0.2950, 33,894, 1.0%
   100k, Online, 4.1170, 24,290, 0.1%
   100k, Offline, 3.1020, 32,237, 0.1%

Key Observations
----------------

* **Most models sustain 20k–35k obs/s** even at 100k points.
* **Offline mode** yields 30–50% better throughput thanks to reduced
  Python overhead.
* **CV% is usually < 1%**, indicating stable, repeatable performance
  across runs.
* **Student-t Grid** trades speed for robustness; expect a 6–7× slowdown
  versus fixed-ν.
