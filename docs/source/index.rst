Fast-BOCPD Documentation
========================

**The world's fastest Bayesian online changepoint detection library.**

Fast-BOCPD is 10-1,500x faster than competing implementations while providing a simple, Pythonic API for detecting regime changes in time series data.

.. image:: https://img.shields.io/pypi/v/fast-bocpd.svg
   :target: https://pypi.org/project/fast-bocpd/
   :alt: PyPI Version

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/TiaanViviers/Fast_BOCPD/blob/main/LICENSE
   :alt: License

Key Features
------------

🚀 **Blazing Fast**
   - 14,000-43,000 observations/sec (depending on model)
   - 10x faster than optimized Cython implementations
   - 28x faster than Rust implementations
   - 1,500x faster than PyTorch implementations

📊 **7 Probabilistic Models**
   - Gaussian, Student-t (outlier-robust)
   - Poisson (count data), Bernoulli/Binomial (binary/proportion data)
   - Gamma (positive continuous data)
   - All with conjugate priors for analytical updates

🔄 **Flexible Processing**
   - Online mode: Process data as it arrives (streaming)
   - Offline mode: Batch processing with 2-3x speedup
   - Smart truncation prevents O(n²) memory explosion

🐍 **Pure Python Interface**
   - Intuitive API that "just works"
   - High-performance C backend (transparent to users)
   - Full NumPy integration

Quick Example
-------------

Detect changepoints in 5 lines of code:

.. code-block:: python

   import fast_bocpd as fb
   import numpy as np

   # Generate data with a changepoint at t=100
   data = np.concatenate([np.random.normal(0, 1, 100),
                          np.random.normal(5, 1, 100)])

   # Detect changepoints
   model = fb.GaussianNIG(mu0=0, kappa0=1, alpha0=1, beta0=1)
   hazard = fb.ConstantHazard(lambda_=100)
   detector = fb.BOCPD(model, hazard)
   
   results = detector.batch_update(data)
   changepoints = results.get_changepoints(threshold=0.5)
   
   print(f"Detected changepoint at: {changepoints}")
   # Output: Detected changepoint at: [100]

Installation
------------

.. code-block:: bash

   pip install fast-bocpd

Or from source:

.. code-block:: bash

   git clone https://github.com/TiaanViviers/Fast_BOCPD.git
   cd Fast_BOCPD
   pip install -e .

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/first_detection

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/choosing_model
   user_guide/tuning_parameters
   user_guide/interpreting_results
   user_guide/performance_tips

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Theory & Background

   theory/bayesian_changepoint
   theory/conjugate_priors
   theory/model_comparison
   theory/when_to_use

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/stock_volatility
   examples/sensor_monitoring
   examples/ab_testing

.. toctree::
   :maxdepth: 2
   :caption: Performance

   benchmarks/methodology
   benchmarks/results
   benchmarks/comparison

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   architecture/overview
   architecture/c_backend
   architecture/python_bindings
   architecture/adding_models

Community & Support
-------------------

- **GitHub Issues**: `Report bugs or request features <https://github.com/TiaanViviers/Fast_BOCPD/issues>`_
- **Discussions**: `Ask questions <https://github.com/TiaanViviers/Fast_BOCPD/discussions>`_
- **Citation**: See :ref:`citation`

License
-------

Fast-BOCPD is released under the MIT License. See the `LICENSE <https://github.com/TiaanViviers/Fast_BOCPD/blob/main/LICENSE>`_ file for details.

.. _citation:

Citation
--------

If you use Fast-BOCPD in your research, please cite:

.. code-block:: bibtex

   @software{fastbocpd2025,
     title = {Fast-BOCPD: High-Performance Bayesian Online Changepoint Detection},
     author = {Tiaan Viviers},
     year = {2025},
     url = {https://github.com/TiaanViviers/Fast_BOCPD},
     version = {1.0.0}
   }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
