Quick Start
===========

This guide will get you up and running with Fast-BOCPD in 5 minutes.

Installation
------------

Install via pip:

.. code-block:: bash

   pip install fast-bocpd

Or install from source:

.. code-block:: bash

   git clone https://github.com/TiaanViviers/Fast_BOCPD.git
   cd Fast_BOCPD
   pip install -e .

Basic Usage
-----------

Let's detect a changepoint in some synthetic data.

Step 1: Generate Test Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   # Create data with a mean shift at t=100
   segment1 = np.random.normal(0, 1, 100)   # Mean = 0
   segment2 = np.random.normal(5, 1, 100)   # Mean = 5
   data = np.concatenate([segment1, segment2])

Step 2: Set Up the Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import fast_bocpd as fb
   
   # Choose a model (Gaussian for continuous data)
   model = fb.GaussianNIG(
       mu0=0.0,      # Prior mean
       kappa0=1.0,   # Prior precision
       alpha0=1.0,   # Prior shape
       beta0=1.0     # Prior scale
   )
   
   # Choose a hazard function (constant = memoryless)
   hazard = fb.ConstantHazard(lambda_=100)  # Expected run length
   
   # Create detector
   detector = fb.BOCPD(model, hazard, max_run_length=250)

Step 3: Detect Changepoints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Process data (offline/batch mode)
   results = detector.batch_update(data)
   
   # Extract changepoints
   changepoints = results.get_changepoints(threshold=0.5)
   print(f"Detected changepoints: {changepoints}")
   # Output: Detected changepoints: [100]

Visualizing Results
-------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot data with detected changepoints
   plt.figure(figsize=(12, 4))
   plt.plot(data, 'k-', alpha=0.5, label='Data')
   
   for cp in changepoints:
       plt.axvline(cp, color='r', linestyle='--', 
                   label='Detected changepoint')
   
   plt.xlabel('Time')
   plt.ylabel('Value')
   plt.legend()
   plt.title('Changepoint Detection Results')
   plt.show()

Online (Streaming) Mode
-----------------------

For real-time applications, use online mode:

.. code-block:: python

   from fast_bocpd import OnlineChangeDetector
   
   # Create online wrapper
   online_detector = OnlineChangeDetector(detector)
   
   # Process data point-by-point
   for t, x in enumerate(data):
       result = online_detector.update(x)
       
       # Check for changepoint
       if result.changepoint_detected(threshold=0.5):
           print(f"Changepoint detected at t={t}")

Next Steps
----------

- :doc:`../user_guide/choosing_model` - Learn which model to use for your data
- :doc:`../user_guide/tuning_parameters` - Understand hyperparameters
- :doc:`../examples/stock_volatility` - See a real-world example
- :doc:`../api/index` - Full API reference

Common Mistakes
---------------

**Wrong model for data type**
   Use ``PoissonGamma`` for count data, not ``GaussianNIG``

**Lambda too small**
   If ``lambda_=10`` but changepoints are rare (every 1000 points),
   the detector will be too sensitive

**Not resetting detector**
   Call ``detector.reset()`` between independent datasets

**Forgetting threshold**
   ``get_changepoints()`` requires a threshold (typically 0.3-0.7)
