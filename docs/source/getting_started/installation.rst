Installation
============

.. TODO: Fill this in with detailed installation instructions

Requirements
------------

- Python ≥ 3.8
- NumPy ≥ 1.21.0
- C compiler (gcc, clang, or MSVC)

Installation Methods
--------------------

Via pip (recommended)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install fast-bocpd

From source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/TiaanViviers/Fast_BOCPD.git
   cd Fast_BOCPD
   pip install -e .

Troubleshooting
---------------

Compilation errors
~~~~~~~~~~~~~~~~~~

If you see C compilation errors, ensure you have a C compiler installed:

**Linux/macOS:**

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install build-essential
   
   # macOS (install Xcode Command Line Tools)
   xcode-select --install

**Windows:**

Install Microsoft Visual C++ Build Tools.

Import errors
~~~~~~~~~~~~~

If ``import fast_bocpd`` fails, check that the C library compiled:

.. code-block:: bash

   cd fast_bocpd/_c
   make clean
   make lib
   ls -la ../../build/lib/libbocpd.so  # Should exist

Verifying Installation
----------------------

.. code-block:: python

   import fast_bocpd as fb
   print(fb.__version__)
   
   # Quick test
   model = fb.GaussianNIG(0, 1, 1, 1)
   hazard = fb.ConstantHazard(100)
   detector = fb.BOCPD(model, hazard)
   print("Installation successful!")
