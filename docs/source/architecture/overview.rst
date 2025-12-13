Architecture Overview
=====================

Why We Mix Python and C
-----------------------

Bayesian Online Changepoint Detection (BOCPD) is conceptually simple but
computationally expensive: every observation updates an entire run-length
distribution and re-evaluates predictive likelihoods. Pure Python
implementations spend most of their time in interpreter overhead and
temporary allocations, which makes real-time workloads impractical.

Fast-BOCPD keeps the *ergonomics* of Python while implementing the
algorithm in a lower-level language that can control memory layout,
alignment, and vector-friendly loops. The result is a library that feels
like NumPy but runs closer to C speed.

Layered Architecture
--------------------

The project is split into three layers:

1. **User API (Python, ``fast_bocpd/*.py``)**  
   Constructors, validation, NumPy conversions, and friendly errors all
   live here. Nothing in this layer assumes users know about ``ctypes`` or
   raw pointers.

2. **Bindings (Python, ``fast_bocpd/_bindings.py``)**  
   Describes C structs, enums, and function signatures using ``ctypes``.
   This file is deliberately thin: it marshals NumPy buffers into ``ctypes``
   pointers, loads the shared library, and exposes a Pythonic facade for
   the C API.

3. **Core Engine (C99, ``fast_bocpd/_c/*.c``)**  
   Implements BOCPD, observation models, and hazard functions. The key
   entry points are ``bocpd_init``, ``bocpd_update``, and
   ``bocpd_free`` in ``bocpd_core.c``.

High-Level Data Flow
--------------------

1. ``BOCPD`` Python class validates parameters and builds ``ctypes``
   structures.
2. Bindings call ``bocpd_init`` to allocate aligned buffers in C.
3. For every observation, Python hands a ``double`` (or NumPy array) to
   ``bocpd_update``. The C layer updates run-length posteriors in-place.
4. Probabilities flow back as NumPy views without extra copies.

Design Principles
-----------------

* **Safety first** – C code validates inputs aggressively (with optional
  ``BOCPD_DEBUG_CHECKS``) to prevent silent corruption.
* **Zero-cost abstractions** – Virtual tables (function pointers) let us
  plug in new observation models without branches in the hot loop.
* **Ownership clarity** – Anything allocated in C is freed by
  ``bocpd_free``; Python only owns high-level objects and NumPy arrays.
* **Portability** – We target C99 and keep the code POSIX/Windows
  friendly. No SIMD intrinsics or compiler extensions are required.

Where to Go Next
----------------

* :doc:`c_backend` dives into the statistics buffers, hazard functions,
  and BOCPD state machine.
* :doc:`python_bindings` explains how the shared library is loaded and how
  NumPy buffers are marshaled.
* :doc:`adding_models` walks through implementing a new conjugate model.
