Python–C Bindings
=================

Fast-BOCPD uses ``ctypes`` rather than Cython or cffi. This keeps the
build simple (no code generation step) and works everywhere CPython does.
This page outlines how the bindings are structured so you can extend them
confidently.

Library Loading
---------------

``fast_bocpd/_bindings.py`` attempts three loading strategies:

1. Import the packaged extension module (``fast_bocpd._core``) when the
   project is installed via ``pip``.
2. Use the development shared library from ``build/lib/libbocpd.so`` if
   you built the C code via ``make``.
3. Fallback to a manually built ``libbocpd.{so,dylib,dll}`` inside
   ``fast_bocpd/_c`` for power users.

If none of those succeed, ``_lib`` is ``None`` and the Python layer raises
an informative error when you try to instantiate ``BOCPD``.

Struct Definitions
------------------

Every ``struct`` and ``union`` from the C headers is reproduced as a
``ctypes`` type. The fields must match both name and order, so we keep the
``_bindings.py`` definitions adjacent to the canonical C headers. When you
add a new model:

* Update the C header.
* Update the ``ctypes`` class.
* Expose any helper enums or constants needed by Python.

Function Signatures
-------------------

``ctypes`` needs to know both argument and return types. We wrap each
exported C function with a Python helper that:

1. Raises an exception if the shared library failed to load.
2. Converts Python/Numpy inputs into ``ctypes`` pointers.
3. Calls the C routine and checks the return code.

For example, ``bocpd_init`` receives ``ctypes`` structures for
``ObsModelParams`` and ``HazardParams`` plus an integer ``max_run_length``.
The helper surfaces ``ValueError`` or ``RuntimeError`` when initialization
fails, mirroring the semantics of the Python API.

NumPy Interoperability
----------------------

Updates are fed as ``np.ndarray`` objects wherever possible. We rely on
the fact that NumPy ``double`` arrays expose `.ctypes` pointers, so we can
avoid copying data between Python and C. The bindings ensure buffers are
contiguous and of the correct dtype before calling into C.

Memory Ownership
----------------

The ``BOCPD`` Python class owns a ``ctypes`` pointer to ``BOCPDState``.
When the object is garbage collected (or used as a context manager) we
call ``bocpd_free``. No other C pointers escape to user code. This helps
prevent double-free bugs and makes it safe to expose convenience methods
such as ``get_posterior`` or ``batch_update``.

Troubleshooting
---------------

* **``OSError: cannot load library``** – Build the C extension via
  ``make build`` or ``pip install .`` so the shared library exists.
* **Struct mismatch** – If you add/remove fields in C but forget to update
  the ``ctypes`` definitions, you’ll usually see garbage values or crashes
  during ``bocpd_init``. Keep the definitions in sync.
* **NumPy dtype errors** – The bindings expect ``float64`` arrays for data
  and ``int32``/``int64`` where appropriate. Use ``np.asarray`` with the
  correct dtype in the Python layer before invoking the bindings.
