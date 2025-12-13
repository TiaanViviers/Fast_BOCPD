C Backend Architecture
======================

This page dives into ``fast_bocpd/_c``—the performance-critical portion of
the project. If you plan to modify the BOCPD core, add a new observation
model, or debug numerical issues, start here.

Source Layout
-------------

``bocpd_core.c``
    Owns the BOCPD state machine, allocation logic, run-length recursion,
    and hazard dispatch.

``*_model.c``
    Each observation model (Gaussian-NIG, Student-t, Poisson-Gamma,
    Bernoulli-Beta, Binomial-Beta, Gamma-Gamma) implements a shared
    interface: ``stats_size``, ``prior_stats``, ``update_stats``,
    ``predictive_logpdf``, and ``copy_stats``. The implementations live in
    separate files so they can be optimized independently.

``hazard.c``
    Currently only implements the constant hazard, but it is structured so
    new hazard families can be added later.

``student_t_ng_grid.c``
    A special case that manages a mixture over degree-of-freedom values.
    It keeps an aligned byte blob per run length containing log-weights
    and per-grid statistics.

Key Concepts
------------

Virtual Tables
    ``bocpd_core`` uses function pointers (``ObsModelVTable``) so that the
    hot loop does not need ``switch`` statements. Adding a new observation
    model simply means filling in another vtable and wiring it into the
    ``init_obs_vtable`` function.

Aligned Statistics
    Each run length owns a blob of bytes sized by
    ``stats_size = round_up_align(model->stats_size, STATS_ALIGNMENT)``. We
    use ``max_align_t`` when available and fall back to 16 bytes to stay
    portable across Linux, macOS, and Windows.

Zero-Copy Buffer Swaps
    ``bocpd_update`` writes into ``new_log_joint``/``new_stats`` while
    reading from the old buffers, then swaps the pointers. No data is
    memcpy’d for normal updates, which keeps cache pressure minimal.

Defensive Programming
~~~~~~~~~~~~~~~~~~~~~

The C layer validates everything:

* ``bocpd_init`` rejects invalid hyperparameters once, up front.
* Observation models guard against ``NaN``, negative counts, or corrupted
  statistics.
* Optional ``BOCPD_DEBUG_CHECKS`` zeroes buffers after reset/update to
  expose uninitialized reads during development.

If you ever see ``-INFINITY`` returned from ``bocpd_update``, it usually
means an observation or stats buffer failed validation; check the relevant
model file.

Adding New Functionality
------------------------

Observation models
    Implement the shared interface, add an enum value to
    ``ObsModelType``, update ``ObsModelParams``/``ctypes`` bindings, and
    register the model inside ``init_obs_vtable``. See
    :doc:`adding_models` for a detailed guide.

Hazard functions
    Follow the pattern in ``hazard.c``: keep parameters in a dedicated
    ``struct``, provide ``*_init`` plus ``log_transition_cp`` and
    ``log_transition_cont`` helpers, then extend the hazard ``switch`` in
    ``bocpd_core.c``.

Memory ownership
    ``bocpd_init`` always zeroes the ``BOCPDState`` struct before use and
    ``bocpd_free`` handles every pointer (including optional grids). When
    you add new allocations, ensure they are set to ``NULL`` in the
    failure paths and freed inside ``bocpd_free``.

Testing
-------

Every model ships with focused C unit tests in ``tests/c_tests`` and
matching Python tests in ``tests/python``. The C tests can be run via
``make test``; they exercise parameter validation, numerical stability,
and change-point recursion. Whenever you modify the C layer, run the tests
and consider enabling ``BOCPD_DEBUG_CHECKS`` to catch latent bugs.
