# Fast-BOCPD Documentation

This directory contains all documentation source files. Sphinx generates the final HTML/PDF from these sources.

## Directory Structure

```
docs/
├── source/                    # Sphinx source files (what YOU write)
│   ├── index.rst             # Homepage
│   ├── conf.py               # Sphinx configuration
│   │
│   ├── getting_started/      # TUTORIALS (learning-oriented)
│   │   ├── installation.rst
│   │   ├── quickstart.rst
│   │   └── first_detection.rst
│   │
│   ├── user_guide/           # HOW-TO GUIDES (problem-oriented)
│   │   ├── choosing_model.rst
│   │   ├── tuning_parameters.rst
│   │   ├── interpreting_results.rst
│   │   └── performance_tips.rst
│   │
│   ├── api/                  # API REFERENCE (auto-generated)
│   │   ├── index.rst
│   │   ├── bocpd.rst         # BOCPD class
│   │   ├── models.rst        # All model classes
│   │   ├── hazards.rst       # Hazard functions
│   │   └── utils.rst         # Utility functions
│   │
│   ├── theory/               # EXPLANATION (understanding-oriented)
│   │   ├── bayesian_changepoint.rst
│   │   ├── conjugate_priors.rst
│   │   ├── model_comparison.rst
│   │   └── when_to_use.rst
│   │
│   ├── examples/             # WORKED EXAMPLES
│   │   ├── stock_volatility.rst
│   │   ├── sensor_monitoring.rst
│   │   └── ab_testing.rst
│   │
│   ├── architecture/         # FOR CONTRIBUTORS
│   │   ├── overview.rst
│   │   ├── c_backend.rst
│   │   ├── python_bindings.rst
│   │   └── adding_models.rst
│   │
│   ├── benchmarks/           # PERFORMANCE
│   │   ├── methodology.rst
│   │   ├── results.rst
│   │   └── comparison.rst
│   │
│   └── _static/              # Images, CSS, JS
│       ├── logo.png
│       ├── custom.css
│       └── plots/
│
├── build/                     # Generated output (DON'T commit to git)
│   └── html/                 # Final website
│       ├── index.html
│       ├── api/
│       ├── user_guide/
│       └── ...
│
├── Makefile                  # Build commands (auto-generated)
└── make.bat                  # Windows build (auto-generated)
```

## What Goes Where?

### 1. Getting Started (Tutorials)
**Goal:** Get users from zero to working detection in 5 minutes.

**Files YOU write:**
- `installation.rst` - pip install, troubleshooting
- `quickstart.rst` - 10-line example
- `first_detection.rst` - Step-by-step first project

**Style:** Friendly, assumes no prior knowledge, lots of code examples.

### 2. User Guide (How-To)
**Goal:** Answer "How do I do X?" questions.

**Files YOU write:**
- `choosing_model.rst` - Decision tree: Gaussian vs Student-t vs Poisson...
- `tuning_parameters.rst` - What are mu0, kappa0, alpha0, beta0? How to set them?
- `interpreting_results.rst` - What does run-length probability mean?
- `performance_tips.rst` - When to use batch vs online, max_run_length tuning

**Style:** Task-focused, concrete examples, minimal theory.

### 3. API Reference
**Goal:** "What parameters does GaussianNIG take?"

**Sphinx auto-generates from your docstrings!** You just write:

**api/models.rst:**
```rst
Models
======

.. autoclass:: fast_bocpd.GaussianNIG
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: fast_bocpd.StudentTNG
   :members:
```

Sphinx reads your Python docstrings and creates beautiful HTML.

### 4. Theory (Explanation)
**Goal:** Help users understand WHY things work.

**Files YOU write:**
- `bayesian_changepoint.rst` - BOCPD algorithm explained
- `conjugate_priors.rst` - Why Normal-Inverse-Gamma? Why not just Normal?
- `model_comparison.rst` - Gaussian vs Student-t: when does robustness matter?
- `when_to_use.rst` - BOCPD vs other methods (CUSUM, Prophet, ruptures)

**Style:** More mathematical, references papers, optional reading.

### 5. Examples (Worked Use Cases)
**Goal:** Show Fast-BOCPD in realistic scenarios.

**Option A:** Convert your Jupyter notebooks:
```rst
Stock Volatility Detection
===========================

.. include:: ../../examples/06_real_world_example.ipynb
```

**Option B:** Write prose with embedded code:
```rst
Detecting Server Anomalies
==========================

Imagine you're monitoring API response times...

.. code-block:: python

   import fast_bocpd as fb
   import pandas as pd
   
   # Load server logs
   df = pd.read_csv('response_times.csv')
   data = df['latency_ms'].values
   
   # Detect when latency changes
   model = fb.GaussianNIG(mu0=100, kappa0=1, alpha0=2, beta0=50)
   hazard = fb.ConstantHazard(lambda_=500)
   detector = fb.BOCPD(model, hazard)
   
   results = detector.batch_update(data)
   changepoints = results.get_changepoints(threshold=0.5)
   
   print(f"Found {len(changepoints)} incidents")
```

### 6. Architecture (For Contributors)
**Goal:** Help developers understand internals.

**Files YOU write:**
- `overview.rst` - High-level: Python wrapper → C core
- `c_backend.rst` - How C code is structured, key functions
- `python_bindings.rst` - How ctypes marshals data
- `adding_models.rst` - Tutorial: implementing a new distribution

**C code documentation goes HERE**, not in user-facing docs!

### 7. Benchmarks
**Goal:** Prove Fast-BOCPD is fast.

**Files YOU write:**
- `methodology.rst` - How benchmarks are run
- `results.rst` - Tables/charts from benchmarks/README.md
- `comparison.rst` - Fast-BOCPD vs competitors

You can auto-generate this from your existing `benchmarks/README.md`!

---

## How Sphinx Combines Everything

**You write** (in `docs/source/`):
```
index.rst                    ← Links to all sections
getting_started/
  quickstart.rst             ← Tutorial you write
user_guide/
  choosing_model.rst         ← Guide you write
api/
  models.rst                 ← Just says "autoclass GaussianNIG"
theory/
  conjugate_priors.rst       ← Math explanation you write
```

**Sphinx generates** (in `docs/build/html/`):
```
index.html                   ← Beautiful homepage
getting_started/
  quickstart.html            ← Rendered tutorial
user_guide/
  choosing_model.html        ← Rendered guide
api/
  models.html                ← Auto-generated from docstrings!
theory/
  conjugate_priors.html      ← Rendered math
search.html                  ← Full-text search
genindex.html                ← Alphabetical index
```

**Magic:** Sphinx reads your Python code, extracts docstrings, and creates `api/*.html` automatically!

---

## Example: docs/source/user_guide/choosing_model.rst

Let me show you what a "guide" file looks like:

```rst
Choosing the Right Model
=========================

Fast-BOCPD supports 7 different observation models. Which should you use?

Quick Decision Tree
-------------------

.. image:: ../_static/model_decision_tree.png
   :alt: Model selection flowchart

1. **What type of data do you have?**
   
   - Continuous (prices, temperatures, etc.) → Go to #2
   - Count data (events per hour, clicks) → :class:`~fast_bocpd.PoissonGamma`
   - Binary (yes/no, success/fail) → :class:`~fast_bocpd.BernoulliBeta`
   - Proportions (conversion rates) → :class:`~fast_bocpd.BinomialBeta`

2. **Does your data have outliers?**
   
   - Yes (or unsure) → :class:`~fast_bocpd.StudentTNG`
   - No → :class:`~fast_bocpd.GaussianNIG`

3. **For count data, is λ > 20?**
   
   - Yes → Can use :class:`~fast_bocpd.GaussianNIG` (faster)
   - No → Use :class:`~fast_bocpd.PoissonGamma`

Detailed Model Comparison
--------------------------

Gaussian (Normal-Inverse-Gamma)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:**

- Continuous data with no outliers
- Fastest model (~25,000 obs/sec)
- Assumes data is bell-shaped

**Example:**

.. code-block:: python

   model = GaussianNIG(mu0=0, kappa0=1, alpha0=1, beta0=1)

**Pros:**

- Fast
- Mathematically elegant
- Well-understood

**Cons:**

- Sensitive to outliers
- Assumes constant variance within segments

**Real-world use cases:**

- Sensor data with stable readings
- Stock prices (preprocessed returns)
- Temperature measurements

Student-t (Normal-Gamma)
~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:**

- Continuous data that may have outliers
- More robust than Gaussian
- Only 10% slower than Gaussian

**Example:**

.. code-block:: python

   # Fixed degrees of freedom
   model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1, nu=3.0)
   
   # Or let the model choose best nu
   model = StudentTNG(mu0=0, kappa0=1, alpha0=1, beta0=1,
                      nu=[2, 3, 5, 10, 20])

**Pros:**

- Robust to outliers
- Can infer tail heaviness from data (grid mode)
- Still fast

**Cons:**

- Grid mode is slower (3,500 obs/sec)
- More parameters to tune

**Choosing nu (degrees of freedom):**

- ``nu=1``: Cauchy distribution (extreme outliers)
- ``nu=3-5``: Financial data (fat tails)
- ``nu=10-20``: Mild outliers
- ``nu→∞``: Approaches Gaussian

**Real-world use cases:**

- Financial returns (fat tails)
- Network latency (occasional spikes)
- User engagement metrics

[... continue for other models ...]

Theory: Why Conjugate Priors?
------------------------------

All Fast-BOCPD models use **conjugate priors**, which means:

1. Prior and posterior have the same form (mathematical elegance)
2. Updates are analytical (no MCMC needed = FAST)
3. Sufficient statistics enable O(1) memory

For deeper math, see :doc:`../theory/conjugate_priors`.

See Also
--------

- :doc:`tuning_parameters` - How to set hyperparameters
- :doc:`../theory/model_comparison` - Statistical details
- :doc:`../api/models` - Full API reference
```

**Key features:**
- Starts practical (decision tree)
- Includes code examples
- Explains trade-offs (pros/cons)
- Links to deeper theory (but doesn't force it)
- Cross-references to API docs

---

## C Code Documentation: Where It Goes

**User-facing docs:** NEVER mention C code directly.

**Contributor docs:** `docs/source/architecture/c_backend.rst`

```rst
C Backend Architecture
======================

*This section is for developers contributing to Fast-BOCPD.*

Overview
--------

The C backend (``fast_bocpd/_c/``) implements the performance-critical
BOCPD algorithm. Python code (``fast_bocpd/*.py``) provides a friendly
wrapper.

Directory Structure
-------------------

.. code-block:: text

   fast_bocpd/_c/
   ├── bocpd_core.c/h          # Main BOCPD algorithm
   ├── hazard.c/h              # Hazard function implementations
   ├── gaussian_nig.c/h        # Gaussian model
   ├── student_t_ng.c/h        # Student-t model
   ├── poisson_gamma.c/h       # Poisson model
   └── ...

Key Data Structures
-------------------

BOCPDState
~~~~~~~~~~

Maintains algorithm state across updates:

.. code-block:: c

   typedef struct {
       int T;                    // Current timestep
       int R;                    // Max run length
       double* log_R;           // Log run-length probabilities [R+1]
       double* log_message;     // Message for recursion [R+1]
       ObsModelState* obs_state; // Model-specific sufficient stats
   } BOCPDState;

**Design rationale:** 

- Fixed-size arrays (no realloc = fast)
- Log-space probabilities (numerical stability)
- Opaque ``ObsModelState`` (polymorphism in C)

Core Algorithm
--------------

The main loop in ``bocpd_core.c::bocpd_update()``:

.. code-block:: c

   // Pseudo-code (see bocpd_core.c for full implementation)
   for (int t = 0; t < n_obs; t++) {
       x_t = data[t];
       
       // 1. Evaluate predictive probability
       for (r = 0; r <= min(t, R); r++) {
           log_pred[r] = obs_model->log_pred(obs_state[r], x_t);
       }
       
       // 2. Update run-length distribution
       log_R_new = hazard_update(log_R, log_pred);
       
       // 3. Update sufficient statistics
       for (r = 0; r <= min(t+1, R); r++) {
           obs_model->update(obs_state[r], x_t);
       }
       
       // 4. Truncate (keep only top R run lengths)
       if (t >= R) prune_state();
   }

**Why C?**

- Tight loops over arrays (10-100x faster than Python)
- No GIL (easy parallelization in future)
- Direct memory control (cache-friendly)

Adding a New Model
------------------

See :doc:`adding_models` for full tutorial.

In brief:

1. Define sufficient statistics struct in ``new_model.h``
2. Implement interface in ``new_model.c``:
   - ``init()`` - Initialize prior parameters
   - ``log_pred()`` - Compute log P(x|data)
   - ``update()`` - Update sufficient statistics
3. Register in ``bocpd_core.c::create_obs_model()``
4. Add Python wrapper in ``fast_bocpd/models.py``
5. Add ctypes binding in ``fast_bocpd/_bindings.py``

Mathematical Details
--------------------

For the math behind each model, see :doc:`../theory/conjugate_priors`.

For benchmarking methodology, see :doc:`../benchmarks/methodology`.
```

**Notice:**
- This is in `architecture/`, not user guide!
- Links to theory docs for math
- Practical (how to add models) not exhaustive
- Users never see this unless they click "Architecture"

---

## Minimal Viable Docs (Week 1)

You don't need to write everything at once! Start with:

**Phase 1: Essential user docs (1 day)**
```
docs/source/
├── index.rst                      # Homepage (10 lines)
├── getting_started/
│   └── quickstart.rst            # Copy from notebook (30 minutes)
├── api/
│   └── index.rst                 # Auto-generated (5 minutes)
└── conf.py                       # Sphinx config (auto-generated)
```

**Phase 2: Add guides as users ask questions (1 week)**
```
user_guide/
├── choosing_model.rst            # When users ask "which model?"
├── tuning_parameters.rst         # When users ask "what is kappa0?"
└── interpreting_results.rst      # When users ask "what's run-length?"
```

**Phase 3: Theory for advanced users (2 weeks)**
```
theory/
├── conjugate_priors.rst          # Move math_foundation.md here
└── when_to_use.rst              # BOCPD vs alternatives
```

**Phase 4: Contributor docs (ongoing)**
```
architecture/
└── c_backend.rst                 # For people adding features
```

---

## Should You Commit docs/build/ to Git?

**NO!** Generated files don't belong in git.

Your `.gitignore` should have:
```
docs/build/
docs/_build/
```

**Instead:** 
- Host docs on **Read the Docs** (free for open source)
- They auto-build HTML from your RST files on every commit
- Your docs live at `fast-bocpd.readthedocs.io`

---

## Summary: Your Docs Strategy

1. **Create `docs/source/` structure** (empty folders)
2. **Write index.rst** (homepage, 10 lines)
3. **Set up Sphinx** (`sphinx-quickstart`)
4. **Add quickstart.rst** (copy from notebook)
5. **Add api/index.rst** (auto-generate from docstrings)
6. **Build locally** (`make html`) and preview
7. **Add guides as needed** (iteratively, based on user questions)
8. **Move math_foundation.md** → `theory/conjugate_priors.rst`
9. **Add architecture/c_backend.rst** (for contributors, not users)
10. **Deploy to Read the Docs** (free hosting)

**Time investment:**
- Day 1: Index + Quickstart + API = 2 hours
- Week 1: Add 3-4 user guides = 4 hours
- Week 2: Theory + Architecture = 3 hours
- **Total: ~10 hours for professional docs**

---

Want me to:
1. **Create the docs/source/ folder structure right now?**
2. **Write a template index.rst for you to fill in?**
3. **Show you how to convert your math_foundation.md to RST?**
4. **Set up the Sphinx config?**

Let me know what would be most helpful!
