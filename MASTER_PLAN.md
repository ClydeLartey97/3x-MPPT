# MASTER PLAN: 3x Low-Light MPPT Simulation Engine

> ANY AGENT CONTINUING THIS WORK MUST READ THIS ENTIRE DOCUMENT BEFORE DOING ANYTHING AND MUST FOLLOW THE PLAN EXACTLY, UPDATING STATUS AFTER EACH STEP

**OVERALL PROGRESS: 22 / 24 steps completed**

This document is the single source of truth for the build. Every step below carries a status
indicator: `[NOT STARTED]`, `[IN PROGRESS]`, `[COMPLETED]`, or `[FAILED - reason]`. The status is
updated immediately after each step is finished. Work is committed and pushed to the `main`
branch after every single step, one step per commit. All authorship is attributed to Clyde
Lartey. British English is used throughout; em dashes are never used.

---

## Conventions

- British English in all comments, docstrings, documentation, and output strings.
- No em dashes anywhere; use commas, semicolons, colons, or full stops.
- One step, one commit, one push, all onto `main`.
- The virtual environment lives in `venv/` and is git ignored.

---

## Steps

### Step 1: Create project directory structure [COMPLETED]
- Create `src/`, `src/algorithms/`, `tests/`, `outputs/validation/profiles/`, `outputs/plots/`,
  `outputs/data/`, `docs/`.
- Add empty `__init__.py` files to `src/`, `src/algorithms/`, `tests/`.

### Step 2: Create requirements.txt and install dependencies [COMPLETED]
- Pin numpy, scipy, matplotlib, pandas, streamlit, pytest to the minimum versions given.
- Install into `venv/`.

### Step 3: Create .gitignore [COMPLETED]
- Standard Python entries plus `outputs/` and `venv/`.

### Step 4: Implement cell_model.py with all functions [COMPLETED]
- Single-diode model constants and `cell_current`, `cell_power`, `find_true_mpp`,
  `generate_iv_curve`, plus an open-circuit voltage helper.
- Solve the implicit current equation numerically with `scipy.optimize.brentq`.

### Step 5: Validate cell model, generate I-V curves, save plots [COMPLETED]
- Produce I-V and P-V curves at 1000, 500, 200, 100, 50, 20 W/m2.
- Save to `outputs/validation/`; confirm peak flattening at low irradiance.

### Step 6: Implement irradiance_profiles.py with all profile generators [COMPLETED]
- Office, warehouse, retail, constant low light, and stress test profiles with noise, step
  changes, and occlusion events.

### Step 7: Validate irradiance profiles, save example plots [COMPLETED]
- Save example plots of each profile to `outputs/validation/profiles/`.

### Step 8: Implement algorithms/base.py [COMPLETED]
- Abstract base class defining the common MPPT interface and logging attributes.

### Step 9: Implement algorithms/perturb_observe.py [COMPLETED]
- Standard fixed step P&O algorithm.

### Step 10: Implement algorithms/ags_mppt.py [COMPLETED]
- Adaptive Gradient-Scaled MPPT with gradient scaling, irradiance shift detection, and
  oscillation damping.

### Step 11: Implement simulation.py [COMPLETED]
- SimulationEngine plus SimulationResults dataclass with band, convergence, oscillation
  analysis, CSV export, and text summary.

### Step 12: Implement visualisation.py [COMPLETED]
- All plotting functions at 300 DPI, saved as PNG and SVG, using the defined colour scheme.

### Step 13: Implement main.py [COMPLETED]
- CLI with argparse and all documented options, each with a help string.

### Step 14: Run first full simulation and verify outputs [COMPLETED]
- Office profile, 24h; verify plots and summary generate correctly.

### Step 15: Tune AGS-MPPT parameters if needed [COMPLETED]
- Ensure AGS demonstrably outperforms P&O in low light.
- Tuned defaults: base_step 0.008, max_multiplier 3.0, max_step 0.02, gradient thresholds
  0.10 and 0.01 on an irradiance-normalised gradient, low-light park at 0.26 V below 8 W/m2.
- Validated improvement across profiles: office +2.6%, warehouse +3.2%, retail +1.0%,
  constant 30 W/m2 +0.1%, stress +0.8%. Bright constant light stays within tolerance of P&O.

### Step 16: Write all unit tests in tests/ [COMPLETED]
- Cell model, algorithms, irradiance, and simulation test suites.

### Step 17: Run all tests and fix failures [COMPLETED]

### Step 18: Implement dashboard.py [COMPLETED]
- Streamlit dashboard with all four tabs and dark theme.

### Step 19: Test the Streamlit dashboard runs correctly [COMPLETED]

### Step 20: Write README.md [COMPLETED]

### Step 21: Write docs/ALGORITHM.md [COMPLETED]

### Step 22: Create LICENSE (MIT) [COMPLETED]

### Step 23: Confirm git repository and commit history [NOT STARTED]
- Repository already initialised on `main`; confirm state and history.

### Step 24: Final review [NOT STARTED]
- Run full simulation, verify all plots generate, verify all tests pass.
