# Seminar QF — Codebase Restructuring Plan

A step-by-step refactor checklist. Each task is self-contained and can be completed independently. Work through them in order, since later tasks build on earlier ones.

---

## Task 1 — Move `cds_date_filter.py` and `cds_correlation.py` into `src/data/`

These files are data loaders, not analysis logic. They belong alongside `data_processing.py`.

**Files to move:**
- `src/analysis/cds_date_filter.py` → `src/data/cds_date_filter.py`
- `src/analysis/cds_correlation.py` → `src/data/cds_correlation.py`

**Imports that must be updated after the move:**

| File | Old import | New import |
|---|---|---|
| `src/analysis/monte_carlo_merton.py` | `from src.analysis.cds_date_filter import ...` | `from src.data.cds_date_filter import ...` |
| `src/analysis/monte_carlo_garch.py` | `from src.analysis.cds_date_filter import ...` | `from src.data.cds_date_filter import ...` |
| `src/analysis/monte_carlo_regime_switching.py` | `from src.analysis.cds_date_filter import ...` | `from src.data.cds_date_filter import ...` |
| `src/analysis/monte_carlo_ms_garch.py` | `from src.analysis.cds_date_filter import ...` | `from src.data.cds_date_filter import ...` |
| `src/analysis/model_performance_paper.py` | `from src.analysis.cds_correlation import ...` | `from src.data.cds_correlation import ...` |
| `notebooks/main.ipynb` | `from src.analysis import cds_correlation` / `from src.analysis.cds_correlation import ...` | `from src.data import cds_correlation` / `from src.data.cds_correlation import ...` |

**Guardrail:** After moving, run `python -m src.analysis.model_performance_paper` and confirm it imports cleanly before touching anything else.

---

## Task 2 — Move Monte Carlo runners into a new `src/simulation/` subpackage

These are simulation engines, not analysis logic. Separating them makes the pipeline easier to follow at a glance.

**Files to move:**
- `src/analysis/monte_carlo_merton.py` → `src/simulation/monte_carlo_merton.py`
- `src/analysis/monte_carlo_garch.py` → `src/simulation/monte_carlo_garch.py`
- `src/analysis/monte_carlo_regime_switching.py` → `src/simulation/monte_carlo_regime_switching.py`
- `src/analysis/monte_carlo_ms_garch.py` → `src/simulation/monte_carlo_ms_garch.py`

**New file to create:** `src/simulation/__init__.py` (empty)

**Imports that must be updated after the move:**

| File | Old import | New import |
|---|---|---|
| `notebooks/main.ipynb` | `from src.analysis.monte_carlo_garch import ...` | `from src.simulation.monte_carlo_garch import ...` |
| `notebooks/main.ipynb` | `from src.analysis.monte_carlo_regime_switching import ...` | `from src.simulation.monte_carlo_regime_switching import ...` |
| `notebooks/main.ipynb` | `from src.analysis.monte_carlo_ms_garch import ...` | `from src.simulation.monte_carlo_ms_garch import ...` |
| `notebooks/main.ipynb` | `from src.analysis.monte_carlo_merton import ...` | `from src.simulation.monte_carlo_merton import ...` |
| `notebooks/main.ipynb` | `import src.analysis.monte_carlo_garch as ...` | `import src.simulation.monte_carlo_garch as ...` |
| `notebooks/main.ipynb` | `from src.analysis import monte_carlo_regime_switching` | `from src.simulation import monte_carlo_regime_switching` |
| `notebooks/main.ipynb` | `from src.analysis import monte_carlo_ms_garch` | `from src.simulation import monte_carlo_ms_garch` |

**Guardrail:** After Task 1, the `cds_date_filter` imports inside the monte carlo files will already point to `src.data`. Do not redo those. Run `main.ipynb` up through the MC cells to verify.

---

## Task 3 — Consolidate plotting utilities into `src/utils/plots/`

Plotting code is currently split between two locations. Unify them.

**Files to move:**
- `src/utils/cds_plotter.py` → `src/utils/plots/cds_plotter.py`
- `src/calibration/calibration_plots.py` → `src/utils/plots/calibration_plots.py`

**New files to create:**
- `src/utils/plots/__init__.py` (empty)

**Imports that must be updated after the move:**

| File | Old import | New import |
|---|---|---|
| `notebooks/visualize_results.ipynb` | `from src.utils.cds_plotter import CDSPlotter` | `from src.utils.plots.cds_plotter import CDSPlotter` |
| `notebooks/calibration_analysis.ipynb` | `from src.calibration.calibration_plots import ...` | `from src.utils.plots.calibration_plots import ...` |

**Guardrail:** Check whether `calibration_plots.py` imports anything from `src/calibration/` (e.g. `affine_calibration`). If it does, those imports are unaffected — only the location of `calibration_plots.py` itself changes.

---

## Task 4 — Reorganise `data/output/` subdirectories

Right now CSV tables, PNG figures, and calibration files all sit flat in `data/output/`. Split them clearly.

**Target layout:**
```
data/output/
  tables/          ← all .csv outputs (model results, parameters, DM tests, etc.)
  figures/         ← all .png outputs (scatter plots, time series, etc.)
  calibration/     ← calibrated_spreads_*.csv, calibration_metrics_summary.csv, firm_metrics_*.csv
                      (already a subfolder — keep it, just ensure nothing writes to output/ root for calibration)
```

**Files that write to `data/output/` and must be updated:**

| File | What it writes | Target subdir |
|---|---|---|
| `src/models/garch_model.py` | `garch_parameters.csv` | `tables/` |
| `src/models/regime_switching.py` | `regime_switching_parameters.csv` | `tables/` |
| `src/models/ms_garch_optimized.py` | `ms_garch_parameters.csv` | `tables/` |
| `src/analysis/cds_correlation.py` | `cds_model_vs_market_correlations.csv`, scatter `.png` | `tables/`, `figures/` |
| `src/analysis/model_performance_paper.py` | DM test CSVs, figure PNGs, comparison CSVs | `tables/`, `figures/` |
| `src/calibration/affine_calibration.py` | `calibrated_spreads_*.csv`, `calibration_metrics_summary.csv` | `calibration/` (already) |

**Guardrail:** Update `src/utils/config.py` to expose the new path constants (`TABLES_DIR`, `FIGURES_DIR`) and have every writer use those constants — never hard-code the paths in individual modules. Add `.mkdir(parents=True, exist_ok=True)` calls beside the new constants so subdirectories are auto-created. Any file that *reads* outputs as inputs (e.g. `model_performance_paper.py` reading calibration CSVs) must also be updated to point to the new paths.

---

## Task 5 — Add `pyproject.toml`

Currently the project only runs correctly when the CWD is `Seminar QF/`. A minimal `pyproject.toml` makes `src` importable regardless of CWD.

**File to create:** `Seminar QF/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "seminar-qf"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["src*"]
```

**Then install in editable mode:**
```bash
pip install -e .
```

**Guardrail:** After installing, confirm that `python -c "from src.utils import config"` works from any directory. The `.venv` must be active when running `pip install -e .`.

---

## Task 6 — Add a paper-export script

Figures are generated into `data/output/figures/` (after Task 4) but `paper_appendix/figures/` is always empty. Make the copy explicit and repeatable.

**File to create:** `src/utils/export_paper_figures.py`

This script should:
1. Read a manifest (a hardcoded list, or a glob pattern) of the specific figure files needed for the paper
2. Copy them from `data/output/figures/` into `paper_appendix/figures/`
3. Print a summary of what was copied / what was missing

**Guardrail:** The script must be non-destructive — use `shutil.copy2` (not move). Never delete the source files.

---


## Final structure after all tasks are complete

```
src/
  data/
    __init__.py
    data_processing.py
    cds_date_filter.py      ← moved from analysis/
    cds_correlation.py      ← moved from analysis/
  models/
    __init__.py
    garch_model.py
    ms_garch_optimized.py
    probability_of_default.py
    regime_switching.py
  simulation/               ← new subpackage
    __init__.py
    monte_carlo_merton.py
    monte_carlo_garch.py
    monte_carlo_regime_switching.py
    monte_carlo_ms_garch.py
  calibration/
    __init__.py
    affine_calibration.py
  analysis/
    __init__.py
    model_performance_paper.py
    regime_analysis.py
  utils/
    __init__.py
    config.py
    summary_statistics.py
    plots/                  ← new subpackage
      __init__.py
      cds_plotter.py
      calibration_plots.py

pyproject.toml
requirements.txt
README.md
```
