# Seminar QF Project

## Project Structure

This project has been refactored to be cleaner and more modular.

```
Seminar QF/
├── notebooks/         # Jupyter Notebooks for interactive analysis
│   └── main.ipynb     # Main analysis pipeline (replaces main.py)
├── src/               # Python source code modules
│   ├── analysis/      # Monte Carlo, CDS Calculation
│   ├── data/          # Data Loading & Processing
│   ├── models/        # GARCH, Regime Switching, MS-GARCH Logic
│   └── utils/         # Config and Helper functions
├── data/              # Data storage
│   ├── input/         # Raw input files (Excel, CSV)
│   ├── output/        # Generated results
│   ├── intermediates/ # Cache files
│   └── diagnostics/   # Volatility diagnostics
```

## How to Run

1.  **Open the project** in VS Code.
2.  **Navigate to `notebooks/main.ipynb`**.
3.  **Run the cells** to execute the analysis steps.
    *   The notebook automatically sets up paths to import modules from `src/` and load data from `data/`.
    *   Results are saved to `data/output/`.

## Configuration

File paths are managed in `src/utils/config.py`. If you change folder locations, update this file.

## GARCH Diagnostics

Run these from the project root to diagnose parameter behavior and simulated spread dynamics:

1. Parameter diagnostics:

```bash
python -m src.analysis.garch_parameter_diagnostics
```

2. Simulated spread diagnostics:

```bash
python -m src.analysis.garch_spread_diagnostics
```

Outputs are written to `data/diagnostics/` as CSV summaries and PNG charts.
