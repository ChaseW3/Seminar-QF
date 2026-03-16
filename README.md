# Extending the Merton Model with Regime-Switching GARCH Volatility

This repository contains the code and outputs for the seminar paper:
**"Extending the Merton Model with Regime-Switching GARCH Volatility for Default Risk Estimation"**

## What the paper does

We evaluate whether relaxing key Merton assumptions improves structural default risk estimation. Using inferred firm asset returns, we compare four model specifications against observed CDS spreads (1Y, 3Y, 5Y):

- Merton (constant volatility)
- GARCH-Merton (time-varying volatility)
- Regime-Switching Merton (discrete market states)
- MS-GARCH-Merton (regime-switching + time-varying volatility)

Main takeaway: time-varying volatility materially improves fit to CDS spreads; MS-GARCH helps especially for short maturities and high-leverage firms, but does not uniformly dominate simpler extensions.

## Data and sample

- Daily firm-level market/accounting inputs and CDS data
- Coverage: European firms, 2017-2024 evaluation focus (with longer historical estimation windows)
- CDS maturities: 1, 3, 5 years
- Liquidity/availability filters are applied before CDS-based evaluation

## Repository layout

Project code lives in the nested folder:

- `Seminar QF/src/` - models, simulation, analysis, calibration utilities
- `Seminar QF/notebooks/` - end-to-end analysis notebooks
- `Seminar QF/data/input/` - raw inputs
- `Seminar QF/data/output/` - model outputs and diagnostics
- `Seminar QF/data/output/calibration/` - calibrated spreads and calibration metrics

## Quick start

1. Move to project directory:
   - `cd "Seminar QF"`
2. Create/activate environment and install dependencies:
   - `pip install -r requirements.txt`
3. Run notebooks (recommended order):
   - `notebooks/main.ipynb`
   - `notebooks/calibration_analysis.ipynb`
   - `notebooks/cds_correlation_analysis.ipynb`
   - `notebooks/visualize_results.ipynb`

## Key outputs

Typical generated files include:

- Raw Monte Carlo outputs:
  - `data/output/daily_monte_carlo_merton_results.csv`
  - `data/output/daily_monte_carlo_garch_results.csv`
  - `data/output/daily_monte_carlo_regime_switching_results.csv`
  - `data/output/daily_monte_carlo_ms_garch_results.csv`
- Calibrated spreads and diagnostics:
  - `data/output/calibration/calibrated_spreads_<model>_<maturity>.csv`
  - `data/output/calibration/calibration_metrics_summary.csv`

## Authors

- Thomas de Leeuw
- Chenelle Huijbers
- Lars Gilsing
- Chase Weng

Erasmus University Rotterdam, 2026.
