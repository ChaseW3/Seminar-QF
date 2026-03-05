"""
Affine Calibration of Model-Implied CDS Spreads (In-Sample & Out-of-Sample).

Implements the rolling-window OLS calibration described in:
    S^CDS_{i,t-k}(τ) = β₀^{(m,τ)}(t) + β₁^{(m,τ)}(t) · s^{(m)}_{i,t-k}(τ) + u_{i,t-k}

for k = 1, ..., W  (W = estimation window in weeks, default 52).

The calibrated (out-of-sample) spread at date t is:
    ŝ^{(m)}_{i,t}(τ) = β̂₀^{(m,τ)}(t) + β̂₁^{(m,τ)}(t) · s^{(m)}_{i,t}(τ)

Because calibration parameters are estimated using only information up to t − 1,
the calibrated spread constitutes a strictly out-of-sample prediction.

References
----------
- Malone et al. (2009), methodology for CDS spread comparison
- Byström (2006), correlation of innovations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Default rolling window length (52 weeks ≈ 1 year of daily data).
# We use 252 trading days, which is the standard convention for one calendar year.
DEFAULT_WINDOW_WEEKS = 52
DEFAULT_WINDOW_DAYS = 252

# Models and their spread column prefixes in the MC output files
MODEL_CONFIGS = {
    'GARCH': {
        'mc_file': 'daily_monte_carlo_garch_results.csv',
        'spread_prefix': 'mc_garch_implied_spread',
    },
    'Regime-Switching': {
        'mc_file': 'daily_monte_carlo_regime_switching_results.csv',
        'spread_prefix': 'rs_implied_spread',
    },
    'MS-GARCH': {
        'mc_file': 'daily_monte_carlo_ms_garch_results.csv',
        'spread_prefix': 'mc_ms_garch_implied_spread',
    },
}

MATURITIES = [1, 3, 5]


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: load & merge market CDS data  (self-contained, no dependency on
#  existing src.analysis code)
# ──────────────────────────────────────────────────────────────────────────────

def load_market_cds(input_dir: Path) -> pd.DataFrame:
    """
    Load 1Y, 3Y, 5Y market CDS spread Excel files and return a single
    long-format DataFrame with columns:
        date, company_cds, cds_market_1y_bps, cds_market_3y_bps, cds_market_5y_bps
    """
    input_dir = Path(input_dir)
    frames = {}
    for mat in MATURITIES:
        fp = input_dir / f'CDS_{mat}y_mat_data.xlsx'
        df = pd.read_excel(fp, header=None)

        # Row 3 = company names, data starts row 5
        company_names_raw = df.iloc[3, 2:].tolist()
        company_names = []
        for name in company_names_raw:
            if pd.notna(name):
                clean = name.split(' SNR ')[0] if ' SNR ' in str(name) else str(name)
                company_names.append(clean)
            else:
                company_names.append(None)

        dates = pd.to_datetime(df.iloc[5:, 0], errors='coerce')
        data = df.iloc[5:, 2:].copy()
        data.columns = company_names[:len(data.columns)]
        data['date'] = dates.values

        data_long = data.melt(
            id_vars=['date'],
            var_name='company_cds',
            value_name=f'cds_market_{mat}y_bps',
        )
        data_long[f'cds_market_{mat}y_bps'] = pd.to_numeric(
            data_long[f'cds_market_{mat}y_bps'], errors='coerce'
        )
        data_long = data_long.dropna(subset=['date', 'company_cds'])
        frames[mat] = data_long

    # Merge all maturities
    cds = frames[1]
    for mat in [3, 5]:
        cds = cds.merge(frames[mat], on=['date', 'company_cds'], how='outer')
    return cds


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: map gvkey → CDS company name
# ──────────────────────────────────────────────────────────────────────────────

# Same mapping as in cds_correlation.py – duplicated here on purpose so this
# module is fully self-contained.
# Keys   = company names as they appear in merged_data_with_merton.csv
# Values = company names as they appear in the CDS market Excel files
COMPANY_MAPPING = {
    'ADIDAS AG': 'ADIDAS AG',
    'AIRBUS SE': 'AIRBUS SE',
    'ALLIANZ SE': 'ALLIANZ SE',
    'ANHEUSER-BUSCH INBEV': 'ANHEUSER-BUSCH',
    'AXA SA': 'AXA',
    'BASF SE': 'BASF SE',
    'BAYER AG': 'BAYER AG',
    'BAYERISCHE MOTOREN WERKE AKT': 'BAYER MOTOREN WERKE',
    'BNP PARIBAS': 'BNP PARIBAS SA',
    'DANONE SA': 'DANONE SA',
    'DEUTSCHE POST AG': 'DEUTSCHE POST AG',
    'DEUTSCHE TELEKOM': 'DEUTSCHE TELEKOM AG',
    'ENEL SPA': 'ENEL S.P.A.',
    'ENI SPA': 'ENI S.P.A.',
    'IBERDROLA SA': 'IBERDROLA, S.A.',
    'INFINEON TECHNOLOGIES AG': 'INFINEON TECS',
    'ING GROEP NV': 'ING GROEP N.V.',
    'INTESA SANPAOLO SPA': 'INTESA SANPAOLO',
    'KERING SA': 'KERING SA',
    'KONINKLIJKE AHOLD DELHAIZE': 'KON AHOLD DELHAIZE',
    "L'AIR LIQUIDE SA": 'AIR LIQUIDE SA',
    'LOREAL SA': "L'OREAL",
    'LVMH MOET HENNESSY LOUIS V': 'LVMH MOET HENNESSY',
    'MUNICH RE CO': 'MUNICH REINSURANCE',
    'NOKIA OYJ': 'NOKIA OYJ',
    'ORANGE SA': 'ORANGE S.A.',
    'SANOFI': 'SANOFI SA',
    'SAP SE': 'SAP SE',
    'SCHNEIDER ELECTRIC S E': 'SCHNEIDER ELECTRIC',
    'SIEMENS AG': 'SIEMENS AG',
    'TOTALENERGIES SE': 'TOTALENERGIES SE',
    'UNICREDIT SPA': 'UNICREDIT SPA',
    'VINCI SA': 'VINCI',
    'WOLTERS KLUWER NV': 'WOLTERS KLUWER NV',
}


def _build_gvkey_to_cds_name(merton_file: Path) -> Dict[int, str]:
    """Return a dict  gvkey -> CDS company name  using the Merton file and the mapping."""
    mdf = pd.read_csv(merton_file)
    gvkey_company = mdf[['gvkey', 'company']].drop_duplicates().set_index('gvkey')['company'].to_dict()
    return {gvkey: COMPANY_MAPPING.get(comp, comp) for gvkey, comp in gvkey_company.items()}


# ──────────────────────────────────────────────────────────────────────────────
#  Core: rolling-window OLS calibration
# ──────────────────────────────────────────────────────────────────────────────

def _ols_params(x: np.ndarray, y: np.ndarray,
                max_beta1: float = 1e6) -> Tuple[float, float]:
    """
    Simple OLS for y = β₀ + β₁·x.
    Returns (β₀, β₁).  Falls back to (NaN, NaN) if singular or if the
    estimated β₁ is unreasonably large (indicating near-zero variance in x).

    Parameters
    ----------
    x, y   : arrays of equal length
    max_beta1 : float – safety cap for |β₁|; if exceeded, return (NaN, NaN)
    """
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    # If x has near-zero variance the regression is meaningless
    if np.std(x) < 1e-10:
        return np.nan, np.nan
    X = np.column_stack([np.ones(n), x])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        # Reject absurd estimates that arise from near-degenerate data
        if np.abs(beta[1]) > max_beta1:
            return np.nan, np.nan
        return beta[0], beta[1]
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def rolling_calibration(
    model_spreads: pd.Series,
    market_spreads: pd.Series,
    dates: pd.Series,
    window: int = DEFAULT_WINDOW_DAYS,
) -> pd.DataFrame:
    """
    Perform rolling-window affine calibration for a single firm × maturity.

    At each evaluation date *t* the regression uses the past *window* observations
    (indices t-window … t-1).  The calibrated spread at *t* is then:

        ŝ_t = β̂₀(t) + β̂₁(t) · s_t^{model}

    Parameters
    ----------
    model_spreads : pd.Series  –  raw model-implied spreads (same unit as market)
    market_spreads : pd.Series –  observed CDS spreads (bps)
    dates : pd.Series          –  corresponding dates
    window : int               –  number of past observations in rolling window

    Returns
    -------
    pd.DataFrame with columns:
        date, model_raw, market, beta0, beta1, calibrated, residual
    """
    n = len(model_spreads)
    result = {
        'date': dates.values,
        'model_raw': model_spreads.values,
        'market': market_spreads.values,
        'beta0': np.full(n, np.nan),
        'beta1': np.full(n, np.nan),
        'calibrated': np.full(n, np.nan),
    }

    x_all = model_spreads.values
    y_all = market_spreads.values

    for t in range(window, n):
        # In-sample estimation window: [t-window, t-1]
        x_win = x_all[t - window:t]
        y_win = y_all[t - window:t]

        # Drop NaN pairs within window
        mask = np.isfinite(x_win) & np.isfinite(y_win)
        if mask.sum() < 10:
            continue

        # Require at least 20% of the window to have non-zero model spreads.
        # Otherwise the regression is dominated by (0, market) pairs and β₁ explodes.
        x_valid = x_win[mask]
        y_valid = y_win[mask]
        nonzero_frac = np.mean(np.abs(x_valid) > 1e-6)
        if nonzero_frac < 0.20:
            continue

        b0, b1 = _ols_params(x_valid, y_valid)
        result['beta0'][t] = b0
        result['beta1'][t] = b1

        # Out-of-sample calibrated spread
        if np.isfinite(x_all[t]) and np.isfinite(b0) and np.isfinite(b1):
            result['calibrated'][t] = b0 + b1 * x_all[t]

    df = pd.DataFrame(result)
    df['residual'] = df['market'] - df['calibrated']
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  High-level driver: calibrate all firms × maturities for one model
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_model(
    model_name: str,
    output_dir: Path,
    input_dir: Path,
    merton_file: Path,
    window: int = DEFAULT_WINDOW_DAYS,
    convert_model_to_bps: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Run rolling affine calibration for every (firm, maturity) pair.

    Parameters
    ----------
    model_name : str
        One of 'GARCH', 'Regime-Switching', 'MS-GARCH'.
    output_dir : Path
        Directory containing the daily_monte_carlo_*_results.csv files.
    input_dir : Path
        Directory containing the CDS market Excel files.
    merton_file : Path
        Path to merged_data_with_merton.csv (for gvkey → company mapping).
    window : int
        Rolling window in trading days (default 260 ≈ 52 weeks).
    convert_model_to_bps : bool
        If True, multiply model spreads by 10 000 to convert from decimal to bps.

    Returns
    -------
    dict  –  keyed by maturity string ('1y', '3y', '5y'), each value is a
             DataFrame of all firms stacked, with columns:
             gvkey, company, date, model_raw, market, beta0, beta1, calibrated, residual
    """
    cfg = MODEL_CONFIGS[model_name]
    mc_file = Path(output_dir) / cfg['mc_file']
    spread_prefix = cfg['spread_prefix']

    # Load model output
    print(f"Loading model data from {mc_file.name} ...")
    mc_df = pd.read_csv(mc_file)
    mc_df['date'] = pd.to_datetime(mc_df['date'])

    # Convert model spreads to bps if needed
    if convert_model_to_bps:
        for mat in MATURITIES:
            col = f'{spread_prefix}_{mat}y'
            mc_df[col] = mc_df[col] * 10_000

    # Load market CDS
    print("Loading market CDS data ...")
    market_cds = load_market_cds(input_dir)

    # Build gvkey → CDS company name mapping
    gvkey_map = _build_gvkey_to_cds_name(merton_file)
    mc_df['company'] = mc_df['gvkey'].map(gvkey_map)

    # Merge model + market
    merged_parts = []
    for mat in MATURITIES:
        model_col = f'{spread_prefix}_{mat}y'
        market_col = f'cds_market_{mat}y_bps'

        sub = mc_df[['gvkey', 'company', 'date', model_col]].copy()
        sub = sub.rename(columns={model_col: 'model_spread'})
        sub['maturity'] = mat

        # Merge with market on (date, company)
        market_sub = market_cds[['date', 'company_cds', market_col]].copy()
        market_sub = market_sub.rename(columns={market_col: 'market_spread'})

        sub = sub.merge(
            market_sub,
            left_on=['date', 'company'],
            right_on=['date', 'company_cds'],
            how='inner',
        )
        merged_parts.append(sub)

    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.sort_values(['gvkey', 'maturity', 'date'])

    print(f"Matched {len(merged):,} firm-date-maturity observations across "
          f"{merged['gvkey'].nunique()} firms.")

    # Run calibration per (firm, maturity)
    results: Dict[str, pd.DataFrame] = {}
    for mat in MATURITIES:
        mat_key = f'{mat}y'
        mat_data = merged[merged['maturity'] == mat].copy()
        frames = []

        firms = mat_data['gvkey'].unique()
        for gvkey in firms:
            firm_data = mat_data[mat_data['gvkey'] == gvkey].sort_values('date').reset_index(drop=True)
            if len(firm_data) < window + 10:
                continue

            cal = rolling_calibration(
                model_spreads=firm_data['model_spread'],
                market_spreads=firm_data['market_spread'],
                dates=firm_data['date'],
                window=window,
            )
            cal['gvkey'] = gvkey
            cal['company'] = firm_data['company'].iloc[0]
            frames.append(cal)

        if frames:
            results[mat_key] = pd.concat(frames, ignore_index=True)
            n_cal = results[mat_key]['calibrated'].notna().sum()
            print(f"  {mat_key}: {n_cal:,} calibrated out-of-sample observations "
                  f"across {len(frames)} firms")
        else:
            print(f"  {mat_key}: insufficient data for calibration")

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Evaluation metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(cal_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute evaluation metrics for a calibration result DataFrame.

    Returns dict with keys:
        rmse_raw, rmse_calibrated, mae_raw, mae_calibrated,
        corr_levels_raw, corr_levels_cal,
        corr_changes_raw, corr_changes_cal,
        mean_beta0, mean_beta1
    """
    valid_raw = cal_df.dropna(subset=['model_raw', 'market'])
    valid_cal = cal_df.dropna(subset=['calibrated', 'market'])

    metrics = {}

    # RMSE
    if len(valid_raw) > 0:
        err_raw = valid_raw['market'] - valid_raw['model_raw']
        metrics['rmse_raw'] = np.sqrt((err_raw ** 2).mean())
        metrics['mae_raw'] = np.abs(err_raw).mean()
        metrics['corr_levels_raw'] = valid_raw['model_raw'].corr(valid_raw['market'])
    else:
        metrics['rmse_raw'] = np.nan
        metrics['mae_raw'] = np.nan
        metrics['corr_levels_raw'] = np.nan

    if len(valid_cal) > 0:
        err_cal = valid_cal['market'] - valid_cal['calibrated']
        metrics['rmse_calibrated'] = np.sqrt((err_cal ** 2).mean())
        metrics['mae_calibrated'] = np.abs(err_cal).mean()
        metrics['corr_levels_cal'] = valid_cal['calibrated'].corr(valid_cal['market'])
    else:
        metrics['rmse_calibrated'] = np.nan
        metrics['mae_calibrated'] = np.nan
        metrics['corr_levels_cal'] = np.nan

    # Correlation of CHANGES (innovations)
    for prefix, col in [('raw', 'model_raw'), ('cal', 'calibrated')]:
        sub = cal_df[['gvkey', 'date', col, 'market']].dropna().sort_values(['gvkey', 'date'])
        if len(sub) > 20:
            sub[f'd_{col}'] = sub.groupby('gvkey')[col].diff()
            sub['d_market'] = sub.groupby('gvkey')['market'].diff()
            sub_clean = sub.dropna(subset=[f'd_{col}', 'd_market'])
            if len(sub_clean) > 10:
                metrics[f'corr_changes_{prefix}'] = sub_clean[f'd_{col}'].corr(sub_clean['d_market'])
            else:
                metrics[f'corr_changes_{prefix}'] = np.nan
        else:
            metrics[f'corr_changes_{prefix}'] = np.nan

    # Calibration parameter summary
    valid_params = cal_df.dropna(subset=['beta0', 'beta1'])
    if len(valid_params) > 0:
        metrics['mean_beta0'] = valid_params['beta0'].mean()
        metrics['mean_beta1'] = valid_params['beta1'].mean()
        metrics['std_beta0'] = valid_params['beta0'].std()
        metrics['std_beta1'] = valid_params['beta1'].std()
    else:
        metrics['mean_beta0'] = np.nan
        metrics['mean_beta1'] = np.nan
        metrics['std_beta0'] = np.nan
        metrics['std_beta1'] = np.nan

    return metrics


def compute_firm_level_metrics(cal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-firm evaluation metrics.

    Returns DataFrame with one row per firm and columns for RMSE, correlation, etc.
    """
    rows = []
    for gvkey, grp in cal_df.groupby('gvkey'):
        m = compute_metrics(grp)
        m['gvkey'] = gvkey
        m['company'] = grp['company'].iloc[0]
        m['n_obs'] = grp['calibrated'].notna().sum()
        rows.append(m)
    return pd.DataFrame(rows)
