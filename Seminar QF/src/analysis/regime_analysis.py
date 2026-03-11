"""
MS-GARCH Parameter Analysis & GARCH Dynamics Visualization.

Provides:
  - extract_regime_metrics(ms_garch_df) → DataFrame of derived per-regime metrics
  - plot_parameter_comparison(metrics_df)
  - plot_persistence_and_volatility(metrics_df)
  - plot_news_impact_curves(metrics_df)
  - plot_volatility_impulse_response(metrics_df)
  - plot_regime_probabilities_ts(ms_garch_df)  [only if smoothed probs exist]
  - run_ms_garch_analysis(ms_garch_df)  ← convenience wrapper

All functions are designed to work with the ms_garch DataFrame as loaded from
``data/output/ms_garch_parameters.csv`` (columns: omega_0/1, alpha_0/1, beta_0/1,
nu_0/1, mu_0/1, p00, p11, …).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# ---------------------------------------------------------------------------
# Colours consistent with the rest of regime_analysis.ipynb
# ---------------------------------------------------------------------------
_R0_COLOR = '#3498db'
_R1_COLOR = '#e74c3c'
_REGIME_LABELS = {0: 'Regime 0 (Low Vol)', 1: 'Regime 1 (High Vol)'}
_REGIME_COLORS = {0: _R0_COLOR, 1: _R1_COLOR}


# ===================================================================
# A) Parameter extraction + derived metrics
# ===================================================================

def _trimmed_mean(s, pct=0.05):
    """Mean after dropping the top and bottom *pct* fraction (NaN-safe)."""
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    trimmed = s[(s >= lo) & (s <= hi)]
    return trimmed.mean() if len(trimmed) > 0 else s.median()


def extract_regime_metrics(ms_garch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional *average* per-regime GARCH metrics from the
    MS-GARCH parameter DataFrame.

    Returns a tidy DataFrame with rows = regimes, columns = metrics.

    Metrics computed:
        omega, alpha, beta, nu, mu,
        persistence  (= alpha + beta),
        stationary   (bool, persistence < 1),
        uncond_var   (omega / (1 - persistence) if stationary, else NaN),
        uncond_vol   (sqrt(uncond_var))  — **robust** trimmed mean of per-row values,
        half_life    (ln(0.5) / ln(persistence) if 0 < persistence < 1),
        p_stay       (self-transition probability),
        p_switch     (1 - p_stay),
        steady_state (long-run probability of being in this regime),
        expected_duration (1 / (1 - p_stay)).

    Near-IGARCH rows (persistence >= 0.999) are excluded from unconditional
    volatility computation to avoid denominator-explosion artefacts.
    """
    required = ['omega_0', 'alpha_0', 'beta_0']
    missing = [c for c in required if c not in ms_garch_df.columns]
    if missing:
        raise ValueError(f"ms_garch_df is missing required columns: {missing}")

    rows = []
    for r in (0, 1):
        sfx = str(r)
        omega = ms_garch_df.get(f'omega_{sfx}', pd.Series(dtype=float)).mean()
        alpha = ms_garch_df.get(f'alpha_{sfx}', pd.Series(dtype=float)).mean()
        beta  = ms_garch_df.get(f'beta_{sfx}', pd.Series(dtype=float)).mean()
        nu    = ms_garch_df.get(f'nu_{sfx}', pd.Series(dtype=float)).mean()
        mu    = ms_garch_df.get(f'mu_{sfx}', pd.Series(dtype=float)).mean()

        persistence = alpha + beta
        stationary  = persistence < 1.0

        # --- Robust per-row unconditional vol ---
        # Compute per-row, then take trimmed mean to avoid near-IGARCH inflation
        omega_s = ms_garch_df.get(f'omega_{sfx}', pd.Series(dtype=float))
        pers_s  = ms_garch_df.get(f'alpha_{sfx}', pd.Series(dtype=float)) + \
                  ms_garch_df.get(f'beta_{sfx}', pd.Series(dtype=float))
        denom_s = (1 - pers_s).where((1 - pers_s) > 0.001, other=np.nan)
        uncond_var_s = omega_s / denom_s
        uncond_vol_s = np.sqrt(uncond_var_s)
        uncond_vol = _trimmed_mean(uncond_vol_s)
        uncond_var = uncond_vol ** 2 if not np.isnan(uncond_vol) else np.nan

        if stationary and 0 < persistence < 1:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.nan

        # Transition probs
        p_stay_col = 'p00' if r == 0 else 'p11'
        p_stay = ms_garch_df[p_stay_col].mean() if p_stay_col in ms_garch_df.columns else np.nan

        p_switch = 1 - p_stay if not np.isnan(p_stay) else np.nan

        # Steady-state probability
        p00_mean = ms_garch_df['p00'].mean() if 'p00' in ms_garch_df.columns else np.nan
        p11_mean = ms_garch_df['p11'].mean() if 'p11' in ms_garch_df.columns else np.nan
        denom = 2 - p00_mean - p11_mean
        if not np.isnan(denom) and abs(denom) > 1e-12:
            if r == 0:
                steady_state = (1 - p11_mean) / denom
            else:
                steady_state = (1 - p00_mean) / denom
        else:
            steady_state = np.nan

        expected_duration = 1 / (1 - p_stay) if (not np.isnan(p_stay) and p_stay < 1) else np.nan

        rows.append({
            'regime': r,
            'omega': omega,
            'alpha': alpha,
            'beta': beta,
            'nu': nu,
            'mu': mu,
            'persistence': persistence,
            'stationary': stationary,
            'uncond_var': uncond_var,
            'uncond_vol': uncond_vol,
            'half_life': half_life,
            'p_stay': p_stay,
            'p_switch': p_switch,
            'steady_state': steady_state,
            'expected_duration': expected_duration,
        })

    return pd.DataFrame(rows).set_index('regime')


# ===================================================================
# B) Plots
# ===================================================================

# --- 1) Parameter comparison bar chart -----------------------------------

def plot_parameter_comparison(metrics_df: pd.DataFrame, figsize=(14, 5)):
    """Grouped bar chart of omega, alpha, beta per regime (+ separate nu panel if available)."""
    garch_params = ['omega', 'alpha', 'beta']
    has_nu = 'nu' in metrics_df.columns and metrics_df['nu'].notna().all()

    ncols = 2 if has_nu else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize,
                             gridspec_kw={'width_ratios': [3, 1]} if has_nu else None)
    if ncols == 1:
        axes = [axes]

    # --- Left panel: omega / alpha / beta ---
    ax = axes[0]
    n = len(garch_params)
    x = np.arange(n)
    width = 0.35

    vals_0 = [metrics_df.loc[0, p] for p in garch_params]
    vals_1 = [metrics_df.loc[1, p] for p in garch_params]

    bars0 = ax.bar(x - width / 2, vals_0, width, label=_REGIME_LABELS[0],
                   color=_R0_COLOR, alpha=0.7, edgecolor='black')
    bars1 = ax.bar(x + width / 2, vals_1, width, label=_REGIME_LABELS[1],
                   color=_R1_COLOR, alpha=0.7, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in garch_params], fontsize=12)
    ax.set_ylabel('Parameter Value')
    ax.set_title('MS-GARCH: GARCH Parameters by Regime', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in (bars0, bars1):
        for bar in bars:
            h = bar.get_height()
            fmt = f'{h:.2e}' if abs(h) < 0.001 else f'{h:.4f}'
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    fmt, ha='center', va='bottom', fontsize=8)

    # --- Right panel: nu (degrees of freedom) ---
    if has_nu:
        ax2 = axes[1]
        x2 = np.arange(1)
        v0 = metrics_df.loc[0, 'nu']
        v1 = metrics_df.loc[1, 'nu']
        ax2.bar(x2 - width / 2, [v0], width, color=_R0_COLOR, alpha=0.7, edgecolor='black')
        ax2.bar(x2 + width / 2, [v1], width, color=_R1_COLOR, alpha=0.7, edgecolor='black')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(['Nu (df)'], fontsize=12)
        ax2.set_title('Degrees of Freedom', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for xp, v in [(x2[0] - width / 2, v0), (x2[0] + width / 2, v1)]:
            ax2.text(xp, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


# --- 2) Persistence + unconditional volatility --------------------------

def plot_persistence_and_volatility(metrics_df: pd.DataFrame, figsize=(14, 5)):
    """Two-panel figure: persistence per regime (left) and unconditional volatility (right)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    regimes = metrics_df.index.tolist()
    labels  = [_REGIME_LABELS[r] for r in regimes]
    colors  = [_REGIME_COLORS[r] for r in regimes]

    # -- Persistence --
    ax = axes[0]
    vals = metrics_df['persistence'].values
    bars = ax.bar(labels, vals, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, label='Stationarity bound')
    ax.set_ylabel('Persistence (α + β)')
    ax.set_title('GARCH Persistence by Regime', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # -- Unconditional volatility --
    ax = axes[1]
    uvols = metrics_df['uncond_vol'].values * 100  # percentage
    bars = ax.bar(labels, uvols, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Unconditional Volatility (%)')
    ax.set_title('Unconditional Volatility by Regime', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, uvols):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f'{v:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


# --- 3) News Impact Curve -----------------------------------------------

def plot_news_impact_curves(metrics_df: pd.DataFrame, k: float = 4.0,
                            n_points: int = 500, figsize=(12, 6)):
    """
    News Impact Curve per regime.

    h_next(eps) = omega_r + alpha_r * eps^2 + beta_r * h_bar_r

    where h_bar_r is unconditional variance (if stationary) or falls back to
    omega / 0.05 as a rough proxy.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine shared eps range from the regime with highest h_bar
    h_bars = {}
    for r in metrics_df.index:
        uv = metrics_df.loc[r, 'uncond_var']
        if np.isnan(uv):
            # fallback: omega / 0.05 as rough proxy
            uv = metrics_df.loc[r, 'omega'] / 0.05
            warnings.warn(f"Regime {r} non-stationary; using fallback h_bar = {uv:.2e}")
        h_bars[r] = uv

    max_h_bar = max(h_bars.values())
    eps_max = k * np.sqrt(max_h_bar)
    eps = np.linspace(-eps_max, eps_max, n_points)

    for r in metrics_df.index:
        omega = metrics_df.loc[r, 'omega']
        alpha = metrics_df.loc[r, 'alpha']
        beta  = metrics_df.loc[r, 'beta']
        h_bar = h_bars[r]

        h_next = omega + alpha * eps ** 2 + beta * h_bar
        # Convert to volatility (%) for intuitive y-axis
        vol_next = np.sqrt(h_next) * 100

        ax.plot(eps * 100, vol_next, color=_REGIME_COLORS[r],
                linewidth=2, label=_REGIME_LABELS[r])

    ax.set_xlabel('Shock (ε × 100)', fontsize=12)
    ax.set_ylabel('Next-period Volatility (%)', fontsize=12)
    ax.set_title('News Impact Curve by Regime', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- 4) Volatility Impulse Response -------------------------------------

def plot_volatility_impulse_response(metrics_df: pd.DataFrame,
                                     shock_multiplier: float = 4.0,
                                     horizon: int = 50,
                                     figsize=(12, 6)):
    """
    Volatility impulse-response per regime.

    Start at h0 = h_bar + shock_multiplier * h_bar, then propagate with
    eps = 0: h_{t+1} = omega + beta * h_t.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for r in metrics_df.index:
        omega = metrics_df.loc[r, 'omega']
        beta  = metrics_df.loc[r, 'beta']
        uv    = metrics_df.loc[r, 'uncond_var']
        if np.isnan(uv):
            uv = metrics_df.loc[r, 'omega'] / 0.05

        h_bar = uv
        h = np.empty(horizon)
        h[0] = h_bar + shock_multiplier * h_bar  # initial shocked variance

        for t in range(1, horizon):
            h[t] = omega + beta * h[t - 1]

        vol = np.sqrt(h) * 100  # percentage
        vol_bar = np.sqrt(h_bar) * 100

        ax.plot(range(horizon), vol, color=_REGIME_COLORS[r], linewidth=2,
                label=_REGIME_LABELS[r])
        ax.axhline(vol_bar, color=_REGIME_COLORS[r], linestyle=':', linewidth=1, alpha=0.6)

    ax.set_xlabel('Days After Shock', fontsize=12)
    ax.set_ylabel('Volatility (%)', fontsize=12)
    ax.set_title('Volatility Impulse Response by Regime '
                 f'(shock = {shock_multiplier}× unconditional var)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- 5) (Optional) Smoothed regime probabilities over time ---------------

def plot_regime_probabilities_ts(ms_garch_df: pd.DataFrame, figsize=(14, 5)):
    """
    Plot *steady-state* regime probabilities averaged across firms over time.

    Uses columns ``steady_state_regime_0`` / ``steady_state_regime_1`` if they
    exist (computed earlier in the notebook), otherwise computes them from
    p00 / p11.  This is *not* a full Hamilton-filter; it's only the ergodic
    probability implied by each firm's estimated transition matrix.
    """
    df = ms_garch_df.copy()

    # Ensure steady-state cols exist
    for col, r in [('steady_state_regime_0', 0), ('steady_state_regime_1', 1)]:
        if col not in df.columns:
            if 'p00' in df.columns and 'p11' in df.columns:
                denom = 2 - df['p00'] - df['p11']
                if r == 0:
                    df[col] = (1 - df['p11']) / denom
                else:
                    df[col] = (1 - df['p00']) / denom
            else:
                print("Cannot compute regime probabilities: p00/p11 columns missing.")
                return

    if 'date' not in df.columns:
        print("Cannot plot time series: 'date' column missing.")
        return

    ts = df.groupby('date')[['steady_state_regime_0', 'steady_state_regime_1']].mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts.index, ts['steady_state_regime_0'], color=_R0_COLOR, linewidth=2,
            label=_REGIME_LABELS[0])
    ax.plot(ts.index, ts['steady_state_regime_1'], color=_R1_COLOR, linewidth=2,
            label=_REGIME_LABELS[1])
    ax.fill_between(ts.index, 0, 1, alpha=0.05, color='gray')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Ergodic Probability')
    ax.set_xlabel('Date')
    ax.set_title('MS-GARCH: Average Steady-State Regime Probabilities Over Time',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ===================================================================
# C) Per-company alpha/beta summary statistics over time
# ===================================================================

_MATURITY_MAP = {'1Y': '1Y', '3Y': '3Y', '5Y': '5Y',
                 1: '1Y', 3: '3Y', 5: '5Y',
                 '1': '1Y', '3': '3Y', '5': '5Y'}


def compute_param_summary_by_company(
    df: pd.DataFrame,
    windows_df: pd.DataFrame,
    maturity: str | int,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Compute summary statistics of the alpha and beta parameters **over the
    actual simulation window** for each company (gvkey), for a given CDS
    maturity.

    Only firms that have a window defined for the requested maturity are
    included — firms without a window (e.g. those excluded from the CDS
    comparison for that maturity) are silently dropped.

    The statistics are computed exclusively over the date range
    [start_date, end_date] from ``gvkey_maturity_simulation_windows.csv``,
    matching the window actually used in the Monte Carlo and CDS comparison.
    Two extra columns — ``window_start`` and ``window_end`` — are added to
    the output so the window is always visible alongside the statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded from ``daily_asset_returns_with_msgarch.csv``.
        Must contain columns: gvkey, date, alpha_0, beta_0, alpha_1, beta_1.
    windows_df : pd.DataFrame
        DataFrame loaded from ``gvkey_maturity_simulation_windows.csv``.
        Must contain columns: gvkey, maturity, start_date, end_date.
    maturity : str or int
        CDS maturity to filter on.  Accepts ``'1Y'``, ``'3Y'``, ``'5Y'``
        or the integer equivalents ``1``, ``3``, ``5``.
    save_path : str | None
        If provided, the long-form result is saved as a CSV to this path.

    Returns
    -------
    pd.DataFrame
        Long-form DataFrame with one row per (gvkey, regime, parameter).
        Columns: gvkey, window_start, window_end, regime, param,
        count, mean, median, std, min, Q1, Q3, max, range, IQR, n_outliers.

    Statistics computed per (gvkey, regime, parameter)
    ---------------------------------------------------
        count       – number of non-null observations inside the window
        mean        – arithmetic mean
        median      – 50th percentile
        std         – standard deviation (volatility of the parameter)
        min / max   – extremes
        Q1 / Q3     – 25th / 75th percentile
        range       – max − min
        IQR         – interquartile range (Q3 − Q1)
        n_outliers  – values beyond 1.5 × IQR from the hinges (Tukey fences)

    Examples
    --------
    >>> summary = compute_param_summary_by_company(df_msgarch, windows_df, maturity='5Y')
    >>> summary = compute_param_summary_by_company(df_msgarch, windows_df, maturity=1,
    ...                                            save_path='summary_1y.csv')
    """
    # -- Validate maturity --
    mat_key = _MATURITY_MAP.get(maturity)
    if mat_key is None:
        raise ValueError(
            f"maturity must be one of {list(_MATURITY_MAP.keys())}, got {maturity!r}"
        )

    required_df = ['gvkey', 'date', 'alpha_0', 'beta_0', 'alpha_1', 'beta_1']
    missing_df = [c for c in required_df if c not in df.columns]
    if missing_df:
        raise ValueError(f"df is missing required columns: {missing_df}")

    required_w = ['gvkey', 'maturity', 'start_date', 'end_date']
    missing_w = [c for c in required_w if c not in windows_df.columns]
    if missing_w:
        raise ValueError(f"windows_df is missing required columns: {missing_w}")

    # -- Filter windows to the requested maturity --
    win = windows_df[windows_df['maturity'] == mat_key].copy()
    win['gvkey']      = win['gvkey'].astype(int)
    win['start_date'] = pd.to_datetime(win['start_date'])
    win['end_date']   = pd.to_datetime(win['end_date'])

    if len(win) == 0:
        raise ValueError(f"No windows found for maturity '{mat_key}'.")

    # -- Prepare msgarch DataFrame --
    df_work = df.copy()
    df_work['gvkey'] = df_work['gvkey'].astype(int)
    df_work['date']  = pd.to_datetime(df_work['date'])

    # -- Helper: per-series statistics --
    def _stats(s: pd.Series) -> dict:
        s = s.dropna()
        if len(s) == 0:
            return {k: np.nan for k in
                    ['count', 'mean', 'median', 'std', 'min',
                     'Q1', 'Q3', 'max', 'range', 'IQR', 'n_outliers']}
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr    = q3 - q1
        fence_lo = q1 - 1.5 * iqr
        fence_hi = q3 + 1.5 * iqr
        return {
            'count':      int(len(s)),
            'mean':       s.mean(),
            'median':     s.median(),
            'std':        s.std(),
            'min':        s.min(),
            'Q1':         q1,
            'Q3':         q3,
            'max':        s.max(),
            'range':      s.max() - s.min(),
            'IQR':        iqr,
            'n_outliers': int(((s < fence_lo) | (s > fence_hi)).sum()),
        }

    records = []
    skipped = []

    for _, wrow in win.iterrows():
        gvkey      = int(wrow['gvkey'])
        start_date = wrow['start_date']
        end_date   = wrow['end_date']

        # Slice msgarch to this firm's window
        firm_df = df_work[
            (df_work['gvkey'] == gvkey) &
            (df_work['date']  >= start_date) &
            (df_work['date']  <= end_date)
        ]

        param_cols = ['alpha_0', 'beta_0', 'alpha_1', 'beta_1']
        firm_valid = firm_df.dropna(subset=param_cols, how='all')

        if len(firm_valid) == 0:
            skipped.append(gvkey)
            continue

        for regime in (0, 1):
            for param in ('alpha', 'beta'):
                col = f'{param}_{regime}'
                if col not in firm_valid.columns:
                    continue
                stats_dict = _stats(firm_valid[col])
                records.append({
                    'gvkey':        gvkey,
                    'window_start': start_date.date(),
                    'window_end':   end_date.date(),
                    'regime':       (f'Regime {regime} '
                                     f'({_REGIME_LABELS[regime].split("(")[1].rstrip(")")})'),
                    'param':        param,
                    **stats_dict,
                })

    if skipped:
        warnings.warn(
            f"The following gvkeys had no valid parameter rows inside the "
            f"{mat_key} window and were skipped: {skipped}"
        )

    summary = pd.DataFrame(records)

    n_firms = summary['gvkey'].nunique() if len(summary) > 0 else 0
    print(f"Maturity {mat_key}: {n_firms} firms included "
          f"(window file had {len(win)} entries).")

    if save_path is not None:
        summary.to_csv(save_path, index=False)
        print(f"Summary saved to: {save_path}")

    return summary


def display_param_summary(summary: pd.DataFrame):
    """
    Pretty-print the per-company alpha/beta summary table produced by
    :func:`compute_param_summary_by_company`.

    Pivots the tidy long-form table into a wide format:
    rows = gvkey (with window_start / window_end), columns = (regime, param, statistic).
    """
    try:
        from IPython.display import display as ipy_display
        _display = ipy_display
    except ImportError:
        _display = print

    stat_cols = ['count', 'mean', 'median', 'std', 'min', 'Q1', 'Q3',
                 'max', 'range', 'IQR', 'n_outliers']

    # Attach window columns to the index so they show up in the output
    summary_idx = summary.set_index(['gvkey', 'window_start', 'window_end'])

    # Build a pivot: index=(gvkey, window_start, window_end), columns=(regime, param, stat)
    pivot = summary_idx.pivot_table(
        index=['gvkey', 'window_start', 'window_end'],
        columns=['regime', 'param'],
        values=stat_cols,
        aggfunc='first',
    )
    # Reorder levels so it reads (regime, param, stat)
    pivot = pivot.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)

    with pd.option_context(
        'display.float_format', '{:.6f}'.format,
        'display.max_columns', None,
        'display.width', None,
    ):
        _display(pivot)

    return pivot


# ===================================================================
# Convenience wrapper
# ===================================================================

def run_ms_garch_analysis(ms_garch_df: pd.DataFrame, verbose: bool = True):
    """
    Run full MS-GARCH GARCH-dynamics analysis and print/plot everything.

    Parameters
    ----------
    ms_garch_df : pd.DataFrame
        The DataFrame loaded from ``ms_garch_parameters.csv``.
    verbose : bool
        If True, print the derived metrics table.

    Returns
    -------
    pd.DataFrame
        The derived regime metrics table.
    """
    # A) Extract & display metrics
    metrics = extract_regime_metrics(ms_garch_df)

    if verbose:
        print('=' * 80)
        print('MS-GARCH  —  DERIVED PER-REGIME METRICS')
        print('=' * 80)
        display_cols = ['omega', 'alpha', 'beta', 'nu', 'persistence',
                        'stationary', 'uncond_vol', 'half_life',
                        'p_stay', 'steady_state', 'expected_duration']
        # Filter to columns that exist
        display_cols = [c for c in display_cols if c in metrics.columns]
        with pd.option_context('display.float_format', '{:.6f}'.format):
            print(metrics[display_cols].to_string())
        print()

    # B) Plots
    plot_parameter_comparison(metrics)
    plot_persistence_and_volatility(metrics)
    plot_news_impact_curves(metrics)
    plot_volatility_impulse_response(metrics)
    plot_regime_probabilities_ts(ms_garch_df)

    return metrics
