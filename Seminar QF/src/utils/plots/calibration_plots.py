import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional, List

# Plotting for the affine calibration module


def plot_timeseries(
    cal_df: pd.DataFrame,
    gvkey: int,
    title: Optional[str] = None,
    figsize: tuple = (14, 5),
    save_path: Optional[str] = None,
):
    # Time-series for market, raw-model, and calibrated spreads for one firm
    firm = cal_df[cal_df['gvkey'] == gvkey].sort_values('date').copy()
    if firm.empty:
        print(f"No data for gvkey {gvkey}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(firm['date'], firm['market'], label='Market CDS', color='black', linewidth=1.2, alpha=0.9)
    ax.plot(firm['date'], firm['model_raw'], label='Model (raw)', color='tab:blue', linewidth=0.9, alpha=0.6, linestyle='--')
    ax.plot(firm['date'], firm['calibrated'], label='Model (calibrated)', color='tab:red', linewidth=1.0, alpha=0.85)

    company = firm['company'].iloc[0] if 'company' in firm.columns else f'Firm {gvkey}'
    ax.set_title(title or f'{company} – Raw vs Calibrated CDS Spread')
    ax.set_xlabel('Date')
    ax.set_ylabel('CDS Spread (bps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_scatter_comparison(
    cal_df: pd.DataFrame,
    model_name: str = '',
    maturity: str = '',
    figsize: tuple = (13, 5.5),
    axis_limit: Optional[float] = None,
    save_path: Optional[str] = None,
):
    # Scatter plots for raw model vs market and calibrated vs market
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, col, label in zip(
        axes,
        ['model_raw', 'calibrated'],
        ['Raw Model', 'Calibrated Model'],
    ):
        valid = cal_df[['market', col]].dropna()
        if valid.empty:
            ax.set_title(f'{label} – no data')
            continue

        ax.scatter(valid['market'], valid[col], s=3, alpha=0.15, color='steelblue')

        maxv = axis_limit or max(valid['market'].quantile(0.99), valid[col].quantile(0.99)) * 1.05
        ax.plot([0, maxv], [0, maxv], 'r--', linewidth=1, label='45° line')

        ax.set_xlim(0, maxv)
        ax.set_ylim(0, maxv)
        ax.set_xlabel('Market CDS (bps)')
        ax.set_ylabel(f'{label} CDS (bps)')
        ax.set_title(f'{label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(f'{model_name} {maturity} – Market vs Model Spreads', fontsize=13, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_rolling_betas(
    cal_df: pd.DataFrame,
    model_name: str = '',
    maturity: str = '',
    figsize: tuple = (14, 7),
    save_path: Optional[str] = None,
):
    # Line plots of cross-sectional median betas over time with IQR bands
    valid = cal_df.dropna(subset=['beta0', 'beta1'])
    if valid.empty:
        print("No calibration parameters to plot.")
        return

    daily = valid.groupby('date').agg(
        b0_median=('beta0', 'median'),
        b0_q25=('beta0', lambda x: x.quantile(0.25)),
        b0_q75=('beta0', lambda x: x.quantile(0.75)),
        b1_median=('beta1', 'median'),
        b1_q25=('beta1', lambda x: x.quantile(0.25)),
        b1_q75=('beta1', lambda x: x.quantile(0.75)),
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['date'])

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # β₀
    ax = axes[0]
    ax.plot(daily['date'], daily['b0_median'], color='tab:blue', label='Median β₀')
    ax.fill_between(daily['date'], daily['b0_q25'], daily['b0_q75'], alpha=0.2, color='tab:blue')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel('β₀ (intercept, bps)')
    ax.set_title(f'{model_name} {maturity} – Rolling Calibration Parameters')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # β₁
    ax = axes[1]
    ax.plot(daily['date'], daily['b1_median'], color='tab:orange', label='Median β₁')
    ax.fill_between(daily['date'], daily['b1_q25'], daily['b1_q75'], alpha=0.2, color='tab:orange')
    ax.axhline(1, color='grey', linestyle=':', linewidth=0.8, label='β₁ = 1 (no scaling)')
    ax.set_ylabel('β₁ (slope)')
    ax.set_xlabel('Date')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_rmse_comparison(
    metrics_dict: Dict[str, Dict[str, Dict[str, float]]],
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
):
    # Grouped bar chart for RMSE raw vs callibrated for each model and maturity
    rows = []
    for model, mats in metrics_dict.items():
        for mat, m in mats.items():
            rows.append({
                'Model': model,
                'Maturity': mat,
                'RMSE Raw': m.get('rmse_raw', np.nan),
                'RMSE Calibrated': m.get('rmse_calibrated', np.nan),
            })
    df = pd.DataFrame(rows)

    mats = df['Maturity'].unique()
    models = df['Model'].unique()
    x = np.arange(len(models))
    width = 0.15
    n_mats = len(mats)

    fig, ax = plt.subplots(figsize=figsize)
    for i, mat in enumerate(mats):
        sub = df[df['Maturity'] == mat]
        offset = (i - n_mats / 2 + 0.5) * width * 2
        bars_raw = ax.bar(x + offset - width / 2, sub['RMSE Raw'], width,
                          label=f'{mat} Raw', alpha=0.5, edgecolor='black', linewidth=0.5)
        bars_cal = ax.bar(x + offset + width / 2, sub['RMSE Calibrated'], width,
                          label=f'{mat} Calibrated', alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('RMSE (bps)')
    ax.set_title('RMSE: Raw vs Calibrated Model Spreads')
    ax.legend(fontsize=8, ncol=n_mats * 2)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def metrics_summary_table(
    metrics_dict: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    # Build dataset of all evaluation metrics
    rows = []
    for model, mats in metrics_dict.items():
        for mat, m in mats.items():
            row = {'Model': model, 'Maturity': mat}
            row.update(m)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_mean_median_vs_market(
    cal_df: pd.DataFrame,
    model_name: str = '',
    maturity: str = '',
    figsize: tuple = (15, 10),
    save_path: Optional[str] = None,
):
    # Time series cross-sectional mean and median of calibrated, raw and market spreads
    valid = cal_df.dropna(subset=['calibrated', 'market']).copy()
    if valid.empty:
        print("No valid calibrated data to plot.")
        return

    valid['date'] = pd.to_datetime(valid['date'])
    daily = valid.groupby('date').agg(
        mean_market=('market', 'mean'),
        mean_raw=('model_raw', 'mean'),
        mean_cal=('calibrated', 'mean'),
        median_market=('market', 'median'),
        median_raw=('model_raw', 'median'),
        median_cal=('calibrated', 'median'),
        n_firms=('gvkey', 'nunique'),
    ).reset_index().sort_values('date')

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for ax, agg in zip(axes, ['mean', 'median']):
        ax.plot(daily['date'], daily[f'{agg}_market'],
                color='black', linewidth=2.2, label='Market CDS', alpha=0.85)
        ax.plot(daily['date'], daily[f'{agg}_raw'],
                color='tab:blue', linewidth=1.2, linestyle='--', label='Raw Model', alpha=0.55)
        ax.plot(daily['date'], daily[f'{agg}_cal'],
                color='tab:red', linewidth=1.6, label='Calibrated Model', alpha=0.85)

        ax.set_title(
            f'{agg.capitalize()} Spreads – {model_name} {maturity}',
            fontsize=12, fontweight='bold',
        )
        ax.set_ylabel(f'{agg.capitalize()} CDS Spread (bps)')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def _assign_leverage_groups(
    cal_df: pd.DataFrame,
    merton_file,
    leverage_metric: str = 'debt_to_equity',
    n_groups: int = 3,
) -> tuple:
    # Attach a leverage_group column to cal_df using firm-level leverage from the Merton file
    lev_df = pd.read_csv(merton_file, parse_dates=['date'])
    required = ['gvkey', 'date', 'liabilities_total', 'mkt_cap', 'asset_value']
    lev_df = lev_df[[c for c in required if c in lev_df.columns]].copy()

    if leverage_metric == 'debt_to_equity':
        lev_df['leverage_value'] = lev_df['liabilities_total'] / lev_df['mkt_cap'].replace(0, np.nan)
        leverage_label = 'Debt / Equity'
    else:
        lev_df['leverage_value'] = lev_df['liabilities_total'] / lev_df['asset_value'].replace(0, np.nan)
        leverage_label = 'Debt / Assets'

    # Median leverage per firm
    firm_lev = (
        lev_df.dropna(subset=['leverage_value'])
        .groupby('gvkey', as_index=False)['leverage_value']
        .median()
    )
    # Quantile-based groups
    quantiles = np.linspace(0, 1, n_groups + 1)
    edges = np.unique(firm_lev['leverage_value'].quantile(quantiles).values)
    actual = len(edges) - 1
    group_names = [f'Leverage Group {i+1} (Low\u2192High)' for i in range(actual)]
    edges_adj = edges.copy()
    edges_adj[0] -= 1e-12
    edges_adj[-1] += 1e-12
    firm_lev['leverage_group'] = pd.cut(
        firm_lev['leverage_value'], bins=edges_adj, labels=group_names, include_lowest=True,
    )

    out = cal_df.copy()
    out = out.merge(firm_lev[['gvkey', 'leverage_group', 'leverage_value']], on='gvkey', how='left')
    return out, leverage_label, group_names


def plot_mean_median_by_leverage(
    cal_df: pd.DataFrame,
    merton_file,
    model_name: str = '',
    maturity: str = '',
    leverage_metric: str = 'debt_to_equity',
    n_groups: int = 3,
    aggregation: str = 'mean',
    figsize_per_group: tuple = (15, 4.2),
    save_path: Optional[str] = None,
):
    # One subplot per leverage group for aggregated calibrated-model vs market spreads over time
    merged, leverage_label, group_names = _assign_leverage_groups(
        cal_df, merton_file, leverage_metric, n_groups,
    )
    merged = merged.dropna(subset=['leverage_group', 'calibrated', 'market'])
    merged['date'] = pd.to_datetime(merged['date'])

    groups_present = [g for g in group_names if g in merged['leverage_group'].unique()]
    n_panels = len(groups_present)
    if n_panels == 0:
        print("No data with leverage groups available.")
        return

    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(figsize_per_group[0], figsize_per_group[1] * n_panels),
                             sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ax, group_name in zip(axes, groups_present):
        sub = merged[merged['leverage_group'] == group_name].copy()
        daily = sub.groupby('date', as_index=False).agg(
            agg_market=('market', aggregation),
            agg_raw=('model_raw', aggregation),
            agg_cal=('calibrated', aggregation),
            n_firms=('gvkey', 'nunique'),
        ).sort_values('date')

        ax.plot(daily['date'], daily['agg_market'],
                color='black', linewidth=2.4, label='Market CDS', alpha=0.85)
        ax.plot(daily['date'], daily['agg_raw'],
                color='tab:blue', linewidth=1.0, linestyle='--', label='Raw Model', alpha=0.45)
        ax.plot(daily['date'], daily['agg_cal'],
                color='tab:red', linewidth=1.6, label='Calibrated Model', alpha=0.85)

        ax.set_title(
            f"{group_name}",
            fontsize=11, fontweight='bold',
        )
        ax.set_ylabel(f'{aggregation.capitalize()} CDS (bps)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.suptitle(
        f'{aggregation.capitalize()} Market vs {model_name} CDS by {leverage_label} ({maturity})',
        fontsize=14, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Print leverage group
    firm_info = merged[['gvkey', 'company', 'leverage_group', 'leverage_value']].drop_duplicates('gvkey')
    firm_info = firm_info.sort_values(['leverage_group', 'leverage_value'])
    print(f"\nFirm membership ({leverage_label}):")
    for gn in groups_present:
        g = firm_info[firm_info['leverage_group'] == gn]
        print(f"\n  {gn}  ({len(g)} firms)")
        for _, r in g.iterrows():
            print(f"    gvkey={r['gvkey']}  {r.get('company',''):<35s}  leverage={r['leverage_value']:.4f}")
