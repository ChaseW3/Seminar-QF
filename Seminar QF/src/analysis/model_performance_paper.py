"""
Paper-ready model performance comparison for Monte Carlo CDS models.

This script compares Merton, GARCH, Regime-Switching, and MS-GARCH model-implied
spreads against market CDS spreads and identifies where each model performs best:

1) By company
2) By leverage bucket
3) By maturity (1Y/3Y/5Y)
4) By calendar year

Outputs:
- CSV tables for paper appendices/results
- PNG charts for manuscript figures
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

try:
    from src.analysis.cds_correlation import COMPANY_MAPPING, load_all_market_cds_data
    from src.utils import config
except ImportError:
    from cds_correlation import COMPANY_MAPPING, load_all_market_cds_data
    from src.utils import config


MATURITIES = [1, 3, 5]
MIN_CDS_COMPARISON_DATE = pd.Timestamp("2017-01-01")
MODEL_SPECS = {
    "merton_mc": {
        "file": "daily_monte_carlo_merton_results.csv",
        "prefix": "merton_mc_implied_spread",
        "label": "Merton",
    },
    "garch": {
        "file": "daily_monte_carlo_garch_results.csv",
        "prefix": "mc_garch_implied_spread",
        "label": "GARCH",
    },
    "rs": {
        "file": "daily_monte_carlo_regime_switching_results.csv",
        "prefix": "rs_implied_spread",
        "label": "Regime-Switching",
    },
    "msgarch": {
        "file": "daily_monte_carlo_ms_garch_results.csv",
        "prefix": "mc_ms_garch_implied_spread",
        "label": "MS-GARCH",
    },
}

# Fixed color mapping used across all model comparison plots.
MODEL_LABEL_ORDER = ["GARCH", "MS-GARCH", "Merton", "Regime-Switching"]
MODEL_COLORS = {
    "GARCH": "#4C72B0",
    "MS-GARCH": "#DD8452",
    "Merton": "#55A868",
    "Regime-Switching": "#C44E52",
}


@dataclass
class AnalysisConfig:
    output_dir: Path
    input_dir: Path
    min_obs_segment: int = 60
    min_obs_firm: int = 60
    leverage_quantiles: int = 3
    volatility_window_days: int = 63


def _to_long_market(cds_market_df: pd.DataFrame) -> pd.DataFrame:
    long_frames = []
    for mat in MATURITIES:
        col = f"cds_market_{mat}y_bps"
        if col not in cds_market_df.columns:
            continue
        tmp = cds_market_df[["date", "company_cds", col]].copy()
        tmp = tmp.rename(columns={col: "market_spread_bps"})
        tmp["maturity"] = f"{mat}y"
        long_frames.append(tmp)

    market_long = pd.concat(long_frames, ignore_index=True)
    market_long["date"] = pd.to_datetime(market_long["date"])
    market_long["market_spread_bps"] = pd.to_numeric(market_long["market_spread_bps"], errors="coerce")
    return market_long.dropna(subset=["date", "company_cds", "market_spread_bps"])


def _load_company_mapping(merton_file: Path) -> pd.DataFrame:
    merton_df = pd.read_csv(merton_file, usecols=["gvkey", "company", "liabilities_total", "asset_value", "mkt_cap", "date"])
    merton_df["date"] = pd.to_datetime(merton_df["date"], errors="coerce")
    merton_df = merton_df.dropna(subset=["gvkey", "company", "date"])

    # Prefer liabilities / asset_value; fallback liabilities / mkt_cap if asset_value missing.
    merton_df["leverage_ratio"] = np.where(
        merton_df["asset_value"].gt(0),
        merton_df["liabilities_total"] / merton_df["asset_value"],
        np.where(merton_df["mkt_cap"].gt(0), merton_df["liabilities_total"] / merton_df["mkt_cap"], np.nan),
    )

    cols = ["gvkey", "date", "company", "leverage_ratio"]
    return merton_df[cols].drop_duplicates(subset=["gvkey", "date"])


def _load_model_long(model_key: str, spec: dict, output_dir: Path) -> pd.DataFrame:
    model_path = output_dir / spec["file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    df = pd.read_csv(model_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    long_parts = []
    for mat in MATURITIES:
        col = f"{spec['prefix']}_{mat}y"
        if col not in df.columns:
            continue
        part = df[["gvkey", "date", col]].copy()
        part = part.rename(columns={col: "model_spread_bps"})
        part["model"] = model_key
        part["model_label"] = spec["label"]
        part["maturity"] = f"{mat}y"
        long_parts.append(part)

    if not long_parts:
        raise ValueError(f"No spread columns found for {model_key} in {model_path}")

    model_long = pd.concat(long_parts, ignore_index=True)
    model_long["model_spread_bps"] = pd.to_numeric(model_long["model_spread_bps"], errors="coerce")
    return model_long.dropna(subset=["gvkey", "date", "model_spread_bps"])


def build_panel(cfg: AnalysisConfig) -> pd.DataFrame:
    print("Loading market CDS data...")
    cds_market = load_all_market_cds_data(cfg.input_dir)
    market_long = _to_long_market(cds_market)

    print("Loading company and leverage mapping...")
    mapping_file = cfg.output_dir / "merged_data_with_merton.csv"
    company_map = _load_company_mapping(mapping_file)

    company_static = company_map[["gvkey", "company"]].drop_duplicates(subset=["gvkey"])
    company_static["company_cds"] = company_static["company"].map(COMPANY_MAPPING)

    panel_parts = []
    for model_key, spec in MODEL_SPECS.items():
        print(f"  - Loading {spec['label']}...")
        model_long = _load_model_long(model_key, spec, cfg.output_dir)
        model_long = model_long.merge(company_static, on="gvkey", how="left")
        model_long = model_long.merge(company_map[["gvkey", "date", "leverage_ratio"]], on=["gvkey", "date"], how="left")

        merged = model_long.merge(
            market_long,
            on=["date", "company_cds", "maturity"],
            how="inner",
        )
        panel_parts.append(merged)

    panel = pd.concat(panel_parts, ignore_index=True)
    panel = panel[panel["date"] >= MIN_CDS_COMPARISON_DATE].copy()
    panel = panel.dropna(subset=["market_spread_bps", "model_spread_bps"])
    panel["error_bps"] = panel["market_spread_bps"] - panel["model_spread_bps"]
    panel["abs_error_bps"] = panel["error_bps"].abs()
    panel["sq_error_bps"] = panel["error_bps"] ** 2
    panel["year"] = panel["date"].dt.year

    valid_lev = panel["leverage_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_lev) > 0:
        q = min(cfg.leverage_quantiles, valid_lev.nunique())
        if q >= 2:
            labels = ["Low Leverage", "Mid Leverage", "High Leverage"][:q]
            panel["leverage_group"] = pd.qcut(panel["leverage_ratio"], q=q, labels=labels, duplicates="drop")
        else:
            panel["leverage_group"] = "All"
    else:
        panel["leverage_group"] = "Unknown"

    panel["leverage_group"] = panel["leverage_group"].astype(str)
    panel = panel.sort_values(["gvkey", "maturity", "model", "date"])

    return panel


def add_change_series(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["delta_model"] = out.groupby(["gvkey", "model", "maturity"])["model_spread_bps"].diff()
    out["delta_market"] = out.groupby(["gvkey", "maturity"])["market_spread_bps"].diff()
    return out


def add_period_and_volatility_regimes(panel: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    out = panel.copy()

    out["period"] = "Pre-COVID"
    out.loc[(out["date"] >= "2020-03-01") & (out["date"] <= "2020-12-31"), "period"] = "COVID Shock"
    out.loc[(out["date"] >= "2021-01-01") & (out["date"] <= "2022-12-31"), "period"] = "Recovery"
    out.loc[out["date"] >= "2023-01-01", "period"] = "Post-Recovery"

    market_unique = (
        out[["gvkey", "date", "maturity", "market_spread_bps"]]
        .drop_duplicates(subset=["gvkey", "date", "maturity"])
        .sort_values(["gvkey", "maturity", "date"])
    )
    market_unique["market_delta"] = market_unique.groupby(["gvkey", "maturity"])["market_spread_bps"].diff()

    market_unique["rolling_market_vol"] = (
        market_unique.groupby(["gvkey", "maturity"])["market_delta"]
        .rolling(window=cfg.volatility_window_days, min_periods=max(20, cfg.volatility_window_days // 3))
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    market_unique["vol_regime"] = "Intermediate Vol"
    for mat, sub_idx in market_unique.groupby("maturity").groups.items():
        sub = market_unique.loc[sub_idx, "rolling_market_vol"].dropna()
        if len(sub) < 30:
            continue
        q1 = sub.quantile(0.33)
        q2 = sub.quantile(0.67)
        mask = market_unique["maturity"] == mat
        market_unique.loc[mask & (market_unique["rolling_market_vol"] <= q1), "vol_regime"] = "Low Vol"
        market_unique.loc[mask & (market_unique["rolling_market_vol"] >= q2), "vol_regime"] = "High Vol"

    out = out.merge(
        market_unique[["gvkey", "date", "maturity", "rolling_market_vol", "vol_regime"]],
        on=["gvkey", "date", "maturity"],
        how="left",
    )
    out["vol_regime"] = out["vol_regime"].fillna("Intermediate Vol")
    return out


def compute_performance_summary(panel: pd.DataFrame, segment_cols: list[str], min_obs: int) -> pd.DataFrame:
    records = []

    grouped = panel.groupby(segment_cols + ["model", "model_label"], dropna=False)
    for keys, grp in grouped:
        if len(grp) < min_obs:
            continue

        if not isinstance(keys, tuple):
            keys = (keys,)

        base = {k: v for k, v in zip(segment_cols + ["model", "model_label"], keys)}
        rmse = np.sqrt(np.mean(grp["sq_error_bps"]))
        mae = np.mean(grp["abs_error_bps"])
        bias = np.mean(grp["error_bps"])
        corr_lvl = grp[["model_spread_bps", "market_spread_bps"]].corr().iloc[0, 1]

        valid_chg = grp[["delta_model", "delta_market"]].dropna()
        corr_chg = valid_chg["delta_model"].corr(valid_chg["delta_market"]) if len(valid_chg) >= min_obs else np.nan

        base.update(
            {
                "n_obs": len(grp),
                "rmse_bps": rmse,
                "mae_bps": mae,
                "bias_bps": bias,
                "corr_levels": corr_lvl,
                "corr_changes": corr_chg,
            }
        )
        records.append(base)

    return pd.DataFrame(records)


def _dm_test(loss_a: pd.Series, loss_b: pd.Series) -> tuple[float, float]:
    diff = (loss_a - loss_b).dropna()
    n = len(diff)
    if n < 30:
        return np.nan, np.nan
    mean_d = diff.mean()
    std_d = diff.std(ddof=1)
    if std_d == 0 or np.isnan(std_d):
        return np.nan, np.nan
    stat = mean_d / (std_d / np.sqrt(n))
    pval = 2 * (1 - stats.norm.cdf(abs(stat)))
    return float(stat), float(pval)


def _hw_corr_diff_test(df: pd.DataFrame, model_a: str, model_b: str) -> tuple[float, float, float, float]:
    wide = df.pivot_table(
        index=["gvkey", "date", "maturity"],
        columns="model",
        values="model_spread_bps",
        aggfunc="first",
    )
    market = (
        df[["gvkey", "date", "maturity", "market_spread_bps"]]
        .drop_duplicates(subset=["gvkey", "date", "maturity"])
        .set_index(["gvkey", "date", "maturity"])
    )
    combo = wide.join(market, how="inner")
    combo = combo[[model_a, model_b, "market_spread_bps"]].dropna()
    n = len(combo)
    if n < 30:
        return np.nan, np.nan, np.nan, np.nan

    r_a = combo[model_a].corr(combo["market_spread_bps"])
    r_b = combo[model_b].corr(combo["market_spread_bps"])
    if pd.isna(r_a) or pd.isna(r_b):
        return np.nan, np.nan, np.nan, np.nan

    eps = 1e-10
    z_a = np.arctanh(np.clip(r_a, -1 + eps, 1 - eps))
    z_b = np.arctanh(np.clip(r_b, -1 + eps, 1 - eps))
    se = np.sqrt(2 / max(n - 3, 1))
    if se == 0:
        return np.nan, np.nan, r_a, r_b

    t_hw = (z_a - z_b) / se
    pval = 2 * (1 - stats.norm.cdf(abs(t_hw)))
    return float(t_hw), float(pval), float(r_a), float(r_b)


def _sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def run_pairwise_tests(panel: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dm_rows = []
    hw_rows = []
    model_pairs = list(combinations(MODEL_SPECS.keys(), 2))

    grouped = panel.groupby(group_cols, dropna=False) if group_cols else [((), panel)]
    for keys, grp in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        segment = {k: v for k, v in zip(group_cols, keys)}

        wide = grp.pivot_table(
            index=["gvkey", "date", "maturity"],
            columns="model",
            values="sq_error_bps",
            aggfunc="first",
        )

        for model_a, model_b in model_pairs:
            for maturity in grp["maturity"].dropna().unique():
                idx = grp.loc[grp["maturity"] == maturity, ["gvkey", "date", "maturity"]].drop_duplicates()
                sub = wide.loc[wide.index.isin(pd.MultiIndex.from_frame(idx))]
                if model_a not in sub.columns or model_b not in sub.columns:
                    continue

                loss_a = sub[model_a]
                loss_b = sub[model_b]
                stat, pval = _dm_test(loss_a, loss_b)
                if pd.isna(stat):
                    continue

                mean_a = loss_a.mean()
                mean_b = loss_b.mean()
                dm_rows.append(
                    {
                        **segment,
                        "maturity": maturity,
                        "model1": model_a,
                        "model2": model_b,
                        "n_obs": int(sub[[model_a, model_b]].dropna().shape[0]),
                        "DM_stat": stat,
                        "p_value": pval,
                        "significance": _sig_stars(pval),
                        "result": "Model 1 (lower RMSE)" if mean_a < mean_b else "Model 2 (lower RMSE)",
                    }
                )

            t_hw, p_hw, r_a, r_b = _hw_corr_diff_test(grp, model_a, model_b)
            if pd.isna(t_hw):
                continue
            hw_rows.append(
                {
                    **segment,
                    "Model_1": model_a,
                    "Model_2": model_b,
                    "n_obs": int(
                        grp.pivot_table(
                            index=["gvkey", "date", "maturity"],
                            columns="model",
                            values="model_spread_bps",
                            aggfunc="first",
                        )[[model_a, model_b]].dropna().shape[0]
                    ),
                    "r_market_m1": r_a,
                    "r_market_m2": r_b,
                    "t_HW": t_hw,
                    "p_value": p_hw,
                    "Significance": _sig_stars(p_hw),
                }
            )

    return pd.DataFrame(dm_rows), pd.DataFrame(hw_rows)


def compute_best_model_tables(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = panel[
        [
            "gvkey",
            "company",
            "date",
            "year",
            "maturity",
            "leverage_group",
            "period",
            "vol_regime",
            "model",
            "model_label",
            "abs_error_bps",
        ]
    ].dropna()

    idx_cols = ["gvkey", "date", "maturity"]
    best = base.loc[base.groupby(idx_cols)["abs_error_bps"].idxmin()].copy()
    best = best.rename(columns={"model": "best_model", "model_label": "best_model_label"})

    overall_share = (
        best.groupby(["maturity", "best_model_label"]).size().reset_index(name="n_best")
    )
    overall_share["share"] = overall_share["n_best"] / overall_share.groupby("maturity")["n_best"].transform("sum")

    by_leverage = (
        best.groupby(["leverage_group", "maturity", "best_model_label"]).size().reset_index(name="n_best")
    )
    by_leverage["share"] = by_leverage["n_best"] / by_leverage.groupby(["leverage_group", "maturity"])["n_best"].transform("sum")

    by_year = (
        best.groupby(["year", "maturity", "best_model_label"]).size().reset_index(name="n_best")
    )
    by_year["share"] = by_year["n_best"] / by_year.groupby(["year", "maturity"])["n_best"].transform("sum")

    by_period = (
        best.groupby(["period", "maturity", "best_model_label"]).size().reset_index(name="n_best")
    )
    by_period["share"] = by_period["n_best"] / by_period.groupby(["period", "maturity"])["n_best"].transform("sum")

    by_vol = (
        best.groupby(["vol_regime", "maturity", "best_model_label"]).size().reset_index(name="n_best")
    )
    by_vol["share"] = by_vol["n_best"] / by_vol.groupby(["vol_regime", "maturity"])["n_best"].transform("sum")

    by_period_vol = (
        best.groupby(["period", "vol_regime", "maturity", "best_model_label"]).size().reset_index(name="n_best")
    )
    by_period_vol["share"] = by_period_vol["n_best"] / by_period_vol.groupby(["period", "vol_regime", "maturity"])["n_best"].transform("sum")

    by_company = (
        best.groupby(["gvkey", "company", "maturity", "best_model_label"]).size().reset_index(name="n_best")
    )
    by_company["share"] = by_company["n_best"] / by_company.groupby(["gvkey", "maturity"])["n_best"].transform("sum")
    dominant_company = by_company.loc[by_company.groupby(["gvkey", "maturity"])["share"].idxmax()].copy()
    dominant_company = dominant_company.rename(
        columns={
            "best_model_label": "dominant_model",
            "share": "dominant_share",
        }
    )

    return {
        "best_observation_level": best,
        "best_model_share_overall": overall_share,
        "best_model_share_by_leverage": by_leverage,
        "best_model_share_by_year": by_year,
        "best_model_share_by_period": by_period,
        "best_model_share_by_volatility": by_vol,
        "best_model_share_by_period_volatility": by_period_vol,
        "best_model_by_company": dominant_company,
    }


def compute_msgarch_best_segments(best_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []

    src_period = best_tables.get("best_model_share_by_period", pd.DataFrame())
    if not src_period.empty:
        tmp = src_period[src_period["best_model_label"] == "MS-GARCH"].copy()
        if not tmp.empty:
            tmp["segment_type"] = "period"
            tmp["segment"] = tmp["period"].astype(str) + " | " + tmp["maturity"].astype(str)
            parts.append(tmp[["segment_type", "segment", "share", "n_best"]])

    src_vol = best_tables.get("best_model_share_by_volatility", pd.DataFrame())
    if not src_vol.empty:
        tmp = src_vol[src_vol["best_model_label"] == "MS-GARCH"].copy()
        if not tmp.empty:
            tmp["segment_type"] = "volatility"
            tmp["segment"] = tmp["vol_regime"].astype(str) + " | " + tmp["maturity"].astype(str)
            parts.append(tmp[["segment_type", "segment", "share", "n_best"]])

    src_period_vol = best_tables.get("best_model_share_by_period_volatility", pd.DataFrame())
    if not src_period_vol.empty:
        tmp = src_period_vol[src_period_vol["best_model_label"] == "MS-GARCH"].copy()
        if not tmp.empty:
            tmp["segment_type"] = "period_volatility"
            tmp["segment"] = (
                tmp["period"].astype(str)
                + " | "
                + tmp["vol_regime"].astype(str)
                + " | "
                + tmp["maturity"].astype(str)
            )
            parts.append(tmp[["segment_type", "segment", "share", "n_best"]])

    if not parts:
        return pd.DataFrame(columns=["segment_type", "segment", "share", "n_best"])

    msgarch_best = pd.concat(parts, ignore_index=True)
    msgarch_best = msgarch_best.sort_values(["share", "n_best"], ascending=[False, False])
    return msgarch_best


def create_plots(panel: pd.DataFrame, best_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1) Winner share by leverage and maturity
    tbl = best_tables["best_model_share_by_leverage"].copy()
    if not tbl.empty:
        plt.figure(figsize=(11, 6))
        sns.barplot(
            data=tbl,
            x="leverage_group",
            y="share",
            hue="best_model_label",
            hue_order=MODEL_LABEL_ORDER,
            palette=MODEL_COLORS,
            errorbar=None,
        )
        plt.title("Best-Performing Model Share by Leverage Group")
        plt.xlabel("Leverage Group")
        plt.ylabel("Share of Dates Where Model Has Lowest Absolute Error")
        plt.legend(title="Model", loc="best")
        plt.tight_layout()
        plt.savefig(fig_dir / "best_model_share_by_leverage.png", dpi=220)
        plt.close()

    # 2) Year-maturity heatmaps of best model shares
    by_year = best_tables["best_model_share_by_year"].copy()
    if not by_year.empty:
        for model_label in sorted(by_year["best_model_label"].unique()):
            sub = by_year[by_year["best_model_label"] == model_label]
            pivot = sub.pivot_table(index="year", columns="maturity", values="share", aggfunc="mean")
            plt.figure(figsize=(7, 5))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
            plt.title(f"{model_label}: Share of Wins by Year and Maturity")
            plt.xlabel("Maturity")
            plt.ylabel("Year")
            plt.tight_layout()
            safe_name = model_label.lower().replace("-", "_").replace(" ", "_")
            plt.savefig(fig_dir / f"heatmap_wins_{safe_name}.png", dpi=220)
            plt.close()

    # 3) RMSE trend by year
    perf_year = compute_performance_summary(panel, ["year", "maturity"], min_obs=30)
    if not perf_year.empty:
        g = sns.relplot(
            data=perf_year,
            x="year",
            y="rmse_bps",
            hue="model_label",
            hue_order=MODEL_LABEL_ORDER,
            palette=MODEL_COLORS,
            col="maturity",
            kind="line",
            marker="o",
            facet_kws={"sharey": False},
            height=4,
            aspect=1.1,
        )
        g.set_axis_labels("Year", "RMSE (bps)")
        g.fig.suptitle("RMSE Trends by Model and Maturity", y=1.02)
        g.savefig(fig_dir / "rmse_trend_by_year.png", dpi=220)
        plt.close("all")

    # 4) Firm-level RMSE distribution
    perf_firm = compute_performance_summary(panel, ["gvkey", "company", "maturity", "leverage_group"], min_obs=60)
    if not perf_firm.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=perf_firm,
            x="model_label",
            y="rmse_bps",
            hue="model_label",
            hue_order=MODEL_LABEL_ORDER,
            order=MODEL_LABEL_ORDER,
            palette=MODEL_COLORS,
            showfliers=False,
        )
        leg = plt.gca().get_legend()
        if leg is not None:
            leg.remove()
        plt.title("Firm-Level RMSE Distribution by Model")
        plt.xlabel("Model")
        plt.ylabel("RMSE (bps)")
        plt.tight_layout()
        plt.savefig(fig_dir / "firm_rmse_boxplot.png", dpi=220)
        plt.close()

    # 5) Correlation comparison
    perf_overall = compute_performance_summary(panel, ["maturity"], min_obs=60)
    if not perf_overall.empty:
        corr_long = perf_overall.melt(
            id_vars=["maturity", "model_label"],
            value_vars=["corr_levels", "corr_changes"],
            var_name="corr_type",
            value_name="correlation",
        )
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=corr_long,
            x="maturity",
            y="correlation",
            hue="model_label",
            hue_order=MODEL_LABEL_ORDER,
            palette=MODEL_COLORS,
            errorbar=None,
        )
        plt.title("Model vs Market Correlation by Maturity")
        plt.xlabel("Maturity")
        plt.ylabel("Correlation")
        plt.tight_layout()
        plt.savefig(fig_dir / "correlation_by_model_maturity.png", dpi=220)
        plt.close()

    # 6) Winner share by period (focus on different macro periods including COVID)
    by_period = best_tables.get("best_model_share_by_period", pd.DataFrame())
    if not by_period.empty:
        period_order = ["Pre-COVID", "COVID Shock", "Recovery", "Post-Recovery"]
        by_period["period"] = pd.Categorical(by_period["period"], categories=period_order, ordered=True)
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=by_period.sort_values(["period", "maturity"]),
            x="period",
            y="share",
            hue="best_model_label",
            hue_order=MODEL_LABEL_ORDER,
            palette=MODEL_COLORS,
            errorbar=None,
        )
        plt.title("Best-Performing Model Share by Macro Period")
        plt.xlabel("Period")
        plt.ylabel("Share of Wins")
        plt.tight_layout()
        plt.savefig(fig_dir / "best_model_share_by_period.png", dpi=220)
        plt.close()

    # 7) MS-GARCH wins across period x volatility x maturity
    by_period_vol = best_tables.get("best_model_share_by_period_volatility", pd.DataFrame())
    if not by_period_vol.empty:
        msg = by_period_vol[by_period_vol["best_model_label"] == "MS-GARCH"].copy()
        if not msg.empty:
            msg["period_vol"] = msg["period"].astype(str) + " | " + msg["vol_regime"].astype(str)
            pivot = msg.pivot_table(index="period_vol", columns="maturity", values="share", aggfunc="mean")
            plt.figure(figsize=(8, max(4, 0.35 * len(pivot))))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1)
            plt.title("MS-GARCH Win Share by Period and Volatility Regime")
            plt.xlabel("Maturity")
            plt.ylabel("Period | Volatility")
            plt.tight_layout()
            plt.savefig(fig_dir / "msgarch_win_share_period_volatility.png", dpi=220)
            plt.close()


def run_model_performance_paper(
    output_dir: Path | None = None,
    input_dir: Path | None = None,
    save_subdir: str = "paper_model_comparison",
) -> dict[str, Path]:
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    if input_dir is None:
        input_dir = config.INPUT_DIR

    cfg = AnalysisConfig(output_dir=Path(output_dir), input_dir=Path(input_dir))
    out_dir = cfg.output_dir / save_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PAPER MODEL COMPARISON ANALYSIS")
    print("=" * 90)

    panel = build_panel(cfg)
    panel = add_period_and_volatility_regimes(panel, cfg)
    panel = add_change_series(panel)

    print(f"Unified panel observations: {len(panel):,}")
    panel.to_csv(out_dir / "model_comparison_panel.csv", index=False)

    print("Computing performance tables...")
    overall_perf = compute_performance_summary(panel, ["maturity"], min_obs=cfg.min_obs_segment)
    leverage_perf = compute_performance_summary(panel, ["leverage_group", "maturity"], min_obs=cfg.min_obs_segment)
    year_perf = compute_performance_summary(panel, ["year", "maturity"], min_obs=cfg.min_obs_segment)
    period_perf = compute_performance_summary(panel, ["period", "maturity"], min_obs=cfg.min_obs_segment)
    volatility_perf = compute_performance_summary(panel, ["vol_regime", "maturity"], min_obs=cfg.min_obs_segment)
    period_vol_perf = compute_performance_summary(panel, ["period", "vol_regime", "maturity"], min_obs=cfg.min_obs_segment)
    firm_perf = compute_performance_summary(panel, ["gvkey", "company", "maturity", "leverage_group"], min_obs=cfg.min_obs_firm)

    overall_perf.to_csv(out_dir / "performance_overall.csv", index=False)
    leverage_perf.to_csv(out_dir / "performance_by_leverage.csv", index=False)
    year_perf.to_csv(out_dir / "performance_by_year.csv", index=False)
    period_perf.to_csv(out_dir / "performance_by_period.csv", index=False)
    volatility_perf.to_csv(out_dir / "performance_by_volatility.csv", index=False)
    period_vol_perf.to_csv(out_dir / "performance_by_period_volatility.csv", index=False)
    firm_perf.to_csv(out_dir / "performance_by_company.csv", index=False)

    print("Computing best-model dominance tables...")
    best_tables = compute_best_model_tables(panel)
    for name, df in best_tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    msgarch_best = compute_msgarch_best_segments(best_tables)
    msgarch_best.to_csv(out_dir / "msgarch_best_segments.csv", index=False)

    print("Running pairwise forecast and correlation tests...")
    dm_overall, hw_overall = run_pairwise_tests(panel, group_cols=[])
    dm_lev, hw_lev = run_pairwise_tests(panel, group_cols=["leverage_group"])
    dm_year, hw_year = run_pairwise_tests(panel, group_cols=["year"])
    dm_period, hw_period = run_pairwise_tests(panel, group_cols=["period"])
    dm_vol, hw_vol = run_pairwise_tests(panel, group_cols=["vol_regime"])
    dm_period_vol, hw_period_vol = run_pairwise_tests(panel, group_cols=["period", "vol_regime"])

    dm_overall.to_csv(out_dir / "dm_tests_overall.csv", index=False)
    dm_lev.to_csv(out_dir / "dm_tests_by_leverage.csv", index=False)
    dm_year.to_csv(out_dir / "dm_tests_by_year.csv", index=False)
    dm_period.to_csv(out_dir / "dm_tests_by_period.csv", index=False)
    dm_vol.to_csv(out_dir / "dm_tests_by_volatility.csv", index=False)
    dm_period_vol.to_csv(out_dir / "dm_tests_by_period_volatility.csv", index=False)

    hw_overall.to_csv(out_dir / "hw_tests_overall.csv", index=False)
    hw_lev.to_csv(out_dir / "hw_tests_by_leverage.csv", index=False)
    hw_year.to_csv(out_dir / "hw_tests_by_year.csv", index=False)
    hw_period.to_csv(out_dir / "hw_tests_by_period.csv", index=False)
    hw_vol.to_csv(out_dir / "hw_tests_by_volatility.csv", index=False)
    hw_period_vol.to_csv(out_dir / "hw_tests_by_period_volatility.csv", index=False)

    print("Creating publication-quality figures...")
    create_plots(panel, best_tables, out_dir)

    print("Writing concise text summary...")
    with open(out_dir / "analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("Model Performance Comparison Summary\n")
        f.write("=" * 44 + "\n\n")
        f.write(f"Total panel rows: {len(panel):,}\n")
        f.write(f"Firms: {panel['gvkey'].nunique()}\n")
        f.write(f"Years: {int(panel['year'].min())} - {int(panel['year'].max())}\n\n")

        if not overall_perf.empty:
            f.write("Overall RMSE by maturity and model:\n")
            for mat in sorted(overall_perf["maturity"].unique()):
                sub = overall_perf[overall_perf["maturity"] == mat].sort_values("rmse_bps")
                f.write(f"\n{mat}:\n")
                for _, row in sub.iterrows():
                    f.write(
                        f"  - {row['model_label']}: RMSE={row['rmse_bps']:.2f} bps, "
                        f"Corr(levels)={row['corr_levels']:.3f}, Corr(changes)={row['corr_changes']:.3f}\n"
                    )

        if not best_tables["best_model_share_overall"].empty:
            f.write("\nBest-model share by maturity:\n")
            for mat in sorted(best_tables["best_model_share_overall"]["maturity"].unique()):
                sub = best_tables["best_model_share_overall"]
                sub = sub[sub["maturity"] == mat].sort_values("share", ascending=False)
                f.write(f"\n{mat}:\n")
                for _, row in sub.iterrows():
                    f.write(f"  - {row['best_model_label']}: {100 * row['share']:.1f}%\n")

        if not msgarch_best.empty:
            f.write("\nWhere MS-GARCH is strongest (top segments):\n")
            top = msgarch_best.head(12)
            for _, row in top.iterrows():
                f.write(
                    f"  - [{row['segment_type']}] {row['segment']}: "
                    f"win share={100 * row['share']:.1f}% (n={int(row['n_best'])})\n"
                )

    outputs = {
        "root": out_dir,
        "panel": out_dir / "model_comparison_panel.csv",
        "overall": out_dir / "performance_overall.csv",
        "leverage": out_dir / "performance_by_leverage.csv",
        "year": out_dir / "performance_by_year.csv",
        "period": out_dir / "performance_by_period.csv",
        "volatility": out_dir / "performance_by_volatility.csv",
        "period_volatility": out_dir / "performance_by_period_volatility.csv",
        "firm": out_dir / "performance_by_company.csv",
        "msgarch_best": out_dir / "msgarch_best_segments.csv",
        "summary": out_dir / "analysis_summary.txt",
        "figures": out_dir / "figures",
    }

    print("Analysis complete.")
    print(f"Results saved in: {out_dir}")
    return outputs


if __name__ == "__main__":
    run_model_performance_paper()