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

# Compare calibrated Merton, GARCH, Regime-Switching, and MS-GARCH implied spreads
# against market CDS spreads

MATURITIES = [1, 3, 5]
MIN_CDS_COMPARISON_DATE = pd.Timestamp("2017-01-01")
DEFAULT_PERIOD_RANGES = {
    "Pre-COVID": (None, "2020-02-29"),
    "COVID Shock": ("2020-03-01", "2020-12-31"),
    "Recovery": ("2021-01-01", "2022-12-31"),
    "Post-Recovery": ("2023-01-01", None),
}
MODEL_SPECS = {
    "merton_mc": {
        "calibrated_file_prefix": "calibrated_spreads_merton",
        "label": "Merton Calibrated",
    },
    "garch": {
        "calibrated_file_prefix": "calibrated_spreads_garch",
        "label": "GARCH Calibrated",
    },
    "rs": {
        "calibrated_file_prefix": "calibrated_spreads_regime_switching",
        "label": "Regime-Switching Calibrated",
    },
    "msgarch": {
        "calibrated_file_prefix": "calibrated_spreads_ms_garch",
        "label": "MS-GARCH Calibrated",
    },
}

MODEL_LABEL_ORDER = [
    "GARCH Calibrated",
    "MS-GARCH Calibrated",
    "Merton Calibrated",
    "Regime-Switching Calibrated",
]
MODEL_COLORS = {
    "GARCH Calibrated": "#4C72B0",
    "MS-GARCH Calibrated": "#DD8452",
    "Merton Calibrated": "#55A868",
    "Regime-Switching Calibrated": "#C44E52",
}
MSGARCH_MODEL_LABEL = MODEL_SPECS["msgarch"]["label"]


@dataclass
class AnalysisConfig:
    output_dir: Path
    input_dir: Path
    min_obs_segment: int = 60
    min_obs_firm: int = 60
    leverage_quantiles: int = 3
    volatility_window_days: int = 63
    period_ranges: dict[str, tuple[str | None, str | None]] | None = None


def _normalize_maturity_values(values: list[int | str]) -> list[str]:
    normalized = []
    for value in values:
        if isinstance(value, int):
            normalized.append(f"{value}y")
            continue

        token = str(value).strip().lower()
        if token.endswith("y"):
            normalized.append(token)
            continue

        if token.isdigit():
            normalized.append(f"{token}y")
            continue

        raise ValueError(f"Invalid maturity selector: {value}. Use 1, 3, 5, or '1y'/'3y'/'5y'.")

    return normalized


def _to_selector_list(selector: int | str | list[int | str] | None) -> list[int | str] | None:
    if selector is None:
        return None
    if isinstance(selector, list):
        return selector
    return [selector]


def _apply_runtime_filters(
    panel: pd.DataFrame,
    maturity: int | str | list[int | str] | None,
    period: str | list[str] | None,
    leverage_group: str | list[str] | None,
) -> tuple[pd.DataFrame, dict[str, str | int]]:
    filtered = panel.copy()
    metadata: dict[str, str | int] = {
        "rows_before_filter": int(len(panel)),
    }

    maturity_values = _to_selector_list(maturity)
    period_values = _to_selector_list(period)
    leverage_values = _to_selector_list(leverage_group)

    if maturity_values is not None:
        selected = _normalize_maturity_values(maturity_values)
        available = set(filtered["maturity"].dropna().unique())
        invalid = sorted(set(selected) - available)
        if invalid:
            raise ValueError(f"Invalid maturity selector(s): {invalid}. Available values: {sorted(available)}")
        filtered = filtered[filtered["maturity"].isin(selected)].copy()
        metadata["selected_maturity"] = ",".join(selected)
    else:
        metadata["selected_maturity"] = "ALL"

    if period_values is not None:
        selected = [str(p).strip() for p in period_values]
        available = set(filtered["period"].dropna().unique())
        invalid = sorted(set(selected) - available)
        if invalid:
            raise ValueError(f"Invalid period selector(s): {invalid}. Available values: {sorted(available)}")
        filtered = filtered[filtered["period"].isin(selected)].copy()
        metadata["selected_period"] = ",".join(selected)
    else:
        metadata["selected_period"] = "ALL"

    if leverage_values is not None:
        selected = [str(v).strip() for v in leverage_values]
        available = set(filtered["leverage_group"].dropna().unique())
        invalid = sorted(set(selected) - available)
        if invalid:
            raise ValueError(f"Invalid leverage selector(s): {invalid}. Available values: {sorted(available)}")
        filtered = filtered[filtered["leverage_group"].isin(selected)].copy()
        metadata["selected_leverage_group"] = ",".join(selected)
    else:
        metadata["selected_leverage_group"] = "ALL"

    metadata["rows_after_filter"] = int(len(filtered))
    metadata["firms_after_filter"] = int(filtered["gvkey"].nunique())
    if len(filtered) > 0:
        metadata["date_min"] = str(filtered["date"].min().date())
        metadata["date_max"] = str(filtered["date"].max().date())
    else:
        metadata["date_min"] = "NA"
        metadata["date_max"] = "NA"

    return filtered, metadata


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

    # Prefer liabilities / asset_value, only fallback to liabilities / mkt_cap
    merton_df["leverage_ratio"] = np.where(
        merton_df["asset_value"].gt(0),
        merton_df["liabilities_total"] / merton_df["asset_value"],
        np.where(merton_df["mkt_cap"].gt(0), merton_df["liabilities_total"] / merton_df["mkt_cap"], np.nan),
    )

    cols = ["gvkey", "date", "company", "leverage_ratio"]
    return merton_df[cols].drop_duplicates(subset=["gvkey", "date"])


def _load_model_long(model_key: str, spec: dict, output_dir: Path) -> pd.DataFrame:
    long_parts = []
    calibrated_col = spec.get("calibrated_col", "calibrated")
    calibrated_dir = output_dir / "calibration"
    for mat in MATURITIES:
        model_path = calibrated_dir / f"{spec['calibrated_file_prefix']}_{mat}y.csv"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing calibrated model file: {model_path}")

        df = pd.read_csv(model_path)
        required_cols = {"gvkey", "date", calibrated_col}
        missing_cols = sorted(required_cols - set(df.columns))
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {model_path.name}: {missing_cols}. "
                f"Expected at least {sorted(required_cols)}"
            )

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        part = df[["gvkey", "date", calibrated_col]].copy()
        part = part.rename(columns={calibrated_col: "model_spread_bps"})
        part["model"] = model_key
        part["model_label"] = spec["label"]
        part["maturity"] = f"{mat}y"
        long_parts.append(part)

    if not long_parts:
        raise ValueError(f"No calibrated spread data found for {model_key}")

    model_long = pd.concat(long_parts, ignore_index=True)
    model_long["model_spread_bps"] = pd.to_numeric(model_long["model_spread_bps"], errors="coerce")
    return model_long.dropna(subset=["gvkey", "date", "model_spread_bps"])


# Build the unified panel: merge all model spreads with market CDS data
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


# Add day-over-day change columns for model and market spreads
def add_change_series(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["delta_model"] = out.groupby(["gvkey", "model", "maturity"])["model_spread_bps"].diff()
    out["delta_market"] = out.groupby(["gvkey", "maturity"])["market_spread_bps"].diff()
    return out

def add_period_and_volatility_regimes(panel: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    # Assign macro-period labels and rolling volatility regimes
    out = panel.copy()

    period_ranges = DEFAULT_PERIOD_RANGES.copy()
    if cfg.period_ranges:
        period_ranges.update(cfg.period_ranges)

    out["period"] = "Unassigned"
    for label, (start, end) in period_ranges.items():
        mask = pd.Series(True, index=out.index)
        if start is not None:
            mask &= out["date"] >= pd.Timestamp(start)
        if end is not None:
            mask &= out["date"] <= pd.Timestamp(end)
        out.loc[mask, "period"] = label

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


# Compute RMSE, MAE, bias, and correlation metrics per segment
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


def rank_models_by_segment(
    perf_df: pd.DataFrame,
    segment_cols: list[str],
    metric_col: str,
    higher_is_better: bool,
    metric_name: str,
) -> pd.DataFrame:
    cols = segment_cols + ["model", "model_label", metric_col]
    ranked = perf_df[cols].dropna(subset=[metric_col]).copy()
    if ranked.empty:
        return pd.DataFrame(columns=cols + ["rank_position", "n_models", "metric"])

    sort_cols = segment_cols + [metric_col, "model_label"]
    ascending = [True] * len(segment_cols) + [not higher_is_better, True]
    ranked = ranked.sort_values(sort_cols, ascending=ascending)

    grouped = ranked.groupby(segment_cols, dropna=False)
    ranked["rank_position"] = grouped.cumcount() + 1
    ranked["n_models"] = grouped["model"].transform("size")
    ranked["metric"] = metric_name
    return ranked


def compute_rank_shares(rank_df: pd.DataFrame, segment_cols: list[str]) -> pd.DataFrame:
    if rank_df.empty:
        return pd.DataFrame(columns=segment_cols + ["rank_position", "model_label", "n_rank", "share"])

    out = rank_df.groupby(segment_cols + ["rank_position", "model_label"], dropna=False).size().reset_index(name="n_rank")
    out["share"] = out["n_rank"] / out.groupby(segment_cols + ["rank_position"], dropna=False)["n_rank"].transform("sum")
    return out


def compute_segment_rank_tables(
    perf_tables: dict[str, tuple[pd.DataFrame, list[str]]],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    metric_specs = [
        ("rmse_bps", False),
        ("corr_levels", True),
        ("corr_changes", True),
    ]

    rank_tables: dict[str, pd.DataFrame] = {}
    rank_share_tables: dict[str, pd.DataFrame] = {}
    for table_name, (perf_df, segment_cols) in perf_tables.items():
        for metric_col, higher_is_better in metric_specs:
            rank_name = f"rank_{metric_col}_{table_name}"
            ranked = rank_models_by_segment(
                perf_df=perf_df,
                segment_cols=segment_cols,
                metric_col=metric_col,
                higher_is_better=higher_is_better,
                metric_name=metric_col,
            )
            rank_tables[rank_name] = ranked
            rank_share_tables[f"{rank_name}_share"] = compute_rank_shares(ranked, segment_cols)

    return rank_tables, rank_share_tables


def _write_rank_summary_by_maturity(
    f,
    rank_df: pd.DataFrame,
    metric_col: str,
    title: str,
) -> None:
    if rank_df.empty or "maturity" not in rank_df.columns:
        return

    f.write(f"\n{title} by maturity (rank 1 to 4):\n")
    for mat in sorted(rank_df["maturity"].dropna().unique()):
        sub = rank_df[rank_df["maturity"] == mat].sort_values("rank_position")
        if sub.empty:
            continue

        f.write(f"\n{mat}:\n")
        def _fmt(v: float) -> str:
            return f"{v:.2f}" if metric_col == "rmse_bps" else f"{v:.3f}"

        for _, row in sub.iterrows():
            unit = " bps" if metric_col == "rmse_bps" else ""
            f.write(
                f"  - Rank {int(row['rank_position'])}: "
                f"{row['model_label']} ({metric_col}={_fmt(row[metric_col])}{unit})\n"
            )


# Diebold-Mariano test for predictive accuracy
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


# Hotelling-Williams test for comparing two dependent correlations with market
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


# Run DM and Hotelling-Williams tests for all model pairs within each segment
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


# For each observation, pick the model with lowest absolute error
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


# Identify segments where MS-GARCH has the highest win share
def compute_msgarch_best_segments(best_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []

    src_period = best_tables.get("best_model_share_by_period", pd.DataFrame())
    if not src_period.empty:
        tmp = src_period[src_period["best_model_label"] == MSGARCH_MODEL_LABEL].copy()
        if not tmp.empty:
            tmp["segment_type"] = "period"
            tmp["segment"] = tmp["period"].astype(str) + " | " + tmp["maturity"].astype(str)
            parts.append(tmp[["segment_type", "segment", "share", "n_best"]])

    src_vol = best_tables.get("best_model_share_by_volatility", pd.DataFrame())
    if not src_vol.empty:
        tmp = src_vol[src_vol["best_model_label"] == MSGARCH_MODEL_LABEL].copy()
        if not tmp.empty:
            tmp["segment_type"] = "volatility"
            tmp["segment"] = tmp["vol_regime"].astype(str) + " | " + tmp["maturity"].astype(str)
            parts.append(tmp[["segment_type", "segment", "share", "n_best"]])

    src_period_vol = best_tables.get("best_model_share_by_period_volatility", pd.DataFrame())
    if not src_period_vol.empty:
        tmp = src_period_vol[src_period_vol["best_model_label"] == MSGARCH_MODEL_LABEL].copy()
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


def create_plots(
    panel: pd.DataFrame,
    best_tables: dict[str, pd.DataFrame],
    output_dir: Path,
    rank_share_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    # Generate and save plots

    sns.set_theme(style="whitegrid")
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Winner share by leverage and maturity
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

    # Year-maturity heatmaps of best model shares
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

    # RMSE trends by year
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

    # Firm-level RMSE distribution
    perf_firm = compute_performance_summary(panel, ["gvkey", "company", "maturity", "leverage_group"], min_obs=60)
    if not perf_firm.empty:
        # Box plot firm-level RMSE distribution by model
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

    # Correlation comparison
    perf_overall = compute_performance_summary(panel, ["maturity"], min_obs=60)
    if not perf_overall.empty:
        corr_long = perf_overall.melt(
            id_vars=["maturity", "model_label"],
            value_vars=["corr_levels", "corr_changes"],
            var_name="corr_type",
            value_name="correlation",
        )
        # Bar chart model vs market correlation by maturity
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

    # Winner share by period
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

    # MS-GARCH wins across period, volatility and maturity
    by_period_vol = best_tables.get("best_model_share_by_period_volatility", pd.DataFrame())
    if not by_period_vol.empty:
        msg = by_period_vol[by_period_vol["best_model_label"] == MSGARCH_MODEL_LABEL].copy()
        if not msg.empty:
            msg["period_vol"] = msg["period"].astype(str) + " | " + msg["vol_regime"].astype(str)
            pivot = msg.pivot_table(index="period_vol", columns="maturity", values="share", aggfunc="mean")
            plt.figure(figsize=(8, max(4, 0.35 * len(pivot))))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Greens", vmin=0, vmax=1)
            plt.title(f"{MSGARCH_MODEL_LABEL} Win Share by Period and Volatility Regime")
            plt.xlabel("Maturity")
            plt.ylabel("Period | Volatility")
            plt.tight_layout()
            plt.savefig(fig_dir / "msgarch_win_share_period_volatility.png", dpi=220)
            plt.close()

    if rank_share_tables is None:
        rank_share_tables = {}

    rmse_lev = rank_share_tables.get("rank_rmse_bps_by_leverage_share", pd.DataFrame())
    if not rmse_lev.empty:
        for rank_value, title_stub, file_stub in [
            (2, "Rank 2 RMSE Share", "rank2"),
            (4, "Rank 4 RMSE Share", "rank4"),
        ]:
            sub = rmse_lev[rmse_lev["rank_position"] == rank_value].copy()
            if sub.empty:
                continue
            plt.figure(figsize=(11, 6))
            sns.barplot(
                data=sub,
                x="leverage_group",
                y="share",
                hue="model_label",
                hue_order=MODEL_LABEL_ORDER,
                palette=MODEL_COLORS,
                errorbar=None,
            )
            plt.title(f"{title_stub} by Leverage Group")
            plt.xlabel("Leverage Group")
            plt.ylabel("Share")
            plt.tight_layout()
            plt.savefig(fig_dir / f"rmse_{file_stub}_rank_share_by_leverage.png", dpi=220)
            plt.close()

    rmse_period = rank_share_tables.get("rank_rmse_bps_by_period_share", pd.DataFrame())
    if not rmse_period.empty:
        period_order = ["Pre-COVID", "COVID Shock", "Recovery", "Post-Recovery"]
        rmse_period["period"] = pd.Categorical(rmse_period["period"], categories=period_order, ordered=True)
        for rank_value, title_stub, file_stub in [
            (2, "Rank 2 RMSE Share", "rank2"),
            (4, "Rank 4 RMSE Share", "rank4"),
        ]:
            sub = rmse_period[rmse_period["rank_position"] == rank_value].copy()
            if sub.empty:
                continue
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=sub.sort_values(["period", "maturity"]),
                x="period",
                y="share",
                hue="model_label",
                hue_order=MODEL_LABEL_ORDER,
                palette=MODEL_COLORS,
                errorbar=None,
            )
            plt.title(f"{title_stub} by Macro Period")
            plt.xlabel("Period")
            plt.ylabel("Share")
            plt.tight_layout()
            plt.savefig(fig_dir / f"rmse_{file_stub}_rank_share_by_period.png", dpi=220)
            plt.close()

    corr_period = rank_share_tables.get("rank_corr_levels_by_period_share", pd.DataFrame())
    if not corr_period.empty:
        sub = corr_period[corr_period["rank_position"] == 4].copy()
        if not sub.empty:
            period_order = ["Pre-COVID", "COVID Shock", "Recovery", "Post-Recovery"]
            sub["period"] = pd.Categorical(sub["period"], categories=period_order, ordered=True)
            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=sub.sort_values(["period", "maturity"]),
                x="period",
                y="share",
                hue="model_label",
                hue_order=MODEL_LABEL_ORDER,
                palette=MODEL_COLORS,
                errorbar=None,
            )
            plt.title("Rank 4 corr_levels Share by Macro Period")
            plt.xlabel("Period")
            plt.ylabel("Share")
            plt.tight_layout()
            plt.savefig(fig_dir / "corr_levels_rank4_share_by_period.png", dpi=220)
            plt.close()


def run_model_performance_paper(
    output_dir: Path | None = None,
    input_dir: Path | None = None,
    save_subdir: str = "paper_model_comparison",
    maturity: int | str | list[int | str] | None = None,
    period: str | list[str] | None = None,
    leverage_group: str | list[str] | None = None,
    period_ranges: dict[str, tuple[str | None, str | None]] | None = None,
) -> dict[str, Path]:
    # Build panel, compute all metrics, run tests, save outputs
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    if input_dir is None:
        input_dir = config.INPUT_DIR

    cfg = AnalysisConfig(output_dir=Path(output_dir), input_dir=Path(input_dir), period_ranges=period_ranges)
    out_dir = cfg.output_dir / save_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("PAPER MODEL COMPARISON ANALYSIS")
    print("=" * 90)

    panel = build_panel(cfg)
    panel = add_period_and_volatility_regimes(panel, cfg)
    panel = add_change_series(panel)
    panel, filter_meta = _apply_runtime_filters(panel, maturity=maturity, period=period, leverage_group=leverage_group)

    if panel.empty:
        raise ValueError(
            "No rows remain after applying filters. "
            f"Selected maturity={filter_meta['selected_maturity']}, "
            f"period={filter_meta['selected_period']}, "
            f"leverage_group={filter_meta['selected_leverage_group']}."
        )

    print(f"Unified panel observations: {len(panel):,}")
    print(
        "Active filters: "
        f"maturity={filter_meta['selected_maturity']}; "
        f"period={filter_meta['selected_period']}; "
        f"leverage_group={filter_meta['selected_leverage_group']}"
    )
    panel.to_csv(out_dir / "model_comparison_panel.csv", index=False)
    pd.DataFrame([filter_meta]).to_csv(out_dir / "filter_metadata.csv", index=False)

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

    perf_tables = {
        "overall": (overall_perf, ["maturity"]),
        "by_leverage": (leverage_perf, ["leverage_group", "maturity"]),
        "by_year": (year_perf, ["year", "maturity"]),
        "by_period": (period_perf, ["period", "maturity"]),
        "by_volatility": (volatility_perf, ["vol_regime", "maturity"]),
        "by_period_volatility": (period_vol_perf, ["period", "vol_regime", "maturity"]),
        "by_company": (firm_perf, ["gvkey", "company", "maturity", "leverage_group"]),
    }

    print("Computing metric ranking tables (rank 1-4)...")
    rank_tables, rank_share_tables = compute_segment_rank_tables(perf_tables)
    for name, df in rank_tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    for name, df in rank_share_tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)

    manifest_rows = []
    for name, df in rank_tables.items():
        manifest_rows.append({"table_type": "rank", "name": name, "rows": int(len(df)), "file": f"{name}.csv"})
    for name, df in rank_share_tables.items():
        manifest_rows.append({"table_type": "rank_share", "name": name, "rows": int(len(df)), "file": f"{name}.csv"})
    pd.DataFrame(manifest_rows).to_csv(out_dir / "rank_tables_manifest.csv", index=False)

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
    create_plots(panel, best_tables, out_dir, rank_share_tables=rank_share_tables)

    print("Writing concise text summary...")
    with open(out_dir / "analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write("Model Performance Comparison Summary\n")
        f.write("=" * 44 + "\n\n")
        f.write(f"Total panel rows: {len(panel):,}\n")
        f.write(f"Firms: {panel['gvkey'].nunique()}\n")
        f.write(f"Years: {int(panel['year'].min())} - {int(panel['year'].max())}\n\n")
        f.write("Active Filters:\n")
        f.write(f"  - Maturity: {filter_meta['selected_maturity']}\n")
        f.write(f"  - Period: {filter_meta['selected_period']}\n")
        f.write(f"  - Leverage Group: {filter_meta['selected_leverage_group']}\n")
        f.write(
            "  - Rows before/after filter: "
            f"{filter_meta['rows_before_filter']}/{filter_meta['rows_after_filter']}\n\n"
        )

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
            f.write(f"\nWhere {MSGARCH_MODEL_LABEL} is strongest (top segments):\n")
            top = msgarch_best.head(12)
            for _, row in top.iterrows():
                f.write(
                    f"  - [{row['segment_type']}] {row['segment']}: "
                    f"win share={100 * row['share']:.1f}% (n={int(row['n_best'])})\n"
                )

        _write_rank_summary_by_maturity(
            f,
            rank_tables.get("rank_rmse_bps_overall", pd.DataFrame()),
            metric_col="rmse_bps",
            title="Average RMSE Ranking",
        )
        _write_rank_summary_by_maturity(
            f,
            rank_tables.get("rank_corr_levels_overall", pd.DataFrame()),
            metric_col="corr_levels",
            title="Correlation Levels Ranking",
        )
        _write_rank_summary_by_maturity(
            f,
            rank_tables.get("rank_corr_changes_overall", pd.DataFrame()),
            metric_col="corr_changes",
            title="Correlation Changes Ranking",
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
        "rank_manifest": out_dir / "rank_tables_manifest.csv",
        "rank_rmse_overall": out_dir / "rank_rmse_bps_overall.csv",
        "rank_corr_levels_overall": out_dir / "rank_corr_levels_overall.csv",
        "rank_corr_changes_overall": out_dir / "rank_corr_changes_overall.csv",
        "rank_rmse_share_by_period": out_dir / "rank_rmse_bps_by_period_share.csv",
        "rank_rmse_share_by_leverage": out_dir / "rank_rmse_bps_by_leverage_share.csv",
        "filters": out_dir / "filter_metadata.csv",
        "summary": out_dir / "analysis_summary.txt",
        "figures": out_dir / "figures",
    }

    print("Analysis complete.")
    print(f"Results saved in: {out_dir}")
    return outputs


if __name__ == "__main__":
    run_model_performance_paper()