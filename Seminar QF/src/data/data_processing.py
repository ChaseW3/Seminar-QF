import pandas as pd
import numpy as np
from scipy.special import ndtr
from joblib import Parallel, delayed
import time
from datetime import timedelta
try:
    from src.utils import config
except ImportError:
    from src.utils import config

FILENAME_EQUITY_DATA = config.EQUITY_DATA_FILE
FILENAME_INTEREST_RATES = config.INTEREST_RATES_FILE
SHEET_EQUITY = 0
SHEET_LIABILITY = 1
MIN_OBSERVATIONS = 252  # 1 year of trading days
T_HORIZON = 1.0
TRADING_DAYS_PER_YEAR = 252.0


def load_interest_rates():
    # Function to load interest rate data
    rates_df = pd.read_csv(FILENAME_INTEREST_RATES)
    
    # Pick whichever column in the ECB file contains EURIBOR
    rate_cols = [col for col in rates_df.columns if 'EURIBOR' in col.upper()]
    
    rates_df['DATE'] = pd.to_datetime(rates_df['DATE'])
    rates_df['month_year'] = rates_df['DATE'].dt.strftime('%Y-%m')
    rates_df['risk_free_rate'] = pd.to_numeric(rates_df[rate_cols[0]], errors='coerce') / 100  # ECB file stores rates as percentages
    
    # One rate per month is enough, drop duplicate daily entries
    return rates_df[['month_year', 'risk_free_rate']].drop_duplicates()


def load_and_preprocess_data():
    # Function to load and preprocess equity and liability data
    print("Loading equity data...")
    df = pd.read_excel(FILENAME_EQUITY_DATA, sheet_name=SHEET_EQUITY)
    
    # Rename columns
    df = df.rename(columns={
        "(fic) Current ISO Country Code - Incorporation": "country",
        "(isin) International Security Identification Number": "isin",
        "(datadate) Data Date - Daily Prices": "date",
        "(conm) Company Name": "company",
        "(gvkey) Global Company Key - Company": "gvkey",
        "(cshoc) Shares Outstanding": "shares_out",
        "(prccd) Price - Close - Daily": "close",
        "Market Capitalization (# Shares * Close Price)": "mkt_cap",
    })
    
    # Companies to remove
    gvkeys_to_remove = [
        101248, 25466, 203053, 245663, 340153, 243774, 17828, 333645,
        101305, 61214, 15181, 14140, 100312, 101276, 100737, 214881
    ]
    
    initial_firms = df['gvkey'].nunique()
    df = df[~df['gvkey'].isin(gvkeys_to_remove)]
    removed_firms = initial_firms - df['gvkey'].nunique()
    
    print(f"Removed {removed_firms} flagged companies (data quality issues)")
    print(f"Remaining firms: {df['gvkey'].nunique()}\n")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    
    # Filter out 2025 data, liabilities only available until 2024
    initial_rows = len(df)
    df = df[df["date"].dt.year <= 2024]
    removed_rows = initial_rows - len(df)
    print(f"Filtered out {removed_rows} rows from 2025 (keeping data up to and including 2024)")
    
    df = df.sort_values(["isin", "date"])
    
    # Fill missing prices and shares
    df[["shares_out", "close"]] = (
        df.groupby("isin")[["shares_out", "close"]]
          .ffill()
          .bfill()
    )
    df["mkt_cap"] = df["shares_out"] * df["close"]
    
    print("Loading liability data...")
    df2 = pd.read_excel(FILENAME_EQUITY_DATA, sheet_name=SHEET_LIABILITY)
    
    df2 = df2.rename(columns={
        "(gvkey) Global Company Key - Company": "gvkey",
        "(fyear) Data Year - Fiscal": "fyear",
        "(lt) Liabilities - Total": "liabilities_total",
        "(fdate) Final Date": "fdate",
        "(datadate) Data Date": "datadate" 
    })
    
    # Preprocess Liabilities
    df2["fdate"] = pd.to_datetime(df2["fdate"], errors="coerce")
    df2 = df2.dropna(subset=["fdate"]) # Drop rows without availability date, should be zero
    
    # Liablilities are in millions therefore calculate
    df2["liabilities_total"] = df2["liabilities_total"] * 1_000_000
    print(f"Scaled liabilities from millions to actual currency units (×1,000,000)")
    
    df = df.sort_values("date")
    df2 = df2.sort_values("fdate")
    
    # Merging of datasets
    print("Merging liabilities using Point-in-Time (fdate) logic (backward-looking)...")
    
    # merge_asof matches each equity date to the most recent fdate that has already passed,
    # so we never accidentally use liability data that wasn't public yet on that date
    df = pd.merge_asof(
        df, 
        df2[["gvkey", "fdate", "liabilities_total"]], 
        left_on="date", 
        right_on="fdate", 
        by="gvkey", 
        direction="backward"
    )
    
    df = df.sort_values(["gvkey", "date"])

    print("Loaded liability data")

    # Merge the interest rates
    print("Merging interest rates from ECB data...")
    try:
        rates_df = load_interest_rates()
        df['month_year'] = df['date'].dt.strftime('%Y-%m')
        
        # Merge rates (left join to keep all equity rows)
        df = pd.merge(df, rates_df, on='month_year', how='left')
        
        # Check for missing rates
        missing_count = df['risk_free_rate'].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} rows have missing interest rates. Filling with forward/backward fill...")
            df['risk_free_rate'] = df['risk_free_rate'].ffill().bfill().fillna(0.03) # Default 3% if all else fails
        
        df = df.drop(columns=['month_year'])
        print(f"Successfully merged interest rates. Range: {df['risk_free_rate'].min():.4f} to {df['risk_free_rate'].max():.4f}")
    except Exception as e:
        print(f"⚠ Error merging interest rates: {e}")
        # Ensure column exists even if merge failed to prevent errors later on
        if 'risk_free_rate' not in df.columns:
            df['risk_free_rate'] = 0.03
    
    return df


def merton_newton_raphson_vectorized(
    equity_value_series,
    debt_value_series,
    sigma_A_daily_initial,
    risk_free_rate_annual,
    time_to_maturity_years,
    max_iter=1000,
    tol=1e-4,
):
    # Iterative Merton solver: alternates between updating asset values (Newton-Raphson)
    # and re-estimating asset volatility from the implied asset return series
    
    # Seed asset value as equity + PV of debt
    debt_value_series = debt_value_series.astype(np.float64)
    asset_value_series = (
        equity_value_series.astype(np.float64)
        + debt_value_series * np.exp(-risk_free_rate_annual * time_to_maturity_years)
    )
    
    sigma_A_daily_current = np.float64(sigma_A_daily_initial)
    
    # Outer loop: iterate until sigma_A converges
    for k in range(max_iter):
        sigma_A_daily_old = sigma_A_daily_current
        
        # Inner loop: Newton-Raphson to back out asset values given current sigma_A
        for _ in range(20):
            asset_value_series = np.maximum(asset_value_series, 1e-4)
            sigma_A_annual_current = sigma_A_daily_current * np.sqrt(TRADING_DAYS_PER_YEAR)
            sig_sqrt_T = sigma_A_annual_current * np.sqrt(time_to_maturity_years)
            sig2_half = 0.5 * sigma_A_annual_current ** 2

            if sig_sqrt_T < 1e-10:
                sig_sqrt_T = 1e-10  # avoid division by zero for near-zero vol
            
            # Standard BSM d1 and d2
            d1 = (
                np.log(asset_value_series / debt_value_series)
                + (risk_free_rate_annual + sig2_half) * time_to_maturity_years
            ) / sig_sqrt_T
            d2 = d1 - sig_sqrt_T
            
            Nd1_arr = ndtr(d1)
            Nd2_arr = ndtr(d2)
            
            # Residual: BSM equity formula minus observed equity
            f_val = (
                asset_value_series * Nd1_arr
                - debt_value_series * np.exp(-risk_free_rate_annual * time_to_maturity_years) * Nd2_arr
                - equity_value_series
            )
            f_prime = Nd1_arr  # derivative of BSM equity w.r.t. asset value
            
            # Only step where the derivative is large enough to be safe
            # (skips entries where N(d1) ≈ 0, i.e. deep out-of-the-money)
            step = np.zeros_like(asset_value_series)
            safe_mask = f_prime > 1e-5
            step[safe_mask] = f_val[safe_mask] / f_prime[safe_mask]
            
            asset_value_series = asset_value_series - step
        
        asset_value_series = np.maximum(asset_value_series, 1e-4)
        
        # Re-estimate sigma_A from the implied asset return series
        log_asset_values = np.log(asset_value_series)
        asset_returns_daily = np.diff(log_asset_values)
        
        valid_mask = np.isfinite(asset_returns_daily)
        if np.sum(valid_mask) >= 10:
            sigma_A_daily_new = np.std(asset_returns_daily[valid_mask])
            
            # Stop if sigma_A stopped moving
            if abs(sigma_A_daily_new - sigma_A_daily_old) < tol:
                sigma_A_daily_current = sigma_A_daily_new
                break
            
            sigma_A_daily_current = sigma_A_daily_new
    
    return asset_value_series, sigma_A_daily_current


def process_firm_merton(firm_data, interest_rates_dict, firm_idx, total_firms):
    # Runs the rolling Merton estimation for a single firm.
    # Called in parallel for each firm by run_merton_estimation.
    gvkey = firm_data["gvkey"].iloc[0]
    firm_data = firm_data.sort_values("date").reset_index(drop=True)
    all_dates = firm_data["date"].unique()
    
    results = []
    
    for date_idx, date_t in enumerate(all_dates):
        # Rolling 252-day window ending at the current date
        window_start_idx = max(0, date_idx - MIN_OBSERVATIONS + 1)
        window_df = firm_data.iloc[window_start_idx:date_idx + 1]
        
        if len(window_df) < MIN_OBSERVATIONS:
            continue
        
        # Prefer the rate already merged into the DataFrame, fall back to the dict
        if 'risk_free_rate' in window_df.columns:
            r_vals = window_df.loc[window_df['date'] == date_t, 'risk_free_rate'].values
            if len(r_vals) > 0:
                r_annual = float(r_vals[0])
            else:
                month_str = pd.Timestamp(date_t).strftime('%Y-%m')
                r_annual = interest_rates_dict.get(month_str, 0.05)
        else:
            month_str = pd.Timestamp(date_t).strftime('%Y-%m')
            r_annual = interest_rates_dict.get(month_str, 0.05)
        
        # Equity time series over the window
        equity_value_series = window_df["mkt_cap"].values.astype(np.float64)

        # Use the actual historical debt for each day in the window
        debt_value_series = window_df["liabilities_total"].values.astype(np.float64)
        
        # Skip windows that pre-date the first liability observation
        if np.any(np.isnan(debt_value_series)) or np.any(debt_value_series <= 0):
            continue
        
        # Basic sanity check, need positive equity values to run BSM
        if (
            len(equity_value_series) < 10
            or np.any(equity_value_series <= 0)
        ):
            continue
        
        # Compute initial equity volatility from log-returns over the window
        equity_values_safe = np.maximum(equity_value_series, 1e-4)
        equity_returns_daily = np.diff(np.log(equity_values_safe))
        equity_returns_daily = equity_returns_daily[np.isfinite(equity_returns_daily)]
        
        if len(equity_returns_daily) < 10:
            continue
        
        sigma_E_daily = np.std(equity_returns_daily)
        if sigma_E_daily < 1e-6:
            sigma_E_daily = 0.4 / np.sqrt(252)  # fallback for near-zero vol
        
        time_to_maturity_years = 1.0
        try:
            asset_value_series, sigma_A_daily = merton_newton_raphson_vectorized(
                equity_value_series,
                debt_value_series,
                sigma_E_daily,
                r_annual,
                time_to_maturity_years,
                max_iter=1000,
                tol=1e-4,
            )
        except Exception as e:
            print(f"    ⚠ Error in Merton for gvkey={gvkey}, date={date_t}: {e}")
            continue
        
        # Only keep the last point in the window as the estimate for today
        asset_value_final = asset_value_series[-1]
        
        results.append({
            "gvkey": gvkey,
            "date": date_t,
            "asset_value": asset_value_final,
            "asset_volatility": sigma_A_daily,
        })
        
        if (date_idx + 1) % 100 == 0:
            print(f"  Firm {firm_idx+1}/{total_firms} (gvkey={gvkey}): {date_idx+1}/{len(all_dates)} dates")
    
    return results


def run_merton_estimation(df, interest_rates_df=None, n_jobs=-1, use_cache=False):
    overall_start = time.time()
    
    print(f"\n{'='*80}")
    print("MERTON MODEL ESTIMATION (Vectorized + Parallelized - EXACT ndtr)")
    print(f"{'='*80}\n")
    
    # Only drop rows with missing equity — liabilities can be NaN in early dates
    # and are checked per-window inside process_firm_merton
    print("DEBUG: Date range BEFORE filtering:")
    print(f"  Full df: {df['date'].min()} to {df['date'].max()} ({len(df)} rows, {df['gvkey'].nunique()} firms)")
    
    solver_df = df.dropna(subset=["mkt_cap"]).copy()  # Only require equity data
    solver_df = solver_df.sort_values(["gvkey", "date"])
    
    print("DEBUG: Date range AFTER filtering (equity only):")
    print(f"  solver_df: {solver_df['date'].min()} to {solver_df['date'].max()} ({len(solver_df)} rows, {solver_df['gvkey'].nunique()} firms)")
    
    if solver_df.empty:
        raise ValueError("No valid data found for Merton model estimation.")
    
    firms = sorted(solver_df["gvkey"].unique())
    print(f"Processing {len(firms)} firms with {n_jobs} parallel jobs...\n")
    
    # Dict lookup is much faster than DataFrame slicing inside the tight per-date loop
    if interest_rates_df is not None:
        interest_rates_dict = dict(zip(interest_rates_df['month_year'], interest_rates_df['risk_free_rate']))
    else:
        interest_rates_dict = {}
    
    print(f"Starting parallel Merton estimation...")
    start_parallel = time.time()
    
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_firm_merton)(
            solver_df[solver_df["gvkey"] == gvkey],
            interest_rates_dict,
            i,
            len(firms)
        )
        for i, gvkey in enumerate(firms)
    )
    
    # Each element in results_list is a list of dicts, change into one list
    merton_results = pd.DataFrame([item for sublist in results_list for item in sublist])
    
    if merton_results.empty:
        print("✗ No Merton results generated!")
        return pd.DataFrame(), pd.DataFrame()
    
    parallel_time = time.time() - start_parallel
    print(f"\n✓ Parallel Merton complete in {timedelta(seconds=int(parallel_time))}\n")
    
    print("DEBUG: Merton results date range:")
    print(f"  merton_results: {merton_results['date'].min()} to {merton_results['date'].max()} ({len(merton_results)} rows)")
    
    # Left join to keep all original rows even where Merton couldn't produce an estimate
    df_merged = pd.merge(df, merton_results, on=["gvkey", "date"], how="left", suffixes=("", "_merton"))
    
    print("DEBUG: After merging Merton back to original df:")
    print(f"  df_merged: {df_merged['date'].min()} to {df_merged['date'].max()} ({len(df_merged)} rows)")
    
    first_merton_date = df_merged.dropna(subset=['asset_value'])['date'].min()
    print(f"  First non-null Merton result: {first_merton_date}")
    
    first_liab_date = df_merged.dropna(subset=['liabilities_total'])['date'].min()
    print(f"  First non-null liabilities: {first_liab_date}")
    
    # Compute log-returns across consecutive trading days per firm
    daily_returns_df = merton_results.copy().sort_values(["gvkey", "date"])
    daily_returns_df["asset_return_daily"] = daily_returns_df.groupby("gvkey")["asset_value"].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Flag any suspiciously large single-day moves for review
    print("\n" + "="*60)
    print("EXTREME ASSET RETURN DIAGNOSTICS")
    print("="*60)
    
    diag_df = daily_returns_df.dropna(subset=['asset_return_daily'])
    
    for threshold in [0.30, 0.40, 0.50]:
        extreme_rows = diag_df[diag_df['asset_return_daily'].abs() > threshold]
        
        print(f"\n[Threshold: Absolute Return > {threshold*100:.0f}%]")
        if extreme_rows.empty:
            print("  No extreme returns found at this level.")
        else:
            print(f"  Found {len(extreme_rows)} instances:")
            for gvkey, group in extreme_rows.groupby('gvkey'):
                print(f"  • Firm {gvkey}:")
                for _, row in group.iterrows():
                    ret_val = row['asset_return_daily']
                    date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
                    print(f"      {date_str}: {ret_val*100:+.2f}%")
    
    print("="*60 + "\n")

    # Scale returns by 100 for numerical stability in downstream optimizers
    daily_returns_df["asset_return_daily_scaled"] = daily_returns_df["asset_return_daily"] * 100.0
    
    daily_returns_df = daily_returns_df[[
        "gvkey", "date", "asset_return_daily", "asset_return_daily_scaled", "asset_value", "asset_volatility"
    ]].dropna()
    
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("DATE RANGE SANITY CHECKS")
    print(f"{'='*80}")
    
    equity_min = df['date'].min()
    merged_min = df_merged['date'].min()
    print(f"✓ Original equity data starts: {equity_min}")
    print(f"✓ Merged data (equity + Merton) starts: {merged_min}")
    assert equity_min == merged_min, f"ERROR: Lost early dates! Equity starts {equity_min}, merged starts {merged_min}"
    
    # Merton needs a full 252-day window, so first result is roughly 1 year after data start
    first_merton = df_merged.dropna(subset=['asset_value'])['date'].min()
    expected_merton_start = equity_min + pd.DateOffset(days=252)
    print(f"✓ First non-null Merton result: {first_merton}")
    print(f"  (Expected around {expected_merton_start.strftime('%Y-%m-%d')} after 252-day (1-year) window)")
    
    # Liabilities start later than equity data — this is expected
    first_liab = df_merged.dropna(subset=['liabilities_total'])['date'].min()
    print(f"✓ First non-null liabilities: {first_liab}")
    print(f"  (This is expected - liabilities data starts later than equity)")
    
    complete_data = df_merged.dropna(subset=['asset_value', 'liabilities_total'])
    if not complete_data.empty:
        complete_min = complete_data['date'].min()
        print(f"✓ Complete data (Merton + liabilities): {complete_min}")
        print(f"  ({len(complete_data)} rows with both Merton and liability data)")
    else:
        print(f"⚠ No rows with both Merton and liability data!")
    
    print(f"{'='*80}\n")
    
    print(f"{'='*80}")
    print(f"Merton Estimation Complete")
    print(f"{'='*80}")
    print(f"Total time: {timedelta(seconds=int(overall_time))}")
    print(f"Firms processed: {len(firms)}")
    print(f"Daily results: {len(daily_returns_df):,}")
    print(f"{'='*80}\n")
    
    return df_merged, daily_returns_df

