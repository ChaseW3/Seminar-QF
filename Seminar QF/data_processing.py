# data_processing.py (OPTIMIZED VERSION)

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import ndtr
from joblib import Parallel, delayed
import os
import pickle
import time
from datetime import timedelta

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
FILENAME_EQUITY_DATA = "Jan2025_Accenture_Dataset_ErasmusCase.xlsx"
FILENAME_INTEREST_RATES = "ECB Data Portal_20260125170805.csv"
SHEET_EQUITY = 0
SHEET_LIABILITY = 1
MIN_OBSERVATIONS = 150
T_HORIZON = 1.0
CACHE_DIR = './intermediates/'

os.makedirs(CACHE_DIR, exist_ok=True)


def load_interest_rates():
    """
    Load monthly interest rates from ECB data.
    Returns a dataframe with month_year and risk_free_rate columns.
    """
    rates_df = pd.read_csv(FILENAME_INTEREST_RATES)
    
    rate_cols = [col for col in rates_df.columns if 'EURIBOR' in col.upper()]
    
    rates_df['DATE'] = pd.to_datetime(rates_df['DATE'])
    rates_df['month_year'] = rates_df['DATE'].dt.strftime('%Y-%m')
    rates_df['risk_free_rate'] = pd.to_numeric(rates_df[rate_cols[0]], errors='coerce') / 100
    
    return rates_df[['month_year', 'risk_free_rate']].drop_duplicates()


def load_and_preprocess_data():
    """Reads Excel data, cleanses it, merges equity and liabilities."""
    print("Loading equity data...")
    df = pd.read_excel(FILENAME_EQUITY_DATA, sheet_name=SHEET_EQUITY)
    
    # RENAME COLUMNS FIRST so that 'gvkey' exists
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
    
    # NOW REMOVE FLAGGED COMPANIES (after column renaming)
    gvkeys_to_remove = [
        101248, 25466, 203053, 245663, 19349, 243774, 17828, 333645,
        101305, 61214, 15181, 14140, 100312, 101276, 100737, 214881
    ]
    
    initial_firms = df['gvkey'].nunique()
    df = df[~df['gvkey'].isin(gvkeys_to_remove)]
    removed_firms = initial_firms - df['gvkey'].nunique()
    
    print(f"Removed {removed_firms} flagged companies (gvkeys: {gvkeys_to_remove})")
    print(f"Remaining firms: {df['gvkey'].nunique()}\n")
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.sort_values(["isin", "date"])
    
    # Fill missing prices/shares
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
        "(datadate) Data Date": "date", 
    })
    
    # Merge strategy: Match by Fiscal Year
    df["fyear"] = df["date"].dt.year
    df2_subset = df2[["gvkey", "fyear", "liabilities_total"]].drop_duplicates(subset=["gvkey", "fyear"])
    
    df = pd.merge(df, df2_subset, on=["gvkey", "fyear"], how="left")

    print("Loaded liability data")
    
    return df


def merton_newton_raphson_vectorized(E_vec, B_vec, sigma_A, r, T_val, max_iter=100, tol=1e-4):
    """
    Vectorized Newton-Raphson solver for Merton model (EXACT - NO APPROXIMATION).
    
    Uses scipy.special.ndtr for EXACT normal CDF (not approximation).
    NO Numba JIT - uses pure NumPy for vectorization.
    
    Parameters:
    -----------
    E_vec : array, Equity values
    B_vec : array, Liability values
    sigma_A : float, Initial asset volatility (daily)
    r : float, Risk-free rate
    T_val : float, Time horizon (252 for 1 year daily)
    max_iter : int, Max iterations for Newton-Raphson
    tol : float, Convergence tolerance
    
    Returns:
    --------
    V_A_vec : array, Final asset values
    sigma_A_current : float, Final asset volatility (daily)
    """
    
    V_A_vec = E_vec.astype(np.float64) + B_vec.astype(np.float64)  # Initial guess
    sigma_A_current = np.float64(sigma_A)
    
    for k in range(max_iter):
        sigma_A_old = sigma_A_current
        
        # Inner Loop: Newton-Raphson for V_A (vectorized)
        for _ in range(5):
            V_A_vec = np.maximum(V_A_vec, 1e-4)
            sig_sqrt_T = sigma_A_current * np.sqrt(T_val)
            sig2_half = 0.5 * sigma_A_current ** 2
            
            d1 = (np.log(V_A_vec / B_vec) + (r + sig2_half) * T_val) / sig_sqrt_T
            d2 = d1 - sig_sqrt_T
            
            # EXACT: Use ndtr (scipy's C-optimized normal CDF)
            Nd1_arr = ndtr(d1)
            Nd2_arr = ndtr(d2)
            
            # Black-Scholes formula for equity value
            f_val = V_A_vec * Nd1_arr - B_vec * np.exp(-r * T_val) * Nd2_arr - E_vec
            f_prime = Nd1_arr  # N(d1)
            
            # Newton-Raphson step (vectorized, safe division)
            step = np.zeros_like(V_A_vec)
            safe_mask = f_prime > 1e-5
            step[safe_mask] = f_val[safe_mask] / f_prime[safe_mask]
            
            V_A_vec = V_A_vec - step
        
        V_A_vec = np.maximum(V_A_vec, 1e-4)
        
        # Outer Update: sigma_A from asset returns (daily)
        log_V_A = np.log(V_A_vec)
        ret_A = np.diff(log_V_A)
        
        # Filter valid returns
        valid_mask = np.isfinite(ret_A)
        if np.sum(valid_mask) >= 10:
            # Calculate annualized volatility from daily returns (assuming 252 trading days)
            sigma_A_new = np.std(ret_A[valid_mask]) * np.sqrt(252)
            
            # Check convergence
            if abs(sigma_A_new - sigma_A_old) < tol:
                sigma_A_current = sigma_A_new
                break
            
            sigma_A_current = sigma_A_new
    
    return V_A_vec, sigma_A_current


def process_firm_merton(firm_data, interest_rates_dict, firm_idx, total_firms):
    """
    Process single firm's Merton estimation (for parallelization).
    
    Returns list of daily results for this firm.
    """
    gvkey = firm_data["gvkey"].iloc[0]
    firm_data = firm_data.sort_values("date").reset_index(drop=True)
    all_dates = firm_data["date"].unique()
    
    results = []
    
    for date_idx, date_t in enumerate(all_dates):
        # 252-day rolling window
        start_date = date_t - pd.DateOffset(days=252)
        
        window_mask = (firm_data["date"] > start_date) & (firm_data["date"] <= date_t)
        window_df = firm_data.loc[window_mask]
        
        if len(window_df) < MIN_OBSERVATIONS:
            continue
        
        # Get interest rate
        month_str = pd.Timestamp(date_t).strftime('%Y-%m')
        r_annual = interest_rates_dict.get(month_str, 0.05)
        
        # Inputs
        E_vec = window_df["mkt_cap"].values.astype(np.float64)
        B_vec = window_df["liabilities_total"].values.astype(np.float64) * 1_000_000
        
        # Check validity
        if len(E_vec) < 10 or np.any(E_vec <= 0) or np.any(B_vec <= 0):
            continue
        
        # Initial sigma_E (daily)
        E_safe = np.maximum(E_vec, 1e-4)
        ret_E = np.diff(np.log(E_safe))
        ret_E = ret_E[np.isfinite(ret_E)]
        
        if len(ret_E) < 10:
            continue
        
        sigma_E_daily = np.std(ret_E)
        if sigma_E_daily < 1e-6:
            sigma_E_daily = 0.4 / np.sqrt(252)  # Daily equivalent fallback
            
        sigma_E_annual = sigma_E_daily * np.sqrt(252)
        
        # CALL VECTORIZED FUNCTION (NO NUMBA)
        # Use T = 365/360 consistent with ACT/360 money market convention for rates
        T_val = 1
        try:
            V_A_vec, sigma_A = merton_newton_raphson_vectorized(
                E_vec, B_vec, sigma_E_annual, r_annual, T_val, max_iter=100, tol=1e-4
            )
        except Exception as e:
            print(f"    ⚠ Error in Merton for gvkey={gvkey}, date={date_t}: {e}")
            continue
        
        # Save result
        V_A_final = V_A_vec[-1]
        sigma_A_annualized = sigma_A  # Already annualized
        
        results.append({
            "gvkey": gvkey,
            "date": date_t,
            "asset_value": V_A_final,
            "asset_volatility": sigma_A_annualized,
        })
        
        # Progress
        if (date_idx + 1) % 100 == 0:
            print(f"  Firm {firm_idx+1}/{total_firms} (gvkey={gvkey}): {date_idx+1}/{len(all_dates)} dates")
    
    return results


def run_merton_estimation(df, interest_rates_df=None, n_jobs=-1, use_cache=True):
    """
    Runs the rolling window iterative Merton model on DAILY data (OPTIMIZED).
    
    Uses:
    - Vectorization with NumPy for speed
    - Parallelization with joblib over firms
    - Caching of results
    - EXACT scipy.special.ndtr (no approximation)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with equity and liability information
    interest_rates_df : pd.DataFrame, optional
        DataFrame with columns 'month_year' and 'risk_free_rate'
    n_jobs : int
        Number of parallel jobs. -1 = use all cores
    use_cache : bool
        Whether to use cached results if available
    
    Returns:
    --------
    df_merged : DataFrame with Merton results merged
    daily_returns_df : DataFrame with daily returns and volatilities
    """
    
    overall_start = time.time()
    
    print(f"\n{'='*80}")
    print("MERTON MODEL ESTIMATION (Vectorized + Parallelized - EXACT ndtr)")
    print(f"{'='*80}\n")
    
    # Check cache
    cache_file = os.path.join(CACHE_DIR, 'merton_results_cache.pkl')
    if use_cache and os.path.exists(cache_file):
        print("Loading cached Merton results...")
        try:
            merton_results = pd.read_pickle(cache_file)
            print(f"✓ Loaded {len(merton_results):,} cached results\n")
            df_merged = pd.merge(df, merton_results, on=["gvkey", "date"], how="left", suffixes=("", "_merton"))
            
            daily_returns_df = merton_results.copy().sort_values(["gvkey", "date"])
            daily_returns_df["asset_return_daily"] = daily_returns_df.groupby("gvkey")["asset_value"].transform(
                lambda x: np.log(x / x.shift(1))
            )
            daily_returns_df["asset_volatility"] = daily_returns_df["asset_volatility"] / np.sqrt(252)
            daily_returns_df = daily_returns_df[["gvkey", "date", "asset_return_daily", "asset_value", "asset_volatility"]].dropna()
            
            return df_merged, daily_returns_df
        except Exception as e:
            print(f"⚠ Cache load failed ({e}), recomputing...\n")
    
    # Filter valid data
    solver_df = df.dropna(subset=["mkt_cap", "liabilities_total"]).copy()
    solver_df = solver_df.sort_values(["gvkey", "date"])
    
    if solver_df.empty:
        raise ValueError("No valid data found for Merton model estimation.")
    
    firms = sorted(solver_df["gvkey"].unique())
    print(f"Processing {len(firms)} firms with {n_jobs} parallel jobs...\n")
    
    # Convert interest rates to dict for faster lookup
    if interest_rates_df is not None:
        interest_rates_dict = dict(zip(interest_rates_df['month_year'], interest_rates_df['risk_free_rate']))
    else:
        interest_rates_dict = {}
    
    # PARALLELIZE over firms
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
    
    # Flatten results
    merton_results = pd.DataFrame([item for sublist in results_list for item in sublist])
    
    if merton_results.empty:
        print("✗ No Merton results generated!")
        return pd.DataFrame(), pd.DataFrame()
    
    parallel_time = time.time() - start_parallel
    print(f"\n✓ Parallel Merton complete in {timedelta(seconds=int(parallel_time))}\n")
    
    # Cache results
    print(f"Caching {len(merton_results):,} Merton results...")
    merton_results.to_pickle(cache_file)
    print(f"✓ Cached to: {cache_file}\n")
    
    # Merge back to main DF
    df_merged = pd.merge(df, merton_results, on=["gvkey", "date"], how="left", suffixes=("", "_merton"))
    
    # Compute Daily Returns DataFrame
    daily_returns_df = merton_results.copy().sort_values(["gvkey", "date"])
    daily_returns_df["asset_return_daily"] = daily_returns_df.groupby("gvkey")["asset_value"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    
    # CRITICAL: Convert annualized volatility to DAILY for scale matching
    daily_returns_df["asset_volatility"] = daily_returns_df["asset_volatility"] / np.sqrt(252)
    
    daily_returns_df = daily_returns_df[[
        "gvkey", "date", "asset_return_daily", "asset_value", "asset_volatility"
    ]].dropna()
    
    overall_time = time.time() - overall_start
    
    print(f"{'='*80}")
    print(f"Merton Estimation Complete")
    print(f"{'='*80}")
    print(f"Total time: {timedelta(seconds=int(overall_time))}")
    print(f"Firms processed: {len(firms)}")
    print(f"Daily results: {len(daily_returns_df):,}")
    print(f"{'='*80}\n")
    
    return df_merged, daily_returns_df

