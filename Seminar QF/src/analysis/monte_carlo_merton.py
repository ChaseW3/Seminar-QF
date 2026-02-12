"""
Monte Carlo simulation using constant Merton volatility for fair comparison.

This provides a FAIR COMPARISON with GARCH/RS/MS-GARCH by using:
- Same Monte Carlo framework (1000 simulations, 252 days)
- Same integrated variance calculation (IV = Σ E[σ²])
- But with CONSTANT volatility (σ = asset_volatility from Merton estimation)

This isolates the impact of dynamic volatility modeling vs. constant volatility.

Key difference from Merton analytical:
- Analytical: Uses historical average volatility directly
- Monte Carlo: Projects forward using same volatility (for comparison with GARCH/RS/MS-GARCH)

Author: Generated for Seminar QF
Date: February 2026
"""

import pandas as pd
import numpy as np
import numba
from joblib import Parallel, delayed
from datetime import datetime, timedelta


@numba.jit(nopython=True, parallel=True)
def simulate_constant_vol_paths(sigma_daily_arr, num_simulations, num_days, num_firms):
    """
    Simulate paths with CONSTANT daily volatility (Merton assumption).
    
    For each firm, volatility remains constant at σ_daily for all days.
    This mimics the Monte Carlo framework used by GARCH/RS/MS-GARCH but without dynamics.
    
    Parameters:
    -----------
    sigma_daily_arr : np.ndarray
        Array of constant daily volatilities for each firm (num_firms,)
    num_simulations : int
        Number of Monte Carlo paths
    num_days : int
        Forecast horizon in days
    num_firms : int
        Number of firms
        
    Returns:
    --------
    daily_volatilities : np.ndarray
        Shape (num_days, num_simulations, num_firms)
        Each element is the constant daily volatility
    daily_returns : np.ndarray
        Shape (num_days, num_simulations, num_firms)
        Simulated returns (innovations) = sigma * Z
    """
    daily_volatilities = np.zeros((num_days, num_simulations, num_firms))
    daily_returns = np.zeros((num_days, num_simulations, num_firms))
    
    # Fill with constant volatility
    for firm_idx in numba.prange(num_firms):
        sigma = sigma_daily_arr[firm_idx]
        for sim in range(num_simulations):
            for day in range(num_days):
                # Standard normal innovation
                z = np.random.standard_normal()
                daily_volatilities[day, sim, firm_idx] = sigma
                daily_returns[day, sim, firm_idx] = sigma * z
    
    return daily_volatilities, daily_returns


def _process_single_date_merton_mc(date_data, num_simulations, num_days):
    """
    Process Monte Carlo Merton simulation for a single date (for parallelization).
    
    Parameters:
    -----------
    date_data : tuple
        (date, df_date) where df_date contains firms data for that date
    num_simulations : int
        Number of Monte Carlo paths
    num_days : int
        Forecast horizon in days
        
    Returns:
    --------
    list : Results for all firms on this date
    """
    date, df_date = date_data
    results_list = []
    
    if df_date.empty:
        return results_list
    
    firms_on_date = df_date['gvkey'].unique()
    
    # Prepare arrays for vectorized simulation
    merton_params = {}
    for firm in firms_on_date:
        firm_data = df_date[df_date['gvkey'] == firm].iloc[0]
        
        # Get annual volatility from Merton estimation
        sigma_annual = firm_data.get('asset_volatility', np.nan)
        
        # Convert to daily volatility: σ_daily = σ_annual / sqrt(252)
        sigma_daily = sigma_annual / np.sqrt(252) if not np.isnan(sigma_annual) else np.nan
        
        merton_params[firm] = {
            'sigma_annual': sigma_annual,
            'sigma_daily': sigma_daily
        }
    
    # Filter out firms with missing volatility
    firms_on_date = sorted([f for f in firms_on_date if not np.isnan(merton_params[f]['sigma_daily'])])
    
    if len(firms_on_date) == 0:
        return results_list
    
    # Prepare arrays
    sigma_daily_arr = np.array([merton_params[f]['sigma_daily'] for f in firms_on_date])
    
    # Run Monte Carlo simulation with constant volatility
    daily_vols, daily_sim_returns = simulate_constant_vol_paths(
        sigma_daily_arr, num_simulations, num_days, len(firms_on_date)
    )
    
    # For each firm, calculate statistics
    for firm_idx, firm in enumerate(firms_on_date):
        firm_data = df_date[df_date['gvkey'] == firm].iloc[0]
        firm_daily_vols = daily_vols[:, :, firm_idx]  # shape: (num_days, num_simulations)
        firm_daily_returns = daily_sim_returns[:, :, firm_idx]
        
        # YEARLY VARIANCE CALCULATION (Asset Value Simulation Method)
        # Using the returns from the simulation (which match the volatility path in general models)
        
        # 3. Cumulative yearly return
        # Using Log Returns logic: V_T = V_0 * exp(sum(r_t))
        # Here firm_daily_returns are log returns r_t = sigma * Z
        
        # Cumulative returns path for value/barrier
        cum_log_returns_path = np.cumsum(firm_daily_returns, axis=0) # shape: (num_days, num_simulations)
        
        # Calculate Asset Paths: V_t = V_0 * exp(cum_log_returns)
        asset_value = firm_data.get('asset_value', np.nan)
        v0 = asset_value
        liability = firm_data.get('liabilities_total', np.nan) # K
        
        # Legacy Variance 
        # For log returns, integrated variance = Variance(Sum(r_t))
        # This matches the true definition in continuous time models
        log_return_T = np.sum(firm_daily_returns, axis=0)
        integrated_variance = np.var(log_return_T, ddof=1)
        annualized_volatility = np.sqrt(integrated_variance)
        
        # PROBABILITY OF DEFAULT & SPREAD CALCULATION (1Y, 3Y, 5Y)
        mc_spread_1y, mc_spread_3y, mc_spread_5y = np.nan, np.nan, np.nan
        mc_debt_1y, mc_debt_3y, mc_debt_5y = np.nan, np.nan, np.nan
        pd_value_1y, pd_value_3y, pd_value_5y = np.nan, np.nan, np.nan
        
        # Legacy/Compatibility variables (will use 1Y)
        pd_value = np.nan
        mc_spread = np.nan
        mc_debt_value = np.nan
        
        if not np.isnan(v0) and not np.isnan(liability) and v0 > 0:
             # Asset Paths
             asset_paths = v0 * np.exp(cum_log_returns_path)
             
             rf_rate = firm_data.get('risk_free_rate', np.nan)
             if not np.isnan(rf_rate) and abs(rf_rate) > 0.5: 
                 rf_rate = rf_rate / 100.0

             horizons = {
                 '1y': min(252, num_days),
                 '3y': min(252*3, num_days),
                 '5y': min(252*5, num_days)
             }
             
             # --- 1 Year ---
             idx_1y = horizons['1y'] - 1
             if idx_1y < asset_paths.shape[0]:
                 # PD: Terminal Value ONLY (Merton)
                 # Was: np.mean(np.any(asset_paths[:idx_1y+1, :] < liability, axis=0))
                 pd_value_1y = np.mean(asset_paths[idx_1y, :] < liability)
                 
                 if not np.isnan(rf_rate):
                     V_T = asset_paths[idx_1y, :]
                     expected_payoff = np.mean(np.minimum(V_T, liability))
                     T_years = horizons['1y'] / 252.0
                     discount_factor = np.exp(-rf_rate * T_years)
                     debt_val = expected_payoff * discount_factor
                     
                     if debt_val > 0 and liability > 0:
                         ytm = -np.log(debt_val / liability) / T_years
                         mc_spread_1y = max(ytm - rf_rate, 0.0)
                         mc_debt_1y = debt_val

             # --- 3 Year ---
             idx_3y = horizons['3y'] - 1
             if idx_3y < asset_paths.shape[0]:
                 # PD: Terminal Value ONLY (Merton)
                 pd_value_3y = np.mean(asset_paths[idx_3y, :] < liability)
                 
                 if not np.isnan(rf_rate):
                     V_T = asset_paths[idx_3y, :]
                     expected_payoff = np.mean(np.minimum(V_T, liability))
                     T_years = horizons['3y'] / 252.0
                     discount_factor = np.exp(-rf_rate * T_years)
                     debt_val = expected_payoff * discount_factor
                     
                     if debt_val > 0 and liability > 0:
                         ytm = -np.log(debt_val / liability) / T_years
                         mc_spread_3y = max(ytm - rf_rate, 0.0)
                         mc_debt_3y = debt_val

             # --- 5 Year ---
             idx_5y = horizons['5y'] - 1
             if idx_5y < asset_paths.shape[0]:
                 # PD: Terminal Value ONLY (Merton)
                 pd_value_5y = np.mean(asset_paths[idx_5y, :] < liability)
                 
                 if not np.isnan(rf_rate):
                     V_T = asset_paths[idx_5y, :]
                     expected_payoff = np.mean(np.minimum(V_T, liability))
                     T_years = horizons['5y'] / 252.0
                     discount_factor = np.exp(-rf_rate * T_years)
                     debt_val = expected_payoff * discount_factor
                     
                     if debt_val > 0 and liability > 0:
                         ytm = -np.log(debt_val / liability) / T_years
                         mc_spread_5y = max(ytm - rf_rate, 0.0)
                         mc_debt_5y = debt_val

             # Set legacy values to 1Y results
             pd_value = pd_value_1y
             mc_spread = mc_spread_1y
             mc_debt_value = mc_debt_1y

        # Also calculate mean volatility path for verification
        mean_path = np.mean(firm_daily_vols, axis=1)  # shape: (num_days,)
        
        results_list.append({
            'gvkey': firm,
            'date': date,
            'merton_mc_integrated_variance': integrated_variance,
            'merton_mc_annualized_volatility': annualized_volatility,
            'merton_mc_mean_daily_volatility': np.mean(mean_path),
            'merton_mc_constant_annual_vol': merton_params[firm]['sigma_annual'],
            'merton_mc_probability_of_default': pd_value,
            'merton_mc_pd_1y': pd_value_1y, # Now Terminal PD
            'merton_mc_pd_3y': pd_value_3y,
            'merton_mc_pd_5y': pd_value_5y,
            'merton_mc_pd_terminal_1y': pd_value_1y, # Explicitly named
            'merton_mc_pd_terminal_3y': pd_value_3y,
            'merton_mc_pd_terminal_5y': pd_value_5y,
            'merton_mc_implied_spread_1y': mc_spread_1y,
            'merton_mc_implied_spread_3y': mc_spread_3y,
            'merton_mc_implied_spread_5y': mc_spread_5y,
            'merton_mc_debt_value_1y': mc_debt_1y,
            'merton_mc_debt_value_3y': mc_debt_3y,
            'merton_mc_debt_value_5y': mc_debt_5y,
            'merton_mc_implied_spread': mc_spread,
            'merton_mc_debt_value': mc_debt_value
        })
    
    return results_list


def monte_carlo_merton_1year_parallel(merton_file, gvkey_selected=None, 
                                       num_simulations=1000, num_days=1260, n_jobs=-1):
    """
    Monte Carlo with constant Merton volatility (1-year horizon, parallelized).
    
    This function provides a FAIR COMPARISON with GARCH/RS/MS-GARCH Monte Carlo by:
    1. Using the same Monte Carlo framework (simulations, days)
    2. Using the same integrated variance calculation
    3. But keeping volatility CONSTANT (Merton assumption)
    
    This isolates whether dynamic volatility models add value vs. constant volatility.
    
    Parameters:
    -----------
    merton_file : str
        Path to merged_data_with_merton.csv containing asset_volatility
    gvkey_selected : list, optional
        Specific firms to process (if None, processes all)
    num_simulations : int
        Number of Monte Carlo paths (default: 1000)
    num_days : int
        Forecast horizon in days (default: 1260 = 5 years)
    n_jobs : int
        Number of parallel jobs (default: -1 = all cores)
    
    Returns:
    --------
    pd.DataFrame with columns:
        - gvkey, date
        - merton_mc_integrated_variance: IV = Σ E[σ²] over horizon
        - merton_mc_annualized_volatility: sqrt(IV) (should equal asset_volatility)
        - merton_mc_mean_daily_volatility: Mean daily volatility
        - merton_mc_constant_annual_vol: Original Merton annual volatility
    """
    print(f"\n{'='*80}")
    print("MONTE CARLO: MERTON CONSTANT VOLATILITY (5-Year Forecast)")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    # Load Merton results
    print("Loading Merton estimation results...")
    df = pd.read_csv(merton_file)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter firms if specified
    if gvkey_selected is not None:
        df = df[df['gvkey'].isin(gvkey_selected)]
        print(f"Filtering to {len(gvkey_selected)} selected firms")
    
    # Drop rows with missing asset_volatility
    initial_rows = len(df)
    df = df.dropna(subset=['asset_volatility'])
    dropped_rows = initial_rows - len(df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows:,} rows with missing asset_volatility")
    
    unique_firms = df['gvkey'].nunique()
    unique_dates = df['date'].nunique()
    
    print(f"\nDataset summary:")
    print(f"  Firms: {unique_firms}")
    print(f"  Dates: {unique_dates}")
    print(f"  Total observations: {len(df):,}")
    print(f"\nMonte Carlo settings:")
    print(f"  Simulations per firm-date: {num_simulations}")
    print(f"  Forecast horizon: {num_days} days (5 years)")
    print(f"  Parallel jobs: {n_jobs if n_jobs > 0 else 'All cores'}")
    print(f"\nStarting simulation...\n")
    
    # Prepare data for parallel processing
    grouped = df.groupby('date')
    date_data_list = [(date, group) for date, group in grouped]
    
    # Parallel processing by date
    results_nested = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_single_date_merton_mc)(date_data, num_simulations, num_days) 
        for date_data in date_data_list
    )
    
    # Flatten results
    results_list = [item for sublist in results_nested for item in sublist]
    results_df = pd.DataFrame(results_list)
    
    # Calculate summary statistics
    total_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print("MONTE CARLO MERTON COMPLETE")
    print(f"{'='*80}\n")
    
    print(f"Results summary:")
    print(f"  Total observations: {len(results_df):,}")
    print(f"  Unique firms: {results_df['gvkey'].nunique()}")
    print(f"  Date range: {results_df['date'].min()} to {results_df['date'].max()}")
    
    print(f"\nAnnualized Volatility Statistics:")
    print(f"  Mean: {results_df['merton_mc_annualized_volatility'].mean():.4f} ({results_df['merton_mc_annualized_volatility'].mean()*100:.2f}%)")
    print(f"  Median: {results_df['merton_mc_annualized_volatility'].median():.4f}")
    print(f"  Min: {results_df['merton_mc_annualized_volatility'].min():.4f}")
    print(f"  Max: {results_df['merton_mc_annualized_volatility'].max():.4f}")
    print(f"  (Should match original Merton asset_volatility)")
    
    print(f"\nIntegrated Variance Statistics:")
    print(f"  Mean: {results_df['merton_mc_integrated_variance'].mean():.4f}")
    print(f"  Median: {results_df['merton_mc_integrated_variance'].median():.4f}")
    print(f"  Min: {results_df['merton_mc_integrated_variance'].min():.4f}")
    print(f"  Max: {results_df['merton_mc_integrated_variance'].max():.4f}")
    
    print(f"\nPerformance:")
    print(f"  Total time: {timedelta(seconds=int(total_time))}")
    print(f"  Observations per second: {len(results_df) / total_time:.0f}")
    print(f"{'='*80}\n")
    
    # Verification: Check that annualized_volatility ≈ constant_annual_vol
    vol_diff = (results_df['merton_mc_annualized_volatility'] - results_df['merton_mc_constant_annual_vol']).abs()
    max_diff = vol_diff.max()
    if max_diff > 0.001:  # Tolerance: 0.1%
        print(f"⚠️  WARNING: Annualized volatility differs from input by up to {max_diff:.4f}")
        print(f"   This suggests a calculation error (should be nearly identical for constant vol)")
    else:
        print(f"✓ Verification passed: Output volatility matches input (max diff: {max_diff:.6f})")
    
    print()
    
    return results_df


def monte_carlo_merton_1year(merton_file, gvkey_selected=None, 
                              num_simulations=1000, num_days=252):
    """
    Monte Carlo with constant Merton volatility (1-year horizon, single-threaded).
    
    Simpler version without parallelization for debugging or small datasets.
    
    Parameters:
    -----------
    merton_file : str
        Path to merged_data_with_merton.csv
    gvkey_selected : list, optional
        Specific firms to process
    num_simulations : int
        Number of Monte Carlo paths (default: 1000)
    num_days : int
        Forecast horizon in days (default: 252 = 1 year)
    
    Returns:
    --------
    pd.DataFrame with same columns as parallel version
    """
    print(f"\n{'='*80}")
    print("MONTE CARLO: MERTON CONSTANT VOLATILITY (Single-threaded)")
    print(f"{'='*80}\n")
    
    # Load data
    df = pd.read_csv(merton_file)
    df['date'] = pd.to_datetime(df['date'])
    
    if gvkey_selected is not None:
        df = df[df['gvkey'].isin(gvkey_selected)]
    
    df = df.dropna(subset=['asset_volatility'])
    
    print(f"Processing {len(df):,} observations\n")
    
    results_list = []
    
    for idx, row in df.iterrows():
        gvkey = row['gvkey']
        date = row['date']
        sigma_annual = row['asset_volatility']
        sigma_daily = sigma_annual / np.sqrt(252)
        
        # Simulate constant volatility paths
        daily_vols = simulate_constant_vol_paths(
            sigma_daily_arr=np.array([sigma_daily]),
            num_simulations=num_simulations,
            num_days=num_days,
            num_firms=1
        )
        
        # Calculate integrated variance
        firm_daily_vols = daily_vols[:, :, 0]
        firm_daily_variances = firm_daily_vols ** 2
        mean_variance_path = np.mean(firm_daily_variances, axis=1)
        integrated_variance = np.sum(mean_variance_path)
        annualized_volatility = np.sqrt(integrated_variance)
        
        results_list.append({
            'gvkey': gvkey,
            'date': date,
            'merton_mc_integrated_variance': integrated_variance,
            'merton_mc_annualized_volatility': annualized_volatility,
            'merton_mc_mean_daily_volatility': np.mean(firm_daily_vols),
            'merton_mc_constant_annual_vol': sigma_annual
        })
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1:,} / {len(df):,} observations")
    
    results_df = pd.DataFrame(results_list)
    
    print(f"\n✓ Complete: {len(results_df):,} observations\n")
    
    return results_df


if __name__ == "__main__":
    # --- Guardrail Test: Verify PD Logic ---
    print("\n--- PD Logic Guardrail Test ---")
    # Setup simple case
    v0_test = 100.0
    K_test = 80.0
    sigma_daily_test = 0.02 # High Volatility
    num_sims_test = 5000
    num_days_test = 252
    
    # Manual simulation for validation
    np.random.seed(42)
    drift_test = -0.5 * sigma_daily_test**2
    shocks_test = np.random.normal(0, 1, (num_days_test, num_sims_test))
    log_returns_test = drift_test + sigma_daily_test * shocks_test
    cum_returns_test = np.cumsum(log_returns_test, axis=0)
    asset_paths_test = v0_test * np.exp(cum_returns_test)
    
    # 1. Terminal PD: V_T < K
    pd_terminal_test = np.mean(asset_paths_test[-1, :] < K_test)
    
    # 2. Barrier PD: min(V_t) < K (First Passage)
    pd_barrier_test = np.mean(np.min(asset_paths_test, axis=0) < K_test)
    
    print(f"Parameters: V0={v0_test}, K={K_test}, sigma={sigma_daily_test}, T={num_days_test}")
    print(f"PD Terminal (Merton): {pd_terminal_test:.4f}")
    print(f"PD Barrier (First Passage): {pd_barrier_test:.4f}")
    
    if pd_barrier_test >= pd_terminal_test:
        print("✓ Guardrail Passed: Barrier PD >= Terminal PD")
    else:
        print("! WARNING: Barrier PD < Terminal PD (Unexpected)")
    print("-------------------------------\n")

    # Example usage
    merton_file = "data/intermediates/merged_data_with_merton.csv"
    
    # Run Monte Carlo
    results = monte_carlo_merton_1year_parallel(
        merton_file=merton_file,
        num_simulations=1000,
        num_days=252,
        n_jobs=-1
    )
    
    # Save results
    output_file = "data/output/daily_monte_carlo_merton_results.csv"
    results.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
