# monte_carlo_merton.py

import pandas as pd
import numpy as np
import numba
from datetime import timedelta
from joblib import Parallel, delayed


@numba.jit(nopython=True, fastmath=True, cache=True)
def simulate_merton_pd_spreads_jit(sigma_daily_arr, mu_arr,
                                    num_simulations, num_firms,
                                    horizon_days, v0_arr, liability_arr, rf_arr):
    """
    Optimized Monte Carlo Merton simulation with constant volatility.
    Uses log-asset evolution to minimize exp() calls.
    Returns only per-firm, per-horizon aggregates (no per-simulation arrays).
    """
    max_days = np.max(horizon_days)
    n_horizons = len(horizon_days)
    
    # Output arrays: (n_horizons, num_firms)
    pd_out = np.full((n_horizons, num_firms), np.nan)
    spread_out = np.full((n_horizons, num_firms), np.nan)
    debt_out = np.full((n_horizons, num_firms), np.nan)
    
    # Pre-compute horizon mask
    is_horizon = np.zeros(max_days, dtype=np.int32)
    horizon_map = np.full(max_days, -1, dtype=np.int32)
    for h in range(n_horizons):
        day_idx = horizon_days[h] - 1
        if day_idx < max_days:
            is_horizon[day_idx] = 1
            horizon_map[day_idx] = h
    
    # Process firms sequentially (Joblib handles parallel dates)
    for f in range(num_firms):
        # Constants
        sigma = sigma_daily_arr[f]
        mu = mu_arr[f]
        v0 = v0_arr[f]
        liability = liability_arr[f]
        rf_rate = rf_arr[f]
        
        # Check validity of Merton inputs
        valid_merton = (not np.isnan(v0)) and (not np.isnan(liability)) and (v0 > 0) and (liability > 0)
        valid_rf = (not np.isnan(rf_rate))
        
        if not valid_merton:
            # Skip this firm - outputs remain NaN
            continue
        
        # Initialize log-space
        log_v0 = np.log(v0)
        log_liability = np.log(liability)
        
        # State Vectors (Size: num_simulations)
        log_asset = np.full(num_simulations, log_v0)
        
        # Accumulators for each horizon
        default_counts = np.zeros(n_horizons)
        payoff_sums = np.zeros(n_horizons)
        
        for day in range(max_days):
            # 1. Vectorized Random Generation (standard normal for Merton)
            z = np.random.standard_normal(num_simulations)
            
            # 2. Vectorized Updates with constant volatility
            eps = sigma * z
            
            # Update log-asset (no exp needed daily)
            log_asset += mu + eps
            
            # 3. Horizon Check
            if is_horizon[day]:
                h = horizon_map[day]
                
                # Default detection in log-space (Terminal PD only for Merton)
                defaults = (log_asset < log_liability).astype(np.float64)
                default_counts[h] = np.sum(defaults)
                
                # Compute asset values at horizon (exp only here)
                asset_T = np.exp(log_asset)
                payoffs = np.minimum(asset_T, liability)
                payoff_sums[h] = np.sum(payoffs)
        
        # Compute PD and spreads for each horizon
        for h in range(n_horizons):
            # PD
            pd_out[h, f] = default_counts[h] / num_simulations
            
            # Expected payoff and debt value
            expected_payoff = payoff_sums[h] / num_simulations
            
            # Compute spreads only if rf_rate is valid
            if valid_rf:
                T_years = horizon_days[h] / 252.0
                debt_val = expected_payoff * np.exp(-rf_rate * T_years)
                debt_out[h, f] = debt_val
                
                if debt_val > 0:
                    ytm = -np.log(debt_val / liability) / T_years
                    spread_out[h, f] = max(ytm - rf_rate, 0.0)
    
    return pd_out, spread_out, debt_out




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
        Forecast horizon in days (should be max of horizons, e.g., 1260 for 5y)
        
    Returns:
    --------
    list : Results for all firms on this date
    """
    date, df_date = date_data
    
    if df_date.empty:
        return []
    
    # Vectorized preparation: drop duplicates and sort
    df_firms = df_date.drop_duplicates('gvkey', keep='first').sort_values('gvkey').reset_index(drop=True)
    firms_list = df_firms['gvkey'].tolist()
    num_firms = len(firms_list)
    
    # Vectorized parameter extraction with defaults and floors
    sigma_annual_arr = np.maximum(df_firms.get('asset_volatility', pd.Series([0.2]*num_firms)).fillna(0.2).values, 1e-4)
    sigma_daily_arr = sigma_annual_arr / np.sqrt(252)  # Convert annual to daily
    
    # Drift term (typically 0 for Merton, but can be set if needed)
    mu_arr = df_firms.get('merton_mu_daily', pd.Series([0.0]*num_firms)).fillna(0.0).values
    
    # Prepare Merton arrays
    v0_arr = df_firms.get('asset_value', pd.Series([np.nan]*num_firms)).fillna(np.nan).values
    liability_arr = df_firms.get('liabilities_total', pd.Series([np.nan]*num_firms)).fillna(np.nan).values
    rf_arr = df_firms.get('risk_free_rate', pd.Series([np.nan]*num_firms)).fillna(np.nan).values
    
    # Scale rf_rate if needed (convert from percentage)
    for i in range(num_firms):
        if not np.isnan(rf_arr[i]) and abs(rf_arr[i]) > 0.5:
            rf_arr[i] = rf_arr[i] / 100.0
    
    # Define horizons: 1y, 3y, 5y (trading days)
    horizon_days = np.array([252, 756, 1260], dtype=np.int32)
    
    # Run simulation
    pd_out, spread_out, debt_out = simulate_merton_pd_spreads_jit(
        sigma_daily_arr, mu_arr,
        num_simulations, num_firms, horizon_days,
        v0_arr, liability_arr, rf_arr
    )
    
    # Extract results by horizon
    pd_1y = pd_out[0, :]
    pd_3y = pd_out[1, :]
    pd_5y = pd_out[2, :]
    
    mc_spread_1y = spread_out[0, :]
    mc_spread_3y = spread_out[1, :]
    mc_spread_5y = spread_out[2, :]
    
    mc_debt_1y = debt_out[0, :]
    mc_debt_3y = debt_out[1, :]
    mc_debt_5y = debt_out[2, :]
    
    # Assemble results
    results_list = []
    for firm_idx, firm in enumerate(firms_list):
        results_list.append({
            'gvkey': firm,
            'date': date,
            'merton_mc_pd_1y': pd_1y[firm_idx],
            'merton_mc_pd_3y': pd_3y[firm_idx],
            'merton_mc_pd_5y': pd_5y[firm_idx],
            'merton_mc_pd_terminal_1y': pd_1y[firm_idx],
            'merton_mc_pd_terminal_3y': pd_3y[firm_idx],
            'merton_mc_pd_terminal_5y': pd_5y[firm_idx],
            'merton_mc_implied_spread_1y': mc_spread_1y[firm_idx],
            'merton_mc_implied_spread_3y': mc_spread_3y[firm_idx],
            'merton_mc_implied_spread_5y': mc_spread_5y[firm_idx],
            'merton_mc_debt_value_1y': mc_debt_1y[firm_idx],
            'merton_mc_debt_value_3y': mc_debt_3y[firm_idx],
            'merton_mc_debt_value_5y': mc_debt_5y[firm_idx],
        })
    
    return results_list



def monte_carlo_merton_1year_parallel(merton_file, gvkey_selected=None, num_simulations=1000, num_days=1260, n_jobs=-1):
    print(f"Loading Merton data from {merton_file}...")
    df = pd.read_csv(merton_file)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter firms if specified
    if gvkey_selected is not None:
        df = df[df['gvkey'].isin(gvkey_selected)]
    
    print(f"Running PARALLELIZED Monte Carlo Merton simulation:")
    print(f"  Firms: {df['gvkey'].nunique()}")
    print(f"  Dates: {df['date'].nunique() if 'date' in df.columns else 1}")
    print(f"  Simulations per firm: {num_simulations:,}")
    print(f"  Forecast horizon: {num_days} days")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Innovation distribution: Normal (constant volatility)")
    
    start_time = pd.Timestamp.now()
    
    # Prepare date groups for parallel processing
    date_groups = []
    if 'date' in df.columns:
        for date, group in df.groupby('date'):
            date_groups.append((date, group))
    else:
        # Single date case
        date_groups = [(pd.Timestamp.now().date(), df)]
    
    print(f"\nProcessing {len(date_groups)} dates in parallel...")
    
    # Parallel processing across dates
    results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_date_merton_mc)(date_data, num_simulations, num_days) 
        for date_data in date_groups
    )
    
    # Flatten results
    results_list = []
    for date_results in results_nested:
        results_list.extend(date_results)
    
    results_df = pd.DataFrame(results_list)
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"PARALLELIZED MONTE CARLO MERTON COMPLETE")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"{'='*80}\n")
    
    return results_df

