
# monte_carlo_regime_switching.py
# Efficient regime-switching Monte Carlo with vectorization

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba
from joblib import Parallel, delayed

os.makedirs('./intermediates/', exist_ok=True)


@numba.jit(nopython=True, parallel=True)
def simulate_regime_switching_paths_vectorized(
    mu_0, mu_1, ar_0, ar_1, sigma_0, sigma_1,
    trans_00, trans_01, trans_10, trans_11,
    num_simulations, num_days, num_firms, initial_regime_probs
):
    """
    Vectorized regime-switching Monte Carlo simulation.
    
    For each firm, simulates:
    1. Regime path using Markov transitions
    2. Returns conditional on regime
    3. Volatility path
    
    Uses efficient matrix operations and Numba JIT compilation.
    
    Parameters:
    -----------
    mu_0, mu_1 : arrays (num_firms,)
        Regime 0 and 1 mean returns
    ar_0, ar_1 : arrays (num_firms,)
        Regime 0 and 1 AR(1) coefficients
    sigma_0, sigma_1 : arrays (num_firms,)
        Regime 0 and 1 volatilities (daily)
    trans_00, trans_01, trans_10, trans_11 : arrays (num_firms,)
        Transition probabilities for each firm
    num_simulations : int
        Number of Monte Carlo paths
    num_days : int
        Forecast horizon (days)
    num_firms : int
        Number of firms
    initial_regime_probs : array (num_firms,)
        Initial regime probabilities (typically 0.5, 0.5)
    
    Returns:
    --------
    daily_volatilities : array (num_days, num_simulations, num_firms)
        Simulated daily volatilities
    regime_paths : array (num_days, num_simulations, num_firms)
        Regime state paths (0 or 1)
    """
    
    daily_volatilities = np.zeros((num_days, num_simulations, num_firms))
    regime_paths = np.zeros((num_days, num_simulations, num_firms), dtype=np.int32)
    
    for sim in numba.prange(num_simulations):
        # Initialize regime (0 or 1) for each firm based on initial probs
        regime = np.zeros(num_firms, dtype=np.int32)
        for f in range(num_firms):
            if np.random.uniform() < initial_regime_probs[f]:
                regime[f] = 1
            else:
                regime[f] = 0
        
        r_prev = np.zeros(num_firms)  # Previous returns for AR(1)
        
        for day in range(num_days):
            # Regime transitions (vectorized)
            for f in range(num_firms):
                if regime[f] == 0:
                    # In regime 0, stay with prob trans_00, switch with prob trans_01
                    if np.random.uniform() < trans_01[f]:
                        regime[f] = 1
                else:
                    # In regime 1, switch with prob trans_10, stay with prob trans_11
                    if np.random.uniform() < trans_10[f]:
                        regime[f] = 0
            
            # Generate returns conditional on regime
            z = np.random.standard_normal(num_firms)
            
            for f in range(num_firms):
                if regime[f] == 0:
                    # Regime 0: AR(1) with low volatility
                    mu = mu_0[f]
                    ar = ar_0[f]
                    sigma = sigma_0[f]
                else:
                    # Regime 1: AR(1) with high volatility
                    mu = mu_1[f]
                    ar = ar_1[f]
                    sigma = sigma_1[f]
                
                # AR(1) return: r_t = μ + φ*r_{t-1} + σ*z_t
                r_curr = mu + ar * r_prev[f] + sigma * z[f]
                r_prev[f] = r_curr
                
                # Volatility = conditional std dev (changes with regime)
                daily_volatilities[day, sim, f] = sigma
            
            # Store regime path
            regime_paths[day, sim, :] = regime
    
    return daily_volatilities, regime_paths


def monte_carlo_regime_switching_1year(
    garch_file, 
    regime_params_file,
    gvkey_selected=None, 
    num_simulations=1000, 
    num_days=252
):
    """
    Run Monte Carlo regime-switching forecast for 1 year (252 trading days).
    
    VECTORIZED: All firms and simulations processed efficiently using Numba.
    
    For each trading date and firm:
    1. Load regime parameters from CSV
    2. Run 1,000 simulations with regime transitions
    3. Calculate mean volatility path (mean across simulations for each day)
    4. Sum mean volatilities over 252 days (cumulative volatility)
    
    Returns one row per firm per date with cumulative volatility.
    
    Parameters:
    -----------
    garch_file : str
        Path to GARCH results (for base data)
    regime_params_file : str
        Path to regime-switching parameters CSV
    gvkey_selected : int or None
        If None, run for ALL firms
    num_simulations : int
        Number of Monte Carlo paths
    num_days : int
        Forecast horizon (days)
    
    Returns:
    --------
    results_df : pd.DataFrame
        Cumulative volatility forecasts per firm per date
    """
    
    start_time = pd.Timestamp.now()
    
    print(f"\n{'='*80}")
    print("MONTE CARLO REGIME-SWITCHING 1-YEAR CUMULATIVE VOLATILITY FORECAST")
    print(f"{'='*80}\n")
    
    # Load regime parameters
    try:
        regime_params = pd.read_csv(regime_params_file)
        print(f"✓ Loaded regime-switching parameters for {len(regime_params)} firms\n")
    except FileNotFoundError:
        print(f"✗ File '{regime_params_file}' not found!")
        print(f"   Run regime switching estimation first.\n")
        return pd.DataFrame()
    
    # Load GARCH base data for dates and firms
    df_garch = pd.read_csv(garch_file)
    df_garch['date'] = pd.to_datetime(df_garch['date'])
    df_garch = df_garch.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_garch):,} GARCH observations")
    
    # Get unique dates and firms
    unique_dates = sorted(df_garch['date'].unique())
    firm_list = sorted(df_garch['gvkey'].unique())
    
    num_dates = len(unique_dates)
    num_firms_total = len(firm_list)
    
    print(f"✓ Unique dates: {num_dates}")
    print(f"✓ Unique firms: {num_firms_total}")
    print(f"✓ Date range: {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Expected output rows: ~{num_dates * num_firms_total:,}\n")
    
    if gvkey_selected is not None:
        firm_list = [gvkey_selected]
        num_firms_total = 1
        print(f"⚠ Filtering to single firm: {gvkey_selected}\n")
    
    results_list = []
    
    # Process each date
    for date_idx, date in enumerate(unique_dates):
        if date_idx % max(1, num_dates // 10) == 0:
            print(f"Progress: {date_idx + 1}/{num_dates} ({date.strftime('%Y-%m-%d')})")
        
        # Get firms with data on this date
        df_date = df_garch[df_garch['date'] == date]
        firms_on_date = sorted(df_date['gvkey'].unique())
        
        # Filter to selected firms if applicable
        firms_on_date = [f for f in firms_on_date if f in firm_list]
        
        if len(firms_on_date) == 0:
            continue
        
        # Get regime parameters for these firms
        regime_data = regime_params[regime_params['gvkey'].isin(firms_on_date)].set_index('gvkey')
        
        if len(regime_data) == 0:
            continue
        
        # PREPARE VECTORIZED ARRAYS
        mu_0_arr = np.array([regime_data.loc[f, 'regime_0_mean'] for f in firms_on_date])
        mu_1_arr = np.array([regime_data.loc[f, 'regime_1_mean'] for f in firms_on_date])
        ar_0_arr = np.array([regime_data.loc[f, 'regime_0_ar'] for f in firms_on_date])
        ar_1_arr = np.array([regime_data.loc[f, 'regime_1_ar'] for f in firms_on_date])
        sigma_0_arr = np.array([regime_data.loc[f, 'regime_0_vol'] for f in firms_on_date])
        sigma_1_arr = np.array([regime_data.loc[f, 'regime_1_vol'] for f in firms_on_date])
        trans_00_arr = np.array([regime_data.loc[f, 'transition_prob_00'] for f in firms_on_date])
        trans_01_arr = np.array([regime_data.loc[f, 'transition_prob_01'] for f in firms_on_date])
        trans_10_arr = np.array([regime_data.loc[f, 'transition_prob_10'] for f in firms_on_date])
        trans_11_arr = np.array([regime_data.loc[f, 'transition_prob_11'] for f in firms_on_date])
        
        # Initial regime probabilities (typically 0.5, 0.5)
        initial_regime_probs = np.full(len(firms_on_date), 0.5)
        
        # RUN VECTORIZED MONTE CARLO SIMULATION
        daily_vols, regime_paths = simulate_regime_switching_paths_vectorized(
            mu_0_arr, mu_1_arr, ar_0_arr, ar_1_arr, sigma_0_arr, sigma_1_arr,
            trans_00_arr, trans_01_arr, trans_10_arr, trans_11_arr,
            num_simulations, num_days, len(firms_on_date), initial_regime_probs
        )
        
        # CALCULATE STATISTICS
        # daily_vols shape: (num_days, num_simulations, num_firms)
        mean_paths = np.mean(daily_vols, axis=1)  # (num_days, num_firms)
        
        cumulative_vols = np.sum(mean_paths, axis=0)  # (num_firms,)
        mean_daily_vols = np.mean(mean_paths, axis=0)
        std_daily_vols = np.std(mean_paths, axis=0)
        min_daily_vols = np.min(mean_paths, axis=0)
        max_daily_vols = np.max(mean_paths, axis=0)
        
        # Calculate regime statistics
        mean_regime_0 = np.mean(regime_paths == 0, axis=(0, 1))  # Fraction time in regime 0
        mean_regime_1 = np.mean(regime_paths == 1, axis=(0, 1))
        
        # APPEND RESULTS
        for firm_idx, firm in enumerate(firms_on_date):
            results_list.append({
                'gvkey': firm,
                'date': date,
                'rs_cumulative_volatility': cumulative_vols[firm_idx],
                'rs_mean_daily_volatility': mean_daily_vols[firm_idx],
                'rs_std_daily_volatility': std_daily_vols[firm_idx],
                'rs_min_daily_volatility': min_daily_vols[firm_idx],
                'rs_max_daily_volatility': max_daily_vols[firm_idx],
                'rs_volatility_forecast': mean_daily_vols[firm_idx],
                'rs_fraction_regime_0': mean_regime_0[firm_idx],
                'rs_fraction_regime_1': mean_regime_1[firm_idx],
            })
    
    results_df = pd.DataFrame(results_list)
    
    print(f"\n{'='*80}")
    print("REGIME-SWITCHING MONTE CARLO COMPLETE")
    print(f"{'='*80}\n")
    
    print(f"Total output rows: {len(results_df):,}")
    print(f"Unique firms: {results_df['gvkey'].nunique()}")
    print(f"Unique dates: {results_df['date'].nunique()}")
    
    # Year-wise breakdown
    results_df['year'] = results_df['date'].dt.year
    print(f"\nRows per year:")
    for year in sorted(results_df['year'].unique()):
        year_count = len(results_df[results_df['year'] == year])
        print(f"  {year}: {year_count:,} rows")
    print()
    
    # Sample statistics
    print(f"Sample statistics (first 3 firms):")
    for firm in results_df['gvkey'].unique()[:3]:
        firm_data = results_df[results_df['gvkey'] == firm]
        print(f"  Firm {firm}: {len(firm_data):,} trading days")
        print(f"    Cumulative vol: {firm_data['rs_cumulative_volatility'].mean():.4f}")
        print(f"    Regime 0 fraction: {firm_data['rs_fraction_regime_0'].mean():.3f}")
        print(f"    Regime 1 fraction: {firm_data['rs_fraction_regime_1'].mean():.3f}")
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Rows per second: {len(results_df) / total_time:.0f}")
    print(f"{'='*80}\n")
    
    return results_df
