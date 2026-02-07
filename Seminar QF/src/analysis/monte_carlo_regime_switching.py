
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
            z_raw = np.random.standard_normal(num_firms)
            # Cap extreme innovations to prevent temporary spikes (consistent with MS-GARCH)
            z = np.maximum(np.minimum(z_raw, 5.0), -5.0)  # Cap at ±5 sigma
            
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
        
        # MEAN VARIANCE CALCULATION
        # 1. Square to get variances
        daily_variances = daily_vols ** 2
        
        # 2. Mean across simulations (Expected Conditional Variance)
        mean_variance_paths = np.mean(daily_variances, axis=1)  # (num_days, num_firms)
        
        # 3. Sum over horizon (Integrated Variance)
        integrated_variances = np.sum(mean_variance_paths, axis=0)  # (num_firms,)
        
        # Other stats
        mean_daily_vols = np.mean(daily_vols, axis=(0,1)) # overall mean
        
        # Calculate regime statistics
        mean_regime_0 = np.mean(regime_paths == 0, axis=(0, 1))  # Fraction time in regime 0
        mean_regime_1 = np.mean(regime_paths == 1, axis=(0, 1))
        
        # APPEND RESULTS
        for firm_idx, firm in enumerate(firms_on_date):
            results_list.append({
                'gvkey': firm,
                'date': date,
                'rs_integrated_variance': integrated_variances[firm_idx],
                'rs_mean_daily_volatility': mean_daily_vols[firm_idx],
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
        print(f"    Integrated Variance: {firm_data['rs_integrated_variance'].mean():.4f}")
        print(f"    Regime 0 fraction: {firm_data['rs_fraction_regime_0'].mean():.3f}")
        print(f"    Regime 1 fraction: {firm_data['rs_fraction_regime_1'].mean():.3f}")
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Rows per second: {len(results_df) / total_time:.0f}")
    print(f"{'='*80}\n")
    
    return results_df


def _process_single_date_rs_mc(date_data, num_simulations, num_days):
    """
    Process Monte Carlo regime-switching simulation for a single date (for parallelization).
    """
    date, df_date, regime_params_dict, firms_on_date = date_data
    results_list = []
    
    if len(firms_on_date) == 0:
        return results_list
    
    # Prepare arrays for vectorized simulation
    n_firms = len(firms_on_date)
    mu_0_arr = np.zeros(n_firms)
    mu_1_arr = np.zeros(n_firms)
    ar_0_arr = np.zeros(n_firms)
    ar_1_arr = np.zeros(n_firms)
    sigma_0_arr = np.zeros(n_firms)
    sigma_1_arr = np.zeros(n_firms)
    trans_00_arr = np.zeros(n_firms)
    trans_01_arr = np.zeros(n_firms)
    trans_10_arr = np.zeros(n_firms)
    trans_11_arr = np.zeros(n_firms)
    
    valid_firms = []
    for f_idx, firm in enumerate(firms_on_date):
        if firm not in regime_params_dict:
            continue
        params = regime_params_dict[firm]
        mu_0_arr[f_idx] = params['regime_0_mean']
        mu_1_arr[f_idx] = params['regime_1_mean']
        ar_0_arr[f_idx] = params['regime_0_ar']
        ar_1_arr[f_idx] = params['regime_1_ar']
        sigma_0_arr[f_idx] = params['regime_0_vol']
        sigma_1_arr[f_idx] = params['regime_1_vol']
        trans_00_arr[f_idx] = params['transition_prob_00']
        trans_01_arr[f_idx] = params['transition_prob_01']
        trans_10_arr[f_idx] = params['transition_prob_10']
        trans_11_arr[f_idx] = params['transition_prob_11']
        valid_firms.append((f_idx, firm))
    
    if len(valid_firms) == 0:
        return results_list
    
    # Initial regime probabilities
    initial_regime_probs = np.full(n_firms, 0.5)
    
    # Run vectorized Monte Carlo
    daily_vols, regime_paths = simulate_regime_switching_paths_vectorized(
        mu_0_arr, mu_1_arr, ar_0_arr, ar_1_arr, sigma_0_arr, sigma_1_arr,
        trans_00_arr, trans_01_arr, trans_10_arr, trans_11_arr,
        num_simulations, num_days, n_firms, initial_regime_probs
    )
    
    # Calculate statistics
    daily_variances = daily_vols ** 2
    mean_variance_paths = np.mean(daily_variances, axis=1)
    integrated_variances = np.sum(mean_variance_paths, axis=0)
    mean_daily_vols = np.mean(daily_vols, axis=(0, 1))
    mean_regime_0 = np.mean(regime_paths == 0, axis=(0, 1))
    mean_regime_1 = np.mean(regime_paths == 1, axis=(0, 1))
    
    # Append results for valid firms
    for f_idx, firm in valid_firms:
        results_list.append({
            'gvkey': firm,
            'date': date,
            'rs_integrated_variance': integrated_variances[f_idx],
            'rs_mean_daily_volatility': mean_daily_vols[f_idx],
            'rs_fraction_regime_0': mean_regime_0[f_idx],
            'rs_fraction_regime_1': mean_regime_1[f_idx],
        })
    
    return results_list


def monte_carlo_regime_switching_1year_parallel(
    garch_file, 
    regime_params_file,
    gvkey_selected=None, 
    num_simulations=1000, 
    num_days=252,
    n_jobs=-1
):
    """
    PARALLELIZED Monte Carlo regime-switching forecast for 1 year.
    
    This version processes different dates in parallel for significant speedup.
    """
    start_time = pd.Timestamp.now()
    
    print(f"\n{'='*80}")
    print("PARALLELIZED MONTE CARLO REGIME-SWITCHING 1-YEAR FORECAST")
    print(f"{'='*80}\n")
    
    # Load regime parameters
    try:
        regime_params = pd.read_csv(regime_params_file)
        print(f"✓ Loaded regime-switching parameters for {len(regime_params)} firms")
    except FileNotFoundError:
        print(f"✗ File '{regime_params_file}' not found!")
        return pd.DataFrame()
    
    # Create parameters dictionary for faster lookup
    regime_params_dict = {}
    for _, row in regime_params.iterrows():
        regime_params_dict[row['gvkey']] = {
            'regime_0_mean': row['regime_0_mean'],
            'regime_1_mean': row['regime_1_mean'],
            'regime_0_ar': row['regime_0_ar'],
            'regime_1_ar': row['regime_1_ar'],
            'regime_0_vol': row['regime_0_vol'],
            'regime_1_vol': row['regime_1_vol'],
            'transition_prob_00': row['transition_prob_00'],
            'transition_prob_01': row['transition_prob_01'],
            'transition_prob_10': row['transition_prob_10'],
            'transition_prob_11': row['transition_prob_11'],
        }
    
    # Load GARCH base data
    df_garch = pd.read_csv(garch_file)
    df_garch['date'] = pd.to_datetime(df_garch['date'])
    
    if gvkey_selected is not None:
        df_garch = df_garch[df_garch['gvkey'].isin(gvkey_selected)]
    
    print(f"✓ Loaded {len(df_garch):,} observations")
    print(f"  Firms: {df_garch['gvkey'].nunique()}")
    print(f"  Dates: {df_garch['date'].nunique()}")
    print(f"  Simulations: {num_simulations:,}")
    print(f"  Horizon: {num_days} days")
    print(f"  Parallel jobs: {n_jobs}")
    
    # Prepare date groups
    date_groups = []
    for date, group in df_garch.groupby('date'):
        firms_on_date = list(group['gvkey'].unique())
        date_groups.append((date, group, regime_params_dict, firms_on_date))
    
    print(f"\nProcessing {len(date_groups)} dates in parallel...")
    
    # Parallel processing
    results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_date_rs_mc)(date_data, num_simulations, num_days)
        for date_data in date_groups
    )
    
    # Flatten results
    results_list = []
    for date_results in results_nested:
        results_list.extend(date_results)
    
    results_df = pd.DataFrame(results_list)
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print("PARALLELIZED REGIME-SWITCHING MC COMPLETE")
    print(f"Total rows: {len(results_df):,}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Rows per second: {len(results_df) / total_time:.0f}")
    print(f"{'='*80}\n")
    
    return results_df
