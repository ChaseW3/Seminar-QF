# monte_carlo_garch.py

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba
from scipy import stats
from joblib import Parallel, delayed

os.makedirs('./intermediates/', exist_ok=True)


@numba.jit(nopython=True, parallel=True)
def simulate_garch_paths_vectorized_daily(omega, alpha, beta, sigma_t, returns, 
                                          num_simulations, num_days, num_firms):
    """
    Vectorized GARCH Monte Carlo simulation that returns volatility for EACH day.
    Uses standard normal innovations (legacy function for compatibility).
    
    Returns:
    --------
    daily_volatilities : array, shape (num_days, num_simulations, num_firms)
        Volatility for each day, each simulation, and each firm
    """
    
    daily_volatilities = np.zeros((num_days, num_simulations, num_firms))
    
    for sim in numba.prange(num_simulations):
        sigma = sigma_t.copy()
        
        for day in range(num_days):
            z = np.random.standard_normal(num_firms)
            r = sigma * z
            sigma_squared = omega + alpha * (r ** 2) + beta * (sigma ** 2)
            sigma = np.sqrt(np.maximum(sigma_squared, 1e-6))
            
            # Store volatility for this day and simulation
            daily_volatilities[day, sim, :] = sigma
    
    return daily_volatilities


def simulate_garch_paths_t_dist(omega, alpha, beta, sigma_t, nu, 
                                num_simulations, num_days, num_firms):
    """
    GARCH Monte Carlo simulation with Student's t distributed innovations.
    Non-numba version to support scipy distributions.
    
    Parameters:
    -----------
    omega, alpha, beta : arrays of GARCH parameters per firm
    sigma_t : array of initial volatilities per firm
    nu : array of degrees of freedom per firm (from t-GARCH estimation)
    num_simulations : number of Monte Carlo paths
    num_days : forecast horizon in days
    num_firms : number of firms
    
    Returns:
    --------
    daily_volatilities : array, shape (num_days, num_simulations, num_firms)
    """
    
    daily_volatilities = np.zeros((num_days, num_simulations, num_firms))
    
    for sim in range(num_simulations):
        sigma = sigma_t.copy()
        
        for day in range(num_days):
            # Generate t-distributed innovations for each firm
            # Standardize so variance = 1: t(nu) has variance nu/(nu-2)
            z = np.zeros(num_firms)
            for f in range(num_firms):
                if nu[f] > 2:
                    # t-distribution with standardization
                    scale = np.sqrt((nu[f] - 2) / nu[f])
                    z[f] = stats.t.rvs(df=nu[f]) * scale
                else:
                    # Fallback to normal if nu <= 2 (infinite variance)
                    z[f] = np.random.standard_normal()
            
            r = sigma * z
            sigma_squared = omega + alpha * (r ** 2) + beta * (sigma ** 2)
            sigma = np.sqrt(np.maximum(sigma_squared, 1e-6))
            
            daily_volatilities[day, sim, :] = sigma
    
    return daily_volatilities


def monte_carlo_garch_1year(garch_file, gvkey_selected=None, num_simulations=1000, num_days=252):
    """
    Run Monte Carlo GARCH forecast for 1 year (252 trading days).
    
    For each trading day and firm:
    1. Run 1,000 simulations
    2. Calculate mean volatility across simulations for each day
    3. Sum the mean volatilities over the 252-day window
    
    Returns one row per firm per date with cumulative volatility over the rolling window.
    """
    
    start_time = pd.Timestamp.now()
    
    print(f"\n{'='*80}")
    print("MONTE CARLO GARCH 1-YEAR CUMULATIVE VOLATILITY FORECAST")
    print(f"{'='*80}\n")
    
    # Load GARCH results
    df_garch = pd.read_csv(garch_file)
    df_garch['date'] = pd.to_datetime(df_garch['date'])
    df_garch = df_garch.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_garch):,} total observations")
    
    # Get unique dates and firms
    unique_dates = sorted(df_garch['date'].unique())
    firm_list = sorted(df_garch['gvkey'].unique())
    
    num_dates = len(unique_dates)
    num_firms = len(firm_list)
    
    print(f"✓ Unique dates: {num_dates}")
    print(f"✓ Unique firms: {num_firms}")
    print(f"✓ Date range: {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Expected output rows: ~{num_dates * num_firms:,}\n")
    
    if gvkey_selected is not None:
        firm_list = [gvkey_selected]
        df_garch = df_garch[df_garch['gvkey'] == gvkey_selected]
        num_firms = 1
        print(f"⚠ Filtering to single firm: {gvkey_selected}\n")
    
    results_list = []
    dates_processed = 0
    rows_created = 0
    
    # Process each date
    for date_idx, date in enumerate(unique_dates):
        # Progress reporting every 10%
        if date_idx % max(1, num_dates // 10) == 0:
            print(f"Progress: {date_idx + 1}/{num_dates} ({date.strftime('%Y-%m-%d')})")
        
        # Get GARCH parameters for this date
        df_date = df_garch[df_garch['date'] == date]
        
        if len(df_date) == 0:
            continue
        
        # Get all firms that have data on this date
        firms_on_date = df_date['gvkey'].unique()
        
        # Prepare arrays for vectorized simulation
        garch_params = {}
        for firm in firms_on_date:
            firm_data = df_date[df_date['gvkey'] == firm].iloc[0]
            garch_params[firm] = {
                'omega': max(firm_data.get('garch_omega', 1e-6), 1e-8),
                'alpha': max(firm_data.get('garch_alpha', 0.05), 1e-4),
                'beta': max(firm_data.get('garch_beta', 0.93), 0.0),
                'sigma': max(firm_data.get('garch_volatility', 0.2), 1e-4),
                'return': firm_data.get('return', 0.0),
                'nu': firm_data.get('garch_nu', 30.0)  # Degrees of freedom for t-dist
            }
        
        # Reorder firms consistently
        firms_on_date = sorted(list(firms_on_date))
        
        # Prepare arrays
        omega_arr = np.array([garch_params[f]['omega'] for f in firms_on_date])
        alpha_arr = np.array([garch_params[f]['alpha'] for f in firms_on_date])
        beta_arr = np.array([garch_params[f]['beta'] for f in firms_on_date])
        sigma_arr = np.array([garch_params[f]['sigma'] for f in firms_on_date])
        nu_arr = np.array([garch_params[f]['nu'] for f in firms_on_date])
        
        # Run Monte Carlo simulation with t-distribution innovations
        daily_vols = simulate_garch_paths_t_dist(
            omega_arr, alpha_arr, beta_arr, sigma_arr, nu_arr,
            num_simulations, num_days, len(firms_on_date)
        )
        
        # For each firm, calculate:
        # 1. Mean variance path (mean across simulations for each day)
        # 2. Sum of mean variances over 252 days (Integrated Variance)
        for firm_idx, firm in enumerate(firms_on_date):
            # daily_vols shape: (num_days, num_simulations, num_firms)
            # Get volatilities for this firm across all days and simulations
            firm_daily_vols = daily_vols[:, :, firm_idx]  # shape: (num_days, num_simulations)
            
            # MEAN VARIANCE CALCULATION (Corrected per request)
            # 1. Square the volatilities to get variances
            firm_daily_variances = firm_daily_vols ** 2
            
            # 2. Average variances across simulations (Expected Conditional Variance)
            # \bar{\sigma}^2_{t+h} = \frac{1}{M} \sum \sigma^{2,(m)}_{t+h}
            mean_variance_path = np.mean(firm_daily_variances, axis=1)  # shape: (num_days,)
            
            # 3. Sum expected variances over horizon (Integrated Variance)
            # IV_{t,T} = \sum \bar{\sigma}^2_{t+h}
            integrated_variance = np.sum(mean_variance_path)
            
            # Also store other statistics
            results_list.append({
                'gvkey': firm,
                'date': date,
                'mc_garch_integrated_variance': integrated_variance,
                'mc_garch_mean_daily_volatility': np.mean(mean_variance_path) ** 0.5, # approx
                'mc_garch_std_daily_volatility': np.std(firm_daily_vols),
                'mc_garch_min_daily_volatility': np.min(firm_daily_vols),
                'mc_garch_max_daily_volatility': np.max(firm_daily_vols),
            })
            rows_created += 1
        
        dates_processed += 1
    
    results_df = pd.DataFrame(results_list)
    
    print(f"\n{'='*80}")
    print("MONTE CARLO FORECAST COMPLETE")
    print(f"{'='*80}\n")
    
    print(f"Dates processed: {dates_processed:,}")
    print(f"Rows created: {rows_created:,}")
    print(f"Total output rows: {len(results_df):,}")
    print(f"Unique firms: {results_df['gvkey'].nunique()}")
    print(f"Unique dates: {results_df['date'].nunique()}")
    print(f"Date range: {results_df['date'].min().strftime('%Y-%m-%d')} to {results_df['date'].max().strftime('%Y-%m-%d')}")
    
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
        print(f"    Integrated Variance: {firm_data['mc_garch_integrated_variance'].mean():.4f} ± {firm_data['mc_garch_integrated_variance'].std():.4f}")
        print(f"    Implied Annual Vol:  {np.mean(np.sqrt(firm_data['mc_garch_integrated_variance'])):.4f}")
        print(f"    Range (IV):          [{firm_data['mc_garch_integrated_variance'].min():.4f}, {firm_data['mc_garch_integrated_variance'].max():.4f}]")
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Rows per second: {len(results_df) / total_time:.0f}")
    print(f"{'='*80}\n")
    
    return results_df


def _process_single_date_garch_mc(date_data, num_simulations, num_days):
    """
    Process Monte Carlo GARCH simulation for a single date (for parallelization).
    
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
    garch_params = {}
    for firm in firms_on_date:
        firm_data = df_date[df_date['gvkey'] == firm].iloc[0]
        garch_params[firm] = {
            'omega': max(firm_data.get('garch_omega', 1e-6), 1e-8),
            'alpha': max(firm_data.get('garch_alpha', 0.05), 1e-4),
            'beta': max(firm_data.get('garch_beta', 0.93), 0.0),
            'sigma': max(firm_data.get('garch_volatility', 0.2), 1e-4),
            'return': firm_data.get('return', 0.0),
            'nu': firm_data.get('garch_nu', 30.0)  # Degrees of freedom for t-dist
        }
    
    # Reorder firms consistently
    firms_on_date = sorted(list(firms_on_date))
    
    # Prepare arrays
    omega_arr = np.array([garch_params[f]['omega'] for f in firms_on_date])
    alpha_arr = np.array([garch_params[f]['alpha'] for f in firms_on_date])
    beta_arr = np.array([garch_params[f]['beta'] for f in firms_on_date])
    sigma_arr = np.array([garch_params[f]['sigma'] for f in firms_on_date])
    nu_arr = np.array([garch_params[f]['nu'] for f in firms_on_date])
    
    # Run Monte Carlo simulation with t-distribution innovations
    daily_vols = simulate_garch_paths_t_dist(
        omega_arr, alpha_arr, beta_arr, sigma_arr, nu_arr,
        num_simulations, num_days, len(firms_on_date)
    )
    
    # For each firm, calculate statistics
    for firm_idx, firm in enumerate(firms_on_date):
        firm_daily_vols = daily_vols[:, :, firm_idx]  # shape: (num_days, num_simulations)
        mean_path = np.mean(firm_daily_vols, axis=1)  # shape: (num_days,)
        cumulative_volatility = np.sum(mean_path)
        
        results_list.append({
            'gvkey': firm,
            'date': date,
            'mc_garch_cumulative_volatility': cumulative_volatility,
            'mc_garch_mean_daily_volatility': np.mean(mean_path),
            'mc_garch_std_daily_volatility': np.std(mean_path),
            'mc_garch_max_daily_volatility': np.max(mean_path),
            'mc_garch_min_daily_volatility': np.min(mean_path),
            'mc_garch_p95_daily_volatility': np.percentile(mean_path, 95),
            'mc_garch_p05_daily_volatility': np.percentile(mean_path, 5)
        })
    
    return results_list


def monte_carlo_garch_1year_parallel(garch_file, gvkey_selected=None, num_simulations=1000, num_days=252, n_jobs=-1):
    """
    Parallelized Monte Carlo GARCH forecast for 1 year (252 trading days).
    
    This version processes different dates in parallel for significant speedup.
    
    Parameters:
    -----------
    garch_file : str
        CSV file with GARCH parameters and volatilities
    gvkey_selected : list or None
        List of gvkeys to process, or None for all firms
    num_simulations : int
        Number of Monte Carlo paths per firm per date
    num_days : int
        Forecast horizon in trading days (252 = 1 year)
    n_jobs : int
        Number of parallel jobs (-1 = use all cores)
        
    Returns:
    --------
    pd.DataFrame
        Results with Monte Carlo volatility statistics per firm per date
    """
    print(f"Loading GARCH data from {garch_file}...")
    df = pd.read_csv(garch_file)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter firms if specified
    if gvkey_selected is not None:
        df = df[df['gvkey'].isin(gvkey_selected)]
    
    print(f"Running PARALLELIZED Monte Carlo GARCH simulation:")
    print(f"  Firms: {df['gvkey'].nunique()}")
    print(f"  Dates: {df['date'].nunique() if 'date' in df.columns else 1}")
    print(f"  Simulations per firm: {num_simulations:,}")
    print(f"  Forecast horizon: {num_days} days")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Innovation distribution: Student's t")
    
    start_time = pd.Timestamp.now()
    
    # Prepare date groups for parallel processing
    if 'date' in df.columns:
        date_groups = [(date, group) for date, group in df.groupby('date')]
    else:
        # Single date case
        date_groups = [(pd.Timestamp.now().date(), df)]
    
    print(f"\nProcessing {len(date_groups)} dates in parallel...")
    
    # Parallel processing across dates
    results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_date_garch_mc)(date_data, num_simulations, num_days) 
        for date_data in date_groups
    )
    
    # Flatten results
    results_list = []
    for date_results in results_nested:
        results_list.extend(date_results)
    
    results_df = pd.DataFrame(results_list)
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"PARALLELIZED MONTE CARLO GARCH COMPLETE")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Rows per second: {len(results_df) / total_time:.0f}")
    print(f"Speedup: ~{n_jobs}x expected on {n_jobs}-core system")
    print(f"{'='*80}\n")
    
    return results_df