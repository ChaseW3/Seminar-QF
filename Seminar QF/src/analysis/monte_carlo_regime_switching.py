
# monte_carlo_regime_switching.py
# Efficient regime-switching Monte Carlo with vectorization AND t-distribution

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba
from joblib import Parallel, delayed

os.makedirs('./intermediates/', exist_ok=True)

# =============================================================================
# JIT-COMPILED T-DISTRIBUTION SAMPLER
# =============================================================================

@numba.jit(nopython=True)
def sample_standardized_t(nu):
    """
    Generate a standardized t-distributed random variable using Numba.
    
    Uses the representation: t(ν) = Z / sqrt(V/ν) where Z ~ N(0,1), V ~ χ²(ν)
    Then standardizes so variance = 1: multiply by sqrt((ν-2)/ν)
    """
    if nu <= 2:
        return np.random.standard_normal()
    
    z = np.random.standard_normal()
    v = 0.0
    nu_int = int(nu)
    for _ in range(nu_int):
        v += np.random.standard_normal() ** 2
    
    t_sample = z / np.sqrt(v / nu)
    scale = np.sqrt((nu - 2) / nu)
    return t_sample * scale

@numba.jit(nopython=True, parallel=True)
def simulate_regime_switching_paths_vectorized(
    mu_0, mu_1, ar_0, ar_1, sigma_0, sigma_1, nu_0, nu_1,
    trans_00, trans_01, trans_10, trans_11,
    num_simulations, num_days, num_firms, initial_regime_probs
):
    """
    Vectorized regime-switching Monte Carlo simulation with T-DISTRIBUTION.
    """
    
    daily_volatilities = np.zeros((num_days, num_simulations, num_firms))
    daily_returns = np.zeros((num_days, num_simulations, num_firms))
    regime_paths = np.zeros((num_days, num_simulations, num_firms), dtype=np.int32)
    
    for sim in numba.prange(num_simulations):
        # Initialize regime (0 or 1)
        regime = np.zeros(num_firms, dtype=np.int32)
        for f in range(num_firms):
            if np.random.uniform(0.0, 1.0) < initial_regime_probs[f]:
                regime[f] = 1
            else:
                regime[f] = 0
        
        r_prev = np.zeros(num_firms)
        
        for day in range(num_days):
            # Regime transitions
            for f in range(num_firms):
                if regime[f] == 0:
                    if np.random.uniform(0.0, 1.0) < trans_01[f]:
                        regime[f] = 1
                else:
                    if np.random.uniform(0.0, 1.0) < trans_10[f]:
                        regime[f] = 0
            
            for f in range(num_firms):
                if regime[f] == 0:
                    mu = mu_0[f]
                    ar = ar_0[f]
                    sigma = sigma_0[f]
                    nu = nu_0[f]
                else:
                    mu = mu_1[f]
                    ar = ar_1[f]
                    sigma = sigma_1[f]
                    nu = nu_1[f]
                    
                # Generate t-distributed innovation
                z = sample_standardized_t(nu)
                # Cap innovations
                z = max(min(z, 5.0), -5.0)
                
                # AR(1) return
                r_curr = mu + ar * r_prev[f] + sigma * z
                r_prev[f] = r_curr
                
                daily_volatilities[day, sim, f] = sigma
                daily_returns[day, sim, f] = r_curr
            
            regime_paths[day, sim, :] = regime
    
    return daily_volatilities, regime_paths, daily_returns


def monte_carlo_regime_switching_1year(
    garch_file, 
    regime_params_file,
    gvkey_selected=None, 
    num_simulations=1000, 
    num_days=252
):
    """
    Run Monte Carlo regime-switching forecast for 1 year (252 days) with T-DISTRIBUTION.
    """
    
    start_time = pd.Timestamp.now()
    
    print(f"\n{'='*80}")
    print("MONTE CARLO REGIME-SWITCHING (T-DIST) 1-YEAR VOLATILITY FORECAST")
    print(f"{'='*80}\n")
    
    # Load regime parameters
    try:
        regime_params = pd.read_csv(regime_params_file)
        # Check for new 'nu' columns from updated estimator
        if 'regime_0_nu' not in regime_params.columns:
            print("⚠ Warning: 'regime_0_nu' not found in parameters. Using default df=100 (Normal approx).")
            regime_params['regime_0_nu'] = 100.0
            regime_params['regime_1_nu'] = 100.0
            
        print(f"✓ Loaded regime-switching parameters for {len(regime_params)} firms\n")
    except FileNotFoundError:
        print(f"✗ File '{regime_params_file}' not found!")
        return pd.DataFrame()
    
    # Load GARCH base data
    df_garch = pd.read_csv(garch_file)
    df_garch['date'] = pd.to_datetime(df_garch['date'])
    df_garch = df_garch.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_garch):,} GARCH observations")
    
    unique_dates = sorted(df_garch['date'].unique())
    firm_list = sorted(df_garch['gvkey'].unique())
    
    if gvkey_selected is not None:
        firm_list = [gvkey_selected]
    
    results_list = []
    
    num_dates = len(unique_dates)
    
    # Process each date
    for date_idx, date in enumerate(unique_dates):
        if date_idx % max(1, num_dates // 10) == 0:
            print(f"Progress: {date_idx + 1}/{num_dates} ({date.strftime('%Y-%m-%d')})")
        
        df_date = df_garch[df_garch['date'] == date]
        firms_on_date = sorted(df_date['gvkey'].unique())
        firms_on_date = [f for f in firms_on_date if f in firm_list]
        
        if len(firms_on_date) == 0:
            continue
        
        regime_data = regime_params[regime_params['gvkey'].isin(firms_on_date)].set_index('gvkey')
        
        if len(regime_data) == 0:
            continue
        
        # PREPARE ARRAYS
        mu_0_arr = np.array([regime_data.loc[f, 'regime_0_mean'] for f in firms_on_date])
        mu_1_arr = np.array([regime_data.loc[f, 'regime_1_mean'] for f in firms_on_date])
        ar_0_arr = np.array([regime_data.loc[f, 'regime_0_ar'] for f in firms_on_date])
        ar_1_arr = np.array([regime_data.loc[f, 'regime_1_ar'] for f in firms_on_date])
        sigma_0_arr = np.array([regime_data.loc[f, 'regime_0_vol'] for f in firms_on_date])
        sigma_1_arr = np.array([regime_data.loc[f, 'regime_1_vol'] for f in firms_on_date])
        nu_0_arr = np.array([regime_data.loc[f, 'regime_0_nu'] for f in firms_on_date])
        nu_1_arr = np.array([regime_data.loc[f, 'regime_1_nu'] for f in firms_on_date])
        
        trans_00_arr = np.array([regime_data.loc[f, 'transition_prob_00'] for f in firms_on_date])
        trans_01_arr = np.array([regime_data.loc[f, 'transition_prob_01'] for f in firms_on_date])
        trans_10_arr = np.array([regime_data.loc[f, 'transition_prob_10'] for f in firms_on_date])
        trans_11_arr = np.array([regime_data.loc[f, 'transition_prob_11'] for f in firms_on_date])
        
        initial_regime_probs = np.full(len(firms_on_date), 0.5)
        
        # RUN SIMULATION
        daily_vols, regime_paths, daily_returns = simulate_regime_switching_paths_vectorized(
            mu_0_arr, mu_1_arr, ar_0_arr, ar_1_arr, sigma_0_arr, sigma_1_arr, nu_0_arr, nu_1_arr,
            trans_00_arr, trans_01_arr, trans_10_arr, trans_11_arr,
            num_simulations, num_days, len(firms_on_date), initial_regime_probs
        )
        
        # CALCULATE STATISTICS
        # Use generated daily_returns which now include t-distributed shocks
        path_cumulative_returns = np.prod(1.0 + daily_returns, axis=0) - 1.0 
        integrated_variances = np.var(path_cumulative_returns, axis=0, ddof=1)
        
        mean_daily_vols = np.mean(daily_vols, axis=(0,1))
        mean_regime_0 = np.mean(regime_paths == 0, axis=(0, 1))
        mean_regime_1 = np.mean(regime_paths == 1, axis=(0, 1))
        
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
    results_df['year'] = results_df['date'].dt.year
    
    print(f"\n{'='*80}")
    print("REGIME-SWITCHING MONTE CARLO COMPLETE")
    print(f"{'='*80}\n")
    
    if hasattr(results_df, 'year'):
         for year in sorted(results_df['year'].unique()):
            print(f"  {year}: {len(results_df[results_df['year'] == year]):,} rows")
    
    return results_df
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
    # Unpack updated tuple
    if len(date_data) == 5:
        date, df_date, regime_params_dict, firms_on_date, merton_data_dict = date_data
    else:
        # Legacy support or fallback
        date, df_date, regime_params_dict, firms_on_date = date_data
        merton_data_dict = {}

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
    
    # YEARLY VARIANCE & PD CALCULATION (Asset Value Simulation Method)
    z_innovations = np.random.standard_normal(daily_vols.shape)
    assets_daily_returns = daily_vols * z_innovations
    
    # Cumulative returns: V_t = V_0 * prod(1+R)
    # Shape: (num_days, num_simulations, num_firms)
    cumulative_returns = np.cumprod(1.0 + assets_daily_returns, axis=0)
    
    path_cumulative_returns = cumulative_returns[-1, :, :] - 1.0
    integrated_variances = np.var(path_cumulative_returns, axis=0, ddof=1)
    
    # Other stats
    mean_daily_vols = np.mean(daily_vols, axis=(0, 1))
    mean_regime_0 = np.mean(regime_paths == 0, axis=(0, 1))
    mean_regime_1 = np.mean(regime_paths == 1, axis=(0, 1))
    
    # Append results for valid firms
    for f_idx, firm in valid_firms:
        # Calculate Probability of Default (PD)
        pd_value = np.nan
        
        if firm in merton_data_dict:
             m_data = merton_data_dict[firm]
             v0 = m_data.get('asset_value', np.nan)
             liability = m_data.get('liabilities_total', np.nan)
             
             if not np.isnan(v0) and not np.isnan(liability) and v0 > 0:
                 # Calculate asset paths: V_t = V_0 * cumulative_returns[t]
                 # cumulative_returns shape: (num_days, num_simulations, n_firms)
                 firm_cum_returns = cumulative_returns[:, :, f_idx]
                 
                 firm_asset_paths = v0 * firm_cum_returns
                 
                 # Check default condition (Asset < Liability) at any point
                 is_below_barrier = firm_asset_paths < liability
                 
                 # Did default happen at any time step?
                 path_defaulted = np.any(is_below_barrier, axis=0)
                 
                 pd_value = np.mean(path_defaulted)

        results_list.append({
            'gvkey': firm,
            'date': date,
            'rs_integrated_variance': integrated_variances[f_idx],
            'rs_mean_daily_volatility': mean_daily_vols[f_idx],
            'rs_fraction_regime_0': mean_regime_0[f_idx],
            'rs_fraction_regime_1': mean_regime_1[f_idx],
            'rs_probability_of_default': pd_value
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
    
    # Load Merton Data for PD calculation
    merton_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(garch_file))), 'merged_data_with_merton.csv')
    if not os.path.exists(merton_file):
        # Try relative to current script/workspace
        merton_file = './data/output/merged_data_with_merton.csv'
    
    merton_lookup = {}
    if os.path.exists(merton_file):
        try:
            df_merton = pd.read_csv(merton_file)
            df_merton['date'] = pd.to_datetime(df_merton['date'])
            # Create dict for fast lookup: (gvkey, date) -> {asset_value, liabilities}
            # Optimize: Group by date first to avoid huge dict?
            # Or just filter in loop. Let's create a full lookup if memory permits, or per date in loop.
            # Using per-date filtering in the loop below is better for parallel pickup
            
            # Pre-group by date for faster access in loop
            merton_by_date = {k: v for k, v in df_merton.groupby('date')}
            print(f"✓ Loaded Merton data for PD calculation ({len(df_merton):,} rows)")
        except Exception as e:
            print(f"⚠ Error loading Merton data: {e}")
            merton_by_date = {}
    else:
        print(f"⚠ Warning: {merton_file} not found. PD will be NaN.")
        merton_by_date = {}

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
        
        # Prepare merton dict for this date
        merton_date_dict = {}
        if date in merton_by_date:
            df_m = merton_by_date[date]
            # Create dict: gvkey -> {asset_value, liabilities_total}
            # Only for firms on this date to save space
            relevant_merton = df_m[df_m['gvkey'].isin(firms_on_date)]
            for _, row in relevant_merton.iterrows():
                merton_date_dict[row['gvkey']] = {
                    'asset_value': row['asset_value'],
                    'liabilities_total': row['liabilities_total']
                }
        
        date_groups.append((date, group, regime_params_dict, firms_on_date, merton_date_dict))
    
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
