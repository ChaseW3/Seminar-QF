# monte_carlo_garch.py

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba
from scipy import stats
from joblib import Parallel, delayed

# Cache directory removed


# =============================================================================
# JIT-COMPILED T-DISTRIBUTION SAMPLER (FAST!)
# =============================================================================

@numba.jit(nopython=True)
def sample_standardized_t(nu):
    """
    Generate a standardized t-distributed random variable using Numba.
    
    Uses the representation: t(ν) = Z / sqrt(V/ν) where Z ~ N(0,1), V ~ χ²(ν)
    Then standardizes so variance = 1: multiply by sqrt((ν-2)/ν)
    
    This is ~100x faster than scipy.stats.t.rvs() in loops!
    """
    if nu <= 2:
        return np.random.standard_normal()
    
    # Generate Z ~ N(0,1)
    z = np.random.standard_normal()
    
    # Generate V ~ chi-squared(nu) using sum of squared normals
    # For integer nu, chi2(nu) = sum of nu standard normal^2
    v = 0.0
    nu_int = int(nu)
    for _ in range(nu_int):
        v += np.random.standard_normal() ** 2
    
    # t = Z / sqrt(V/nu)
    t_sample = z / np.sqrt(v / nu)
    
    # Standardize so variance = 1 (t(nu) has variance nu/(nu-2))
    scale = np.sqrt((nu - 2) / nu)
    return t_sample * scale


@numba.jit(nopython=True, parallel=True)
def simulate_garch_paths_t_jit(omega, alpha, beta, sigma_t, nu,
                               num_simulations, num_days, num_firms):
    """
    JIT-compiled GARCH Monte Carlo with t-distributed innovations.
    
    This is ~100x faster than the scipy version!
    """
    daily_volatilities = np.zeros((num_days, num_simulations, num_firms))
    
    for sim in numba.prange(num_simulations):
        sigma = sigma_t.copy()
        
        for day in range(num_days):
            # Generate t-distributed innovations for each firm
            z = np.zeros(num_firms)
            for f in range(num_firms):
                z_raw = sample_standardized_t(nu[f])
                # Cap extreme innovations to prevent temporary spikes (consistent with MS-GARCH)
                z[f] = max(min(z_raw, 5.0), -5.0)  # Cap at ±5 sigma
            
            # STORE FIRST: sigma is the volatility used to generate THIS day's return
            daily_volatilities[day, sim, :] = sigma
            
            r = sigma * z
            sigma_squared = omega + alpha * (r ** 2) + beta * (sigma ** 2)
            sigma = np.sqrt(np.maximum(sigma_squared, 1e-6))
    
    return daily_volatilities


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
            z_raw = np.random.standard_normal(num_firms)
            # Cap extreme innovations to prevent temporary spikes (consistent with MS-GARCH)
            z = np.maximum(np.minimum(z_raw, 5.0), -5.0)  # Cap at ±5 sigma
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
    
    NOW USES THE FAST JIT-COMPILED VERSION INTERNALLY!
    
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
    # Use the fast JIT-compiled version
    return simulate_garch_paths_t_jit(omega, alpha, beta, sigma_t, nu,
                                      num_simulations, num_days, num_firms)


@numba.jit(nopython=True, parallel=True)
def simulate_garch_paths_t_jit_fully_optimized(omega, alpha, beta, sigma_t, nu,
                                               num_simulations, num_firms,
                                               horizon_days, v0_arr, liability_arr):
    """
    FULLY OPTIMIZED JIT-compiled GARCH Monte Carlo with t-distributed innovations.
    
    Calculates EVERYTHING in one pass without storing full paths:
    - Terminal volatilities at 1y/3y/5y  
    - Volatility statistics (mean/std/max/min)
    - Path-dependent default probabilities
    - Terminal asset values for CDS pricing
    
    Memory usage: ~400x less than naive approach!
    
    Parameters:
    -----------
    horizon_days : array of ints, shape (n_horizons,)
        Time points to store values (e.g., [252, 756, 1260] for 1y, 3y, 5y)
    v0_arr : array (num_firms,)
        Initial asset values (from Merton model)
    liability_arr : array (num_firms,)
        Liability thresholds for default
    
    Returns:
    --------
    terminal_vols : array (n_horizons, num_simulations, num_firms)
        Volatilities at each horizon
    terminal_assets : array (n_horizons, num_simulations, num_firms)
        Asset values at each horizon (for CDS pricing)
    default_indicators : array (n_horizons, num_simulations, num_firms)
        1 if defaulted by this horizon, 0 otherwise
    vol_means, vol_stds, vol_maxs, vol_mins : arrays (num_simulations, num_firms)
        Volatility statistics across all days
    """
    max_days = np.max(horizon_days)
    n_horizons = len(horizon_days)
    
    # Terminal values at horizons only
    terminal_vols = np.zeros((n_horizons, num_simulations, num_firms))
    terminal_assets = np.zeros((n_horizons, num_simulations, num_firms))
    default_indicators = np.zeros((n_horizons, num_simulations, num_firms))
    
    # Statistics accumulators
    vol_means = np.zeros((num_simulations, num_firms))
    vol_stds_accum = np.zeros((num_simulations, num_firms))
    vol_maxs = np.zeros((num_simulations, num_firms))
    vol_mins = np.zeros((num_simulations, num_firms))
    
    for sim in numba.prange(num_simulations):
        sigma = sigma_t.copy()
        asset_value = v0_arr.copy()
        defaulted = np.zeros(num_firms)
        
        # Initialize statistics
        for firm in range(num_firms):
            vol_maxs[sim, firm] = sigma[firm]
            vol_mins[sim, firm] = sigma[firm]
        
        for day in range(max_days):
            for firm in range(num_firms):
                # Generate t-distributed shock (capped to prevent extreme spikes)
                eps_raw = sample_standardized_t(nu[firm])
                eps = max(min(eps_raw, 5.0), -5.0)
                
                # Update running vol statistics (Welford's algorithm)
                old_mean = vol_means[sim, firm]
                vol_means[sim, firm] += (sigma[firm] - old_mean) / (day + 1)
                vol_stds_accum[sim, firm] += (sigma[firm] - old_mean) * (sigma[firm] - vol_means[sim, firm])
                
                if sigma[firm] > vol_maxs[sim, firm]:
                    vol_maxs[sim, firm] = sigma[firm]
                if sigma[firm] < vol_mins[sim, firm]:
                    vol_mins[sim, firm] = sigma[firm]
                
                # Simulate asset return: dV/V = sigma * eps (log price process)
                log_return = sigma[firm] * eps
                asset_value[firm] *= np.exp(log_return)
                
                # Check for default (path-dependent - default if EVER drop below liability)
                if asset_value[firm] < liability_arr[firm]:
                    defaulted[firm] = 1.0
                
                # Store terminal values at horizons
                for h_idx in range(n_horizons):
                    if day == horizon_days[h_idx] - 1:
                        terminal_vols[h_idx, sim, firm] = sigma[firm]
                        terminal_assets[h_idx, sim, firm] = asset_value[firm]
                        default_indicators[h_idx, sim, firm] = defaulted[firm]
                
                # Update GARCH volatility for next period
                sigma[firm] = np.sqrt(omega[firm] + alpha[firm] * (eps * sigma[firm])**2 + beta[firm] * sigma[firm]**2)
                sigma[firm] = max(sigma[firm], 1e-6)  # Ensure positivity
        
        # Finalize standard deviation calculation
        for firm in range(num_firms):
            vol_stds_accum[sim, firm] = np.sqrt(vol_stds_accum[sim, firm] / max_days)
    
    return terminal_vols, terminal_assets, default_indicators, vol_means, vol_stds_accum, vol_maxs, vol_mins


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

    # Load Merton Data for PD calculation and Method 1 Pricing
    merton_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(garch_file))), 'data', 'output', 'merged_data_with_merton.csv')
    if not os.path.exists(merton_file):
         merton_file = './data/output/merged_data_with_merton.csv'

    merton_by_date = {}
    if os.path.exists(merton_file):
        try:
            df_merton = pd.read_csv(merton_file)
            df_merton['date'] = pd.to_datetime(df_merton['date'])
            merton_by_date = {k: v for k, v in df_merton.groupby('date')}
            print(f"✓ Loaded Merton data for Debt Pricing ({len(df_merton):,} rows)")
        except Exception as e:
            print(f"⚠ Error loading Merton data: {e}")
            merton_by_date = {}
    
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
        
        # For each firm, calculate yearly variance using Asset Value Simulation
        # Simulate returns R ~ N(0, sigma) and measure variance of yearly return
        for firm_idx, firm in enumerate(firms_on_date):
            # daily_vols shape: (num_days, num_simulations, num_firms)
            # Get volatilities for this firm across all days and simulations
            firm_daily_vols = daily_vols[:, :, firm_idx]  # shape: (num_days, num_simulations)
            
            # YEARLY VARIANCE CALCULATION (Asset Value Simulation Method)
            # 1. Generate random innovations (standard normal) for Asset Returns
            z_innovations = np.random.standard_normal(firm_daily_vols.shape)
            
            # 2. Daily log returns: r_t ~ N(0, sigma_t)
            firm_daily_returns = firm_daily_vols * z_innovations
            
            # 3. Cumulative yearly return
            # Using Log Returns logic: V_T = V_0 * exp(sum(r_t))
            cum_log_returns = np.sum(firm_daily_returns, axis=0)
            
            # 4. Variance of cumulative log returns (strictly correct for log-return models)
            integrated_variance = np.var(cum_log_returns, ddof=1)
            
            # ---- METHOD 1: MC PRICING OF RISKY DEBT ----
            mc_spread = np.nan
            mc_debt_value = np.nan
            
            # Look up merton data
            firm_merton_data = None
            if date in merton_by_date:
                df_md = merton_by_date[date]
                fr = df_md[df_md['gvkey'] == firm]
                if not fr.empty:
                    firm_merton_data = fr.iloc[0]
            
            if firm_merton_data is not None:
                asset_value = firm_merton_data.get('asset_value', np.nan)
                debt_face = firm_merton_data.get('liabilities_total', np.nan)
                rf_rate = firm_merton_data.get('risk_free_rate', np.nan)
                
                if not np.isnan(asset_value) and not np.isnan(debt_face):
                     # Construct V_T paths
                     # V_T = V_0 * exp(cum_log_returns)
                     V_T_paths = asset_value * np.exp(cum_log_returns)
                     
                     payoffs = np.minimum(V_T_paths, debt_face)
                     expected_payoff = np.mean(payoffs)
                     
                     T_years = num_days / 252.0
                     discount_factor = np.exp(-rf_rate * T_years)
                     risky_debt_value = discount_factor * expected_payoff
                     
                     if risky_debt_value > 0 and debt_face > 0:
                         implied_yield = -1.0/T_years * np.log(risky_debt_value / debt_face)
                         mc_spread = implied_yield - rf_rate
                         mc_debt_value = risky_debt_value

            # Also store other statistics
            results_list.append({
                'gvkey': firm,
                'date': date,
                'mc_garch_integrated_variance': integrated_variance,
                'mc_garch_implied_spread': mc_spread,
                'mc_garch_debt_value': mc_debt_value,
                'mc_garch_mean_daily_volatility': np.mean(firm_daily_vols ** 2) ** 0.5, # RMS volatility
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
    OPTIMIZED Process Monte Carlo GARCH simulation for a single date.
    
    Now only stores terminal values at 1y/3y/5y horizons instead of all daily values!
    Memory usage reduced by ~400x.
    
    Parameters:
    -----------
    date_data : tuple
        (date, df_date, merton_data_dict) 
    num_simulations : int
        Number of Monte Carlo paths
    num_days : int
        Forecast horizon in days (should be max of horizons, e.g., 1260 for 5y)
        
    Returns:
    --------
    list : Results for all firms on this date
    """
    # Unpack with optional merton_data_dict
    if len(date_data) == 3:
        date, df_date, merton_data_dict = date_data
    else:
        date, df_date = date_data
        merton_data_dict = {}

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
    
    # Prepare Merton model parameters for optimized simulation
    v0_arr = np.zeros(len(firms_on_date))
    liability_arr = np.zeros(len(firms_on_date))
    has_merton = {}
    
    for firm_idx, firm in enumerate(firms_on_date):
        if firm in merton_data_dict:
            m_data = merton_data_dict[firm]
            v0 = m_data.get('asset_value', 0.0)
            liability = m_data.get('liabilities_total', 0.0)
            if not np.isnan(v0) and not np.isnan(liability) and v0 > 0:
                v0_arr[firm_idx] = v0
                liability_arr[firm_idx] = liability
                has_merton[firm] = m_data
    
    # Define horizons: 1y, 3y, 5y (trading days)
    horizon_days = np.array([252, 756, 1260], dtype=np.int32)
    
    # Run FULLY OPTIMIZED simulation - calculates EVERYTHING in one pass!
    terminal_vols, terminal_assets, default_indicators, vol_means, vol_stds, vol_maxs, vol_mins = \
        simulate_garch_paths_t_jit_fully_optimized(
            omega_arr, alpha_arr, beta_arr, sigma_arr, nu_arr,
            num_simulations, len(firms_on_date), horizon_days,
            v0_arr, liability_arr
        )
    # terminal_vols, terminal_assets, default_indicators shape: (3, num_simulations, num_firms) for 1y/3y/5y
    # vol_means, vol_stds, vol_maxs, vol_mins shape: (num_simulations, num_firms)
    
    # For each firm, calculate statistics
    for firm_idx, firm in enumerate(firms_on_date):
        # Extract terminal values at each horizon for this firm
        firm_vols_1y = terminal_vols[0, :, firm_idx]  # shape: (num_simulations,)
        firm_vols_3y = terminal_vols[1, :, firm_idx]
        firm_vols_5y = terminal_vols[2, :, firm_idx]
        
        firm_assets_1y = terminal_assets[0, :, firm_idx]
        firm_assets_3y = terminal_assets[1, :, firm_idx]
        firm_assets_5y = terminal_assets[2, :, firm_idx]
        
        firm_defaulted_1y = default_indicators[0, :, firm_idx]
        firm_defaulted_3y = default_indicators[1, :, firm_idx]
        firm_defaulted_5y = default_indicators[2, :, firm_idx]
        
        # Get summary statistics for this firm
        firm_vol_mean = np.mean(vol_means[:, firm_idx])
        firm_vol_std = np.mean(vol_stds[:, firm_idx])
        firm_vol_max = np.max(vol_maxs[:, firm_idx])
        firm_vol_min = np.min(vol_mins[:, firm_idx])
        
        # INTEGRATED VARIANCE CALCULATION using terminal volatilities
        # For 5-year horizon: use terminal values at 1260 days
        integrated_variance = np.var(firm_vols_5y, ddof=1)
        
        # Calculate PD and CDS spreads from pre-computed values
        pd_value_1y = np.mean(firm_defaulted_1y)
        pd_value_3y = np.mean(firm_defaulted_3y)
        pd_value_5y = np.mean(firm_defaulted_5y)
        pd_value = pd_value_1y  # For backward compatibility
        
        mc_spread_1y, mc_spread_3y, mc_spread_5y = np.nan, np.nan, np.nan
        mc_debt_1y, mc_debt_3y, mc_debt_5y = np.nan, np.nan, np.nan
        
        if firm in has_merton:
            m_data = has_merton[firm]
            liability = m_data.get('liabilities_total', np.nan)
            rf_rate = m_data.get('risk_free_rate', np.nan)
            
            # Adjust units if needed (ensure decimal)
            if not np.isnan(rf_rate) and abs(rf_rate) > 0.5:
                rf_rate = rf_rate / 100.0
            
            if not np.isnan(liability) and not np.isnan(rf_rate) and liability > 0:
                # 1-Year CDS Spread
                expected_payoff_1y = np.mean(np.minimum(firm_assets_1y, liability))
                T_years = 1.0
                discount_factor = np.exp(-rf_rate * T_years)
                debt_val_1y = expected_payoff_1y * discount_factor
                if debt_val_1y > 0 and liability > 0:
                    ytm = -np.log(debt_val_1y / liability) / T_years
                    mc_spread_1y = max(ytm - rf_rate, 0.0)
                    mc_debt_1y = debt_val_1y
                
                # 3-Year CDS Spread
                expected_payoff_3y = np.mean(np.minimum(firm_assets_3y, liability))
                T_years = 3.0
                discount_factor = np.exp(-rf_rate * T_years)
                debt_val_3y = expected_payoff_3y * discount_factor
                if debt_val_3y > 0 and liability > 0:
                    ytm = -np.log(debt_val_3y / liability) / T_years
                    mc_spread_3y = max(ytm - rf_rate, 0.0)
                    mc_debt_3y = debt_val_3y
                
                # 5-Year CDS Spread
                expected_payoff_5y = np.mean(np.minimum(firm_assets_5y, liability))
                T_years = 5.0
                discount_factor = np.exp(-rf_rate * T_years)
                debt_val_5y = expected_payoff_5y * discount_factor
                if debt_val_5y > 0 and liability > 0:
                    ytm = -np.log(debt_val_5y / liability) / T_years
                    mc_spread_5y = max(ytm - rf_rate, 0.0)
                    mc_debt_5y = debt_val_5y
        
        # Calculate percentiles from terminal values
        firm_p95 = np.percentile(firm_vols_5y, 95)
        firm_p05 = np.percentile(firm_vols_5y, 5)
        
        results_list.append({
            'gvkey': firm,
            'date': date,
            'mc_garch_integrated_variance': integrated_variance,
            'mc_garch_mean_daily_volatility': firm_vol_mean,
            'mc_garch_std_daily_volatility': firm_vol_std,
            'mc_garch_max_daily_volatility': firm_vol_max,
            'mc_garch_min_daily_volatility': firm_vol_min,
            'mc_garch_p95_daily_volatility': firm_p95,
            'mc_garch_p05_daily_volatility': firm_p05,
            'mc_garch_pd_1y': pd_value_1y,
            'mc_garch_pd_3y': pd_value_3y,
            'mc_garch_pd_5y': pd_value_5y,
            'mc_garch_implied_spread_1y': mc_spread_1y,
            'mc_garch_implied_spread_3y': mc_spread_3y,
            'mc_garch_implied_spread_5y': mc_spread_5y,
            'mc_garch_debt_value_1y': mc_debt_1y,
            'mc_garch_debt_value_3y': mc_debt_3y,
            'mc_garch_debt_value_5y': mc_debt_5y,
        })
    
    return results_list


def monte_carlo_garch_1year_parallel(garch_file, gvkey_selected=None, num_simulations=1000, num_days=1260, n_jobs=-1, merton_df=None):
    """
    Parallelized Monte Carlo GARCH forecast for 1-5 years (default 5 years/1260 days).
    
    This version processes different dates in parallel for significant speedup.
    
    Parameters:
    -----------
    garch_file : str or pd.DataFrame
        CSV file path OR DataFrame with GARCH parameters and volatilities
    gvkey_selected : list or None
        List of gvkeys to process, or None for all firms
    num_simulations : int
        Number of Monte Carlo paths per firm per date
    num_days : int
        Forecast horizon in trading days (default 1260 = 5 years)
    n_jobs : int
        Number of parallel jobs (-1 = use all cores)
    merton_df : pd.DataFrame, optional
        Pre-loaded Merton model data (asset values, liabilities) to avoid reloading
        
    Returns:
    --------
    pd.DataFrame
        Results with Monte Carlo volatility statistics per firm per date
    """
    if isinstance(garch_file, str):
        print(f"Loading GARCH data from {garch_file}...")
        df = pd.read_csv(garch_file)
    else:
        print(f"Using provided GARCH DataFrame...")
        df = garch_file.copy()
    
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
    
    # Load Merton Data for PD calculation
    merton_by_date = {}
    
    if merton_df is not None:
        print(f"✓ Using provided Merton data for PD calculation ({len(merton_df):,} rows)")
        if 'date' in merton_df.columns:
            # Ensure datetime
            if not pd.api.types.is_datetime64_any_dtype(merton_df['date']):
                 merton_df = merton_df.copy()
                 merton_df['date'] = pd.to_datetime(merton_df['date'])
            
            # Pre-group by date for faster access
            merton_by_date = {k: v for k, v in merton_df.groupby('date')}
    else:
        merton_file = None
        # Try to find the file in several locations
        potential_paths = []
        
        if isinstance(garch_file, str):
            # Same directory as input file
            potential_paths.append(os.path.join(os.path.dirname(garch_file), 'merged_data_with_merton.csv'))
            # Common project structure paths
            potential_paths.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(garch_file))), 'data', 'output', 'merged_data_with_merton.csv'))
        
        # Add standard relative paths
        potential_paths.append('./data/output/merged_data_with_merton.csv')
        potential_paths.append('./Seminar QF/data/output/merged_data_with_merton.csv')
        potential_paths.append('../data/output/merged_data_with_merton.csv')
        
        for path in potential_paths:
            if os.path.exists(path):
                merton_file = path
                break
            
        if merton_file and os.path.exists(merton_file):
            try:
                df_merton = pd.read_csv(merton_file)
                df_merton['date'] = pd.to_datetime(df_merton['date'])
                merton_by_date = {k: v for k, v in df_merton.groupby('date')}
                print(f"✓ Loaded Merton data for PD calculation ({len(df_merton):,} rows) from {merton_file}")
            except Exception as e:
                print(f"⚠ Error loading Merton data: {e}")
                merton_by_date = {}
        else:
             print(f"⚠ Warning: Merton data file not found (checked {len(potential_paths)} locations). PD will be NaN.")
             merton_by_date = {}

    date_groups = []
    if 'date' in df.columns:
        for date, group in df.groupby('date'):
            # Prepare merton dict
            merton_date_dict = {}
            if date in merton_by_date:
                df_m = merton_by_date[date]
                firms_on_date = group['gvkey'].unique()
                df_m_relevant = df_m[df_m['gvkey'].isin(firms_on_date)]
                for _, row in df_m_relevant.iterrows():
                    merton_date_dict[row['gvkey']] = {
                        'asset_value': row['asset_value'],
                        'liabilities_total': row['liabilities_total'],
                        'risk_free_rate': row['risk_free_rate']
                    }
            date_groups.append((date, group, merton_date_dict))
    else:
        # Single date case
        date_groups = [(pd.Timestamp.now().date(), df, {})]
    
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
