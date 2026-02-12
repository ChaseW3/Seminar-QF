# monte_carlo_ms_garch.py
"""
Monte Carlo Simulation for Proper MS-GARCH(1,1) with t-Distribution
===================================================================

This module simulates future volatility paths using the MS-GARCH model where:
1. Volatility follows GARCH(1,1) dynamics within EACH regime
2. Regime transitions follow a Markov chain
3. Parameters differ between regimes
4. Innovations follow Student's t distribution to handle fat tails

Key difference from simple regime-switching MC:
- Volatility evolves with GARCH dynamics, not fixed per regime
- σ²_t = ω_k + α_k * ε²_{t-1} + β_k * σ²_{t-1} in regime k
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba
from joblib import Parallel, delayed

@numba.jit(nopython=True)
def sample_standardized_t(nu):
    """
    Generate a standardized t-distributed random variable using Numba.
    Optimized to use direct Chi-Square generation.
    """
    if nu >= 100:
        return np.random.standard_normal()
    if nu <= 2:
        return np.random.standard_normal()
    
    # Generate Z ~ N(0,1)
    z = np.random.standard_normal()
    
    # Generate V ~ chi-squared(nu)
    v = np.random.chisquare(nu)
    
    # t = Z / sqrt(V/nu)
    t_sample = z / np.sqrt(v / nu)
    
    # Standardize so variance = 1
    scale = np.sqrt((nu - 2) / nu)
    return t_sample * scale


@numba.jit(nopython=True, fastmath=True, cache=True)
def simulate_ms_garch_paths_t_jit_vectorized(
    omega_0, omega_1, alpha_0, alpha_1, beta_0, beta_1,
    mu_0, mu_1, p00, p11, nu_0, nu_1,
    initial_sigma2, initial_regime_probs,
    num_simulations, num_firms, horizon_days,
    v0_arr, liability_arr
):
    """
    FULLY OPTIMIZED JIT-compiled MS-GARCH Monte Carlo with t-distribution.
    Vectorized over simulations for SIMD efficiency.
    """
    max_days = np.max(horizon_days)
    n_horizons = len(horizon_days)
    
    # Output arrays: (Horizon, Firm, Sim)
    terminal_vols = np.zeros((n_horizons, num_firms, num_simulations))
    terminal_assets = np.zeros((n_horizons, num_firms, num_simulations))
    default_indicators = np.zeros((n_horizons, num_firms, num_simulations))
    
    # (Regime, Firm, Sim)
    regime_fractions = np.zeros((2, num_firms, num_simulations)) 
    
    # Statistics: (Firm, Sim)
    vol_means = np.zeros((num_firms, num_simulations))
    vol_stds = np.zeros((num_firms, num_simulations))
    vol_maxs = np.zeros((num_firms, num_simulations))
    vol_mins = np.zeros((num_firms, num_simulations))
    
    # Pre-compute horizon mask
    is_horizon = np.zeros(max_days, dtype=np.int32)
    horizon_map = np.full(max_days, -1, dtype=np.int32)
    for h in range(n_horizons):
        day_idx = horizon_days[h] - 1
        if day_idx < max_days:
            is_horizon[day_idx] = 1
            horizon_map[day_idx] = h

    # Process firms sequentially (outer parallel via Joblib)
    for f in range(num_firms):
        # Flatten firm parameters to scalars
        w0, a0, b0 = omega_0[f], alpha_0[f], beta_0[f]
        w1, a1, b1 = omega_1[f], alpha_1[f], beta_1[f]
        m0, m1 = mu_0[f], mu_1[f]
        n0, n1 = nu_0[f], nu_1[f]
        prob00, prob11 = p00[f], p11[f]
        
        init_s2 = initial_sigma2[f]
        prob_regime1 = initial_regime_probs[f]
        v0 = v0_arr[f]
        liability = liability_arr[f]

        # Initialize State Vectors for all Sims
        # Regime: 1 if random < prob, else 0
        regime_1_mask = np.random.random(num_simulations) < prob_regime1
        
        sigma2 = np.full(num_simulations, init_s2)
        sigma = np.sqrt(sigma2)
        eps2_prev = np.zeros(num_simulations)
        asset = np.full(num_simulations, v0)
        
        regime_0_cnt = np.zeros(num_simulations)
        regime_1_cnt = np.zeros(num_simulations)
        
        m_old = sigma.copy()
        s_old = np.zeros(num_simulations)
        curr_max = sigma.copy()
        curr_min = sigma.copy()
        
        # T-dist factors
        t_fact0 = np.sqrt((n0 - 2) / n0) if n0 > 2 else 1.0
        t_fact1 = np.sqrt((n1 - 2) / n1) if n1 > 2 else 1.0
            
        for day in range(max_days):
            # A. Regime Specifics & Updates
            regime_0_cnt += (~regime_1_mask)
            regime_1_cnt += regime_1_mask
            
            # Select parameters based on current regime
            w = np.where(regime_1_mask, w1, w0)
            a = np.where(regime_1_mask, a1, a0)
            b = np.where(regime_1_mask, b1, b0)
            mu = np.where(regime_1_mask, m1, m0)

            # B. GARCH Update (Variance Equation)
            if day > 0:
                sigma2 = w + a * eps2_prev + b * sigma2
            
            # Bounds check
            sigma2 = np.maximum(sigma2, 1e-6)
            sigma2 = np.minimum(sigma2, 0.01)
            
            sigma = np.sqrt(sigma2)

            # C. Stats Update (Welford)
            curr_max = np.maximum(curr_max, sigma)
            curr_min = np.minimum(curr_min, sigma)
            
            n = day + 1
            delta = sigma - m_old
            m_new = m_old + delta / n
            s_old += delta * (sigma - m_new)
            m_old = m_new

            # D. Innovation and Return
            z = np.random.standard_normal(num_simulations)
            
            # T-dist sampling
            # Approximate t-dist via normal if nu > 100
            # Else chi-square. 
            # Doing separate paths for vectorization is tricky without masking or generating both
            # Simple approach: generate both chi-squares, select one.
            n0_safe = max(n0, 1e-4) # Avoid nu=0
            n1_safe = max(n1, 1e-4)
            v0 = np.random.chisquare(n0_safe, num_simulations)
            v1 = np.random.chisquare(n1_safe, num_simulations)
            
            # Reduce risk of division by zero
            v0 = np.maximum(v0, 1e-12)
            v1 = np.maximum(v1, 1e-12)
            
            t0_sample = z / np.sqrt(v0 / n0_safe) * t_fact0
            t1_sample = z / np.sqrt(v1 / n1_safe) * t_fact1
            
            # If nu > 100, just use Z. Check scalar nu to avoid overhead?
            if n0 >= 100: t0_sample = z
            if n1 >= 100: t1_sample = z
            
            z_t = np.where(regime_1_mask, t1_sample, t0_sample)
            
            eps = sigma * z_t
            eps2_prev = eps * eps
            
            log_ret = mu + eps
            asset *= np.exp(log_ret)

            # E. Horizons
            if is_horizon[day]:
                h = horizon_map[day]
                terminal_vols[h, f, :] = sigma
                terminal_assets[h, f, :] = asset
                if liability > 0:
                    default_indicators[h, f, :] = (asset < liability).astype(np.float64)

            # F. Regime Transition
            # p01 = 1 - p00
            # p10 = 1 - p11
            u = np.random.random(num_simulations)
            
            # 0 -> 1 if u > p00 (original logic: if u > p00 -> switch)
            switch_to_1 = (~regime_1_mask) & (u > prob00)
            
            # 1 -> 0 if u > p11
            switch_to_0 = (regime_1_mask) & (u > prob11)
            
            regime_1_mask = (regime_1_mask | switch_to_1) & (~switch_to_0)
            
        # Finalize Stats
        vol_means[f, :] = m_old
        vol_stds[f, :] = np.sqrt(s_old / max_days)
        vol_maxs[f, :] = curr_max
        vol_mins[f, :] = curr_min
        
        regime_fractions[0, f, :] = regime_0_cnt / max_days
        regime_fractions[1, f, :] = regime_1_cnt / max_days
    
    return (terminal_vols, terminal_assets, default_indicators,
            vol_means, vol_stds, vol_maxs, vol_mins, regime_fractions)


def _process_single_date_msgarch_mc(date_data, num_simulations, num_days):
    """
    OPTIMIZED Process Monte Carlo MS-GARCH simulation for a single date.
    
    Now only stores terminal values at 1y/3y/5y horizons instead of all daily values!
    Memory usage reduced by ~400x.
    
    Parameters:
    -----------
    date_data : tuple
        (date, df_date, msgarch_params_dict, merton_data_dict) 
    num_simulations : int
        Number of Monte Carlo paths
    num_days : int
        Forecast horizon in days (should be max of horizons, e.g., 1260 for 5y)
        
    Returns:
    --------
    list : Results for all firms on this date
    """
    # Unpack with optional merton_data_dict
    if len(date_data) == 4:
        date, df_date, msgarch_params, merton_data_dict = date_data
    else:
        date, df_date, msgarch_params = date_data
        merton_data_dict = {}

    results_list = []
    
    if df_date.empty:
        return results_list
    
    firms_on_date = df_date['gvkey'].unique()
    
    for firm in firms_on_date:
        try:
            firm_data = df_date[df_date['gvkey'] == firm].iloc[0]
            
            if firm not in msgarch_params:
                continue
            
            params = msgarch_params[firm]
            
            # Get current volatility (initial condition)
            current_vol = max(firm_data.get('ms_garch_volatility', 0.2), 1e-4)
            initial_sigma2 = current_vol ** 2
            
            # Get current regime probability (or use stationary)  
            regime_prob = firm_data.get('ms_garch_regime_prob', 0.5)
            
            # Prepare Merton model parameters
            v0 = 0.0
            liability = 0.0
            rf_rate = np.nan
            
            if firm in merton_data_dict:
                m_data = merton_data_dict[firm]
                v0_raw = m_data.get('asset_value', 0.0)
                liability_raw = m_data.get('liabilities_total', 0.0)
                rf_rate = m_data.get('risk_free_rate', np.nan)
                
                # Adjust units if needed (ensure decimal)
                if not np.isnan(rf_rate) and abs(rf_rate) > 0.5:
                    rf_rate = rf_rate / 100.0
                
                if not np.isnan(v0_raw) and not np.isnan(liability_raw) and v0_raw > 0:
                    v0 = v0_raw
                    liability = liability_raw
            
            # Define horizons: 1y, 3y, 5y (trading days)
            horizon_days = np.array([252, 756, 1260], dtype=np.int32)
            
            # Run FULLY OPTIMIZED MS-GARCH Monte Carlo - calculates EVERYTHING in one pass!
            # Returns (Horizon, Firm, Sims)
            (terminal_vols, terminal_assets, default_indicators,
             vol_means, vol_stds, vol_maxs, vol_mins, regime_fractions) = \
                simulate_ms_garch_paths_t_jit_vectorized(
                    np.array([params['omega_0']]), np.array([params['omega_1']]),
                    np.array([params['alpha_0']]), np.array([params['alpha_1']]),
                    np.array([params['beta_0']]), np.array([params['beta_1']]),
                    np.array([params['mu_0']]), np.array([params['mu_1']]),
                    np.array([params['p00']]), np.array([params['p11']]),
                    np.array([params['nu_0']]), np.array([params['nu_1']]),
                    np.array([initial_sigma2]),
                    np.array([regime_prob]),
                    num_simulations, 1, horizon_days,
                    np.array([v0]), np.array([liability])
                )
            
            # Extract firm results (firm index 0 since processing one firm)
            # New Shape is (Horizon, Firm, Sim) -> (3, 1, N)
            firm_vols_1y = terminal_vols[0, 0, :]
            firm_vols_3y = terminal_vols[1, 0, :]
            firm_vols_5y = terminal_vols[2, 0, :]
            
            firm_assets_1y = terminal_assets[0, 0, :]
            firm_assets_3y = terminal_assets[1, 0, :]
            firm_assets_5y = terminal_assets[2, 0, :]
            
            firm_defaulted_1y = default_indicators[0, 0, :]
            firm_defaulted_3y = default_indicators[1, 0, :]
            firm_defaulted_5y = default_indicators[2, 0, :]
            
            # Get summary statistics
            # Shape (Firm, Sim) -> (1, N)
            firm_vol_mean = np.mean(vol_means[0, :])
            firm_vol_std = np.mean(vol_stds[0, :])
            firm_vol_max = np.max(vol_maxs[0, :])
            firm_vol_min = np.min(vol_mins[0, :])
            
            # Regime fractions
            regime_0_frac = np.mean(regime_fractions[0, 0, :])
            regime_1_frac = np.mean(regime_fractions[1, 0, :])
            
            # Calculate PD from pre-computed default indicators
            pd_value_1y = np.mean(firm_defaulted_1y)
            pd_value_3y = np.mean(firm_defaulted_3y)
            pd_value_5y = np.mean(firm_defaulted_5y)
            
            # Calculate CDS spreads
            mc_spread_1y, mc_spread_3y, mc_spread_5y = np.nan, np.nan, np.nan
            mc_debt_1y, mc_debt_3y, mc_debt_5y = np.nan, np.nan, np.nan
            
            if v0 > 0 and liability > 0 and not np.isnan(rf_rate):
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
            
            # Legacy compatibility fields
            pd_value = pd_value_1y
            mc_spread = mc_spread_1y
            mc_debt_value = mc_debt_1y
            
            # Calculate percentiles from terminal values
            firm_p95 = np.percentile(firm_vols_5y, 95)
            firm_p05 = np.percentile(firm_vols_5y, 5)
            
            results_list.append({
                'gvkey': firm,
                'date': date,
                'mc_msgarch_mean_daily_volatility': firm_vol_mean,
                'mc_msgarch_std_daily_volatility': firm_vol_std,
                'mc_msgarch_max_daily_volatility': firm_vol_max,
                'mc_msgarch_min_daily_volatility': firm_vol_min,
                'mc_msgarch_p95_daily_volatility': firm_p95,
                'mc_msgarch_p05_daily_volatility': firm_p05,
                'mc_msgarch_frac_regime_0': regime_0_frac,
                'mc_msgarch_frac_regime_1': regime_1_frac,
                'mc_msgarch_probability_of_default': pd_value,
                'mc_msgarch_pd_1y': pd_value_1y, # Now Terminal PD
                'mc_msgarch_pd_3y': pd_value_3y,
                'mc_msgarch_pd_5y': pd_value_5y,
                'mc_msgarch_pd_terminal_1y': pd_value_1y, # Explicitly named
                'mc_msgarch_pd_terminal_3y': pd_value_3y,
                'mc_msgarch_pd_terminal_5y': pd_value_5y,
                'mc_msgarch_implied_spread_1y': mc_spread_1y,
                'mc_msgarch_implied_spread_3y': mc_spread_3y,
                'mc_msgarch_implied_spread_5y': mc_spread_5y,
                'mc_msgarch_debt_value_1y': mc_debt_1y,
                'mc_msgarch_debt_value_3y': mc_debt_3y,
                'mc_msgarch_debt_value_5y': mc_debt_5y,
                'mc_msgarch_implied_spread': mc_spread,
                'mc_msgarch_debt_value': mc_debt_value
            })
            
        except Exception as e:
            print(f"Error processing firm {firm} on date {date}: {e}")
            continue
    
    return results_list


def monte_carlo_ms_garch_1year_parallel(daily_returns_file, ms_garch_params_file, 
                                       gvkey_selected=None, num_simulations=1000, 
                                       num_days=1260, n_jobs=-1, merton_df=None):
    """
    Parallelized Monte Carlo MS-GARCH forecast for 1-5 years (default 1260 days/5 years).
    
    This version processes different dates in parallel for significant speedup.
    
    Parameters:
    -----------
    daily_returns_file : str or pd.DataFrame
        CSV file or DataFrame with daily returns and MS-GARCH volatilities
    ms_garch_params_file : str or pd.DataFrame
        CSV file or DataFrame with MS-GARCH model parameters
    gvkey_selected : list or None
        List of gvkeys to process, or None for all firms
    num_simulations : int
        Number of Monte Carlo paths per firm per date
    num_days : int
        Forecast horizon in trading days (252 = 1 year)
    n_jobs : int
        Number of parallel jobs (-1 = use all cores)
    merton_df : pd.DataFrame, optional
        Pre-loaded Merton data to avoid reloading
        
    Returns:
    --------
    pd.DataFrame
        Results with Monte Carlo volatility and regime statistics per firm per date
    """
    if isinstance(daily_returns_file, str):
        print(f"Loading daily returns from {daily_returns_file}...")
        df = pd.read_csv(daily_returns_file)
    else:
        print(f"Using provided daily returns DataFrame...")
        df = daily_returns_file.copy()
    
    if isinstance(ms_garch_params_file, str):
        print(f"Loading MS-GARCH parameters from {ms_garch_params_file}...")
        params_df = pd.read_csv(ms_garch_params_file)
    else:
        print(f"Using provided MS-GARCH parameters DataFrame...")
        params_df = ms_garch_params_file.copy()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter firms if specified
    if gvkey_selected is not None:
        df = df[df['gvkey'].isin(gvkey_selected)]
        params_df = params_df[params_df['gvkey'].isin(gvkey_selected)]
    
    # Create parameters dictionary for faster lookup
    msgarch_params = {}
    for _, row in params_df.iterrows():
        msgarch_params[row['gvkey']] = {
            'omega_0': row['omega_0'], 'omega_1': row['omega_1'],
            'alpha_0': row['alpha_0'], 'alpha_1': row['alpha_1'],
            'beta_0': row['beta_0'], 'beta_1': row['beta_1'],
            'mu_0': row['mu_0'], 'mu_1': row['mu_1'],
            'p00': row['p00'], 'p11': row['p11'],
            'nu_0': row.get('nu_0', 30.0), 'nu_1': row.get('nu_1', 30.0)
        }
    
    print(f"Running PARALLELIZED Monte Carlo MS-GARCH simulation:")
    print(f"  Firms: {df['gvkey'].nunique()}")
    print(f"  Dates: {df['date'].nunique() if 'date' in df.columns else 1}")
    print(f"  Simulations per firm: {num_simulations:,}")
    print(f"  Forecast horizon: {num_days} days")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Innovation distribution: Student's t per regime")
    
    start_time = pd.Timestamp.now()
    
    # Load Merton Data for PD calculation
    merton_by_date = {}
    
    if merton_df is not None:
        print(f"✓ Using provided Merton data for PD calculation ({len(merton_df):,} rows)")
        if 'date' in merton_df.columns:
             if not pd.api.types.is_datetime64_any_dtype(merton_df['date']):
                 merton_df = merton_df.copy()
                 merton_df['date'] = pd.to_datetime(merton_df['date'])
             merton_by_date = {k: v for k, v in merton_df.groupby('date')}
    else:
        merton_file = None
        # Try to find the file in several locations
        potential_paths = []

        if isinstance(daily_returns_file, str):
            # Same directory as input file
            potential_paths.append(os.path.join(os.path.dirname(daily_returns_file), 'merged_data_with_merton.csv'))
            # Common project structure paths
            potential_paths.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(daily_returns_file))), 'data', 'output', 'merged_data_with_merton.csv'))
        
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
    
    # Prepare date groups for parallel processing
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
            date_groups.append((date, group, msgarch_params, merton_date_dict))
    else:
        date_groups = [(pd.Timestamp.now().date(), df, msgarch_params, {})]
    
    print(f"\nProcessing {len(date_groups)} dates in parallel...")
    
    # OPTIMIZATION: Avoid Joblib overhead for single date / Numba interaction
    use_joblib = True
    if len(date_groups) == 1:
        use_joblib = False
    elif len(date_groups) < n_jobs and n_jobs > 1:
        pass

    if use_joblib:
        # Parallel processing across dates
        print(f"Refuting Numba-parallelism to Joblib workers (dates={len(date_groups)})")
        results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_single_date_msgarch_mc)(date_data, num_simulations, num_days) 
            for date_data in date_groups
        )
    else:
        # Run directly in main process - allows Numba to use all cores effectively
        print("Optimization: Running simulation directly (avoiding Joblib overhead for single/few batches)")
        results_nested = [
            _process_single_date_msgarch_mc(date_data, num_simulations, num_days)
            for date_data in date_groups
        ]
    
    # Flatten results
    results_list = []
    for date_results in results_nested:
        results_list.extend(date_results)
    
    results_df = pd.DataFrame(results_list)
    
    if len(results_df) > 0:
        print(f"\nRegime Fraction Statistics:")
        print(f"  Regime 0 (low vol): mean={results_df['mc_msgarch_frac_regime_0'].mean():.3f}")
        print(f"  Regime 1 (high vol): mean={results_df['mc_msgarch_frac_regime_1'].mean():.3f}")
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    print(f"\n{'='*80}")
    print(f"PARALLELIZED MONTE CARLO MS-GARCH COMPLETE")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Rows per second: {len(results_df) / total_time:.0f}")
    print(f"Speedup: ~{n_jobs}x expected on {n_jobs}-core system")
    print(f"{'='*80}\n")
    
    return results_df
