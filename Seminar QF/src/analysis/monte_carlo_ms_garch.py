# monte_carlo_ms_garch.py

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba
from joblib import Parallel, delayed



@numba.jit(nopython=True, fastmath=True, cache=True)
def simulate_ms_garch_pd_spreads_t_jit(omega_0_arr, omega_1_arr, alpha_0_arr, alpha_1_arr, 
                                       beta_0_arr, beta_1_arr, mu_0_arr, mu_1_arr,
                                       p00_arr, p11_arr, nu_0_arr, nu_1_arr,
                                       sigma_arr, regime_prob_arr,
                                       num_simulations, num_firms,
                                       horizon_days, v0_arr, liability_arr, rf_arr,
                                       use_antithetic=False):
    """
    Optimized Monte Carlo MS-GARCH simulation that computes only PD and spreads.
    Uses log-asset evolution and sigma2 state to minimize exp() calls.
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
        omega_0 = omega_0_arr[f]
        omega_1 = omega_1_arr[f]
        alpha_0 = alpha_0_arr[f]
        alpha_1 = alpha_1_arr[f]
        beta_0 = beta_0_arr[f]
        beta_1 = beta_1_arr[f]
        mu_0 = mu_0_arr[f]
        mu_1 = mu_1_arr[f]
        p00 = p00_arr[f]
        p11 = p11_arr[f]
        nu_0 = nu_0_arr[f]
        nu_1 = nu_1_arr[f]
        initial_sigma = sigma_arr[f]
        regime_prob = regime_prob_arr[f]
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
        sigma2 = np.full(num_simulations, initial_sigma ** 2)
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        log_asset = np.full(num_simulations, log_v0)
        
        # Initialize regime state
        regime_1_mask = np.random.random(num_simulations) < regime_prob
        
        # Accumulators for each horizon
        default_counts = np.zeros(n_horizons)
        payoff_sums = np.zeros(n_horizons)
        
        # T-dist parameters
        check_normal_0 = (nu_0 >= 100)
        check_normal_1 = (nu_1 >= 100)
        
        if nu_0 > 2.05:
            t_factor_0 = np.sqrt((nu_0 - 2) / nu_0)
        else:
            safe_nu_0 = max(nu_0, 2.0001)
            t_factor_0 = np.sqrt((safe_nu_0 - 2) / safe_nu_0)
            
        if nu_1 > 2.05:
            t_factor_1 = np.sqrt((nu_1 - 2) / nu_1)
        else:
            safe_nu_1 = max(nu_1, 2.0001)
            t_factor_1 = np.sqrt((safe_nu_1 - 2) / safe_nu_1)
        
        for day in range(max_days):
            # 1. Vectorized Random Generation per regime
            if use_antithetic and num_simulations > 1:
                half = num_simulations // 2
                z_half = np.random.standard_normal(half)
                z = np.empty(num_simulations)
                z[:half] = z_half
                z[half:(2 * half)] = -z_half
                if num_simulations % 2 == 1:
                    z[num_simulations - 1] = np.random.standard_normal()
            else:
                z = np.random.standard_normal(num_simulations)
            
            # Generate t-distributed samples for both regimes
            if check_normal_0:
                t_sample_0 = z
            else:
                if use_antithetic and num_simulations > 1:
                    half = num_simulations // 2
                    v0_half = np.random.chisquare(nu_0, half)
                    v0 = np.empty(num_simulations)
                    v0[:half] = v0_half
                    v0[half:(2 * half)] = v0_half
                    if num_simulations % 2 == 1:
                        v0[num_simulations - 1] = np.random.chisquare(nu_0)
                else:
                    v0 = np.random.chisquare(nu_0, num_simulations)
                v0 = np.maximum(v0, 1e-12)
                t_sample_0 = z / np.sqrt(v0 / nu_0) * t_factor_0

            if check_normal_1:
                t_sample_1 = z
            else:
                if use_antithetic and num_simulations > 1:
                    half = num_simulations // 2
                    v1_half = np.random.chisquare(nu_1, half)
                    v1 = np.empty(num_simulations)
                    v1[:half] = v1_half
                    v1[half:(2 * half)] = v1_half
                    if num_simulations % 2 == 1:
                        v1[num_simulations - 1] = np.random.chisquare(nu_1)
                else:
                    v1 = np.random.chisquare(nu_1, num_simulations)
                v1 = np.maximum(v1, 1e-12)
                t_sample_1 = z / np.sqrt(v1 / nu_1) * t_factor_1

            t_sample = np.where(regime_1_mask, t_sample_1, t_sample_0)
            
            # 2. Vectorized Updates with regime-specific parameters
            eps = sigma * t_sample
            
            # Update log-asset with regime-specific drift
            log_asset += np.where(regime_1_mask, mu_1 + eps, mu_0 + eps)
            
            # 3. Horizon Check
            if is_horizon[day]:
                h = horizon_map[day]
                
                # Default detection in log-space
                defaults = (log_asset < log_liability).astype(np.float64)
                default_counts[h] = np.sum(defaults)
                
                # Compute asset values at horizon (exp only here)
                asset_T = np.exp(log_asset)
                payoffs = np.minimum(asset_T, liability)
                payoff_sums[h] = np.sum(payoffs)
            
            # 4. GARCH Update using sigma2 state with regime-specific parameters
            sigma2 = np.where(
                regime_1_mask,
                omega_1 + alpha_1 * eps**2 + beta_1 * sigma2,
                omega_0 + alpha_0 * eps**2 + beta_0 * sigma2,
            )
            sigma2 = np.maximum(sigma2, 1e-12)
            sigma = np.sqrt(sigma2)
            
            # 5. Regime Transition
            u = np.random.random(num_simulations)
            switch_to_0 = regime_1_mask & (u > p11)
            switch_to_1 = (~regime_1_mask) & (u > p00)
            regime_1_mask = (regime_1_mask & (~switch_to_0)) | switch_to_1
        
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


def _process_single_date_ms_garch_mc(date_data, num_simulations, num_days, exclude_firms_without_estimated_params=True, use_antithetic=False):
    """
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
    date, df_date, merton_data_dict = date_data
    
    if df_date.empty:
        return []
    
    # Vectorized preparation: drop duplicates and sort
    df_firms = df_date.drop_duplicates('gvkey', keep='first').sort_values('gvkey').reset_index(drop=True)
    firms_list = df_firms['gvkey'].tolist()
    num_firms = len(firms_list)
    
    def _first_existing_series(candidates, default_value):
        for col in candidates:
            if col in df_firms.columns:
                return df_firms[col], True
        return pd.Series([default_value] * num_firms), False

    column_candidates = {
        'omega_0': ['ms_garch_omega_0', 'omega_0'],
        'omega_1': ['ms_garch_omega_1', 'omega_1'],
        'alpha_0': ['ms_garch_alpha_0', 'alpha_0'],
        'alpha_1': ['ms_garch_alpha_1', 'alpha_1'],
        'beta_0': ['ms_garch_beta_0', 'beta_0'],
        'beta_1': ['ms_garch_beta_1', 'beta_1'],
        'mu_0': ['ms_garch_mu_0', 'mu_0'],
        'mu_1': ['ms_garch_mu_1', 'mu_1'],
        'p00': ['ms_garch_p00', 'p00'],
        'p11': ['ms_garch_p11', 'p11'],
        'nu_0': ['ms_garch_nu_0', 'nu_0'],
        'nu_1': ['ms_garch_nu_1', 'nu_1'],
        'volatility': ['ms_garch_volatility'],
        'regime_prob': ['ms_garch_regime_prob'],
    }

    series_map = {}
    has_estimated_model_params = np.ones(num_firms, dtype=bool)

    for key, candidates in column_candidates.items():
        series, exists = _first_existing_series(candidates, np.nan)
        series_map[key] = series
        if exists:
            has_estimated_model_params &= series.notna().values
        else:
            has_estimated_model_params &= False

    # Vectorized parameter extraction with defaults and floors
    omega_0_arr = np.maximum(series_map['omega_0'].fillna(1e-6).values, 1e-8)
    omega_1_arr = np.maximum(series_map['omega_1'].fillna(1e-6).values, 1e-8)
    alpha_0_arr = np.maximum(series_map['alpha_0'].fillna(0.05).values, 1e-4)
    alpha_1_arr = np.maximum(series_map['alpha_1'].fillna(0.05).values, 1e-4)
    beta_0_arr = np.maximum(series_map['beta_0'].fillna(0.93).values, 0.0)
    beta_1_arr = np.maximum(series_map['beta_1'].fillna(0.93).values, 0.0)
    mu_0_arr = series_map['mu_0'].fillna(0.0).values
    mu_1_arr = series_map['mu_1'].fillna(0.0).values
    p00_arr = series_map['p00'].fillna(0.95).values
    p11_arr = series_map['p11'].fillna(0.95).values
    nu_0_arr = series_map['nu_0'].fillna(30.0).values
    nu_1_arr = series_map['nu_1'].fillna(30.0).values
    sigma_arr = np.maximum(series_map['volatility'].fillna(0.02).values, 1e-4)
    regime_prob_arr = series_map['regime_prob'].fillna(0.5).values
    
    # Prepare Merton arrays
    v0_arr = np.full(num_firms, np.nan)
    liability_arr = np.full(num_firms, np.nan)
    rf_arr = np.full(num_firms, np.nan)
    
    for firm_idx, firm in enumerate(firms_list):
        if firm in merton_data_dict:
            m_data = merton_data_dict[firm]
            v0 = m_data.get('asset_value', np.nan)
            liability = m_data.get('liabilities_total', np.nan)
            rf_rate = m_data.get('risk_free_rate', np.nan)
            
            v0_arr[firm_idx] = v0
            liability_arr[firm_idx] = liability
            
            # Scale rf_rate if needed
            if not np.isnan(rf_rate) and abs(rf_rate) > 0.5:
                rf_rate = rf_rate / 100.0
            rf_arr[firm_idx] = rf_rate
    
    # Define horizons: 1y, 3y, 5y (trading days)
    horizon_days = np.array([252, 756, 1260], dtype=np.int32)
    
    # Run simulation
    pd_out, spread_out, debt_out = simulate_ms_garch_pd_spreads_t_jit(
        omega_0_arr, omega_1_arr, alpha_0_arr, alpha_1_arr,
        beta_0_arr, beta_1_arr, mu_0_arr, mu_1_arr,
        p00_arr, p11_arr, nu_0_arr, nu_1_arr,
        sigma_arr, regime_prob_arr,
        num_simulations, num_firms, horizon_days,
        v0_arr, liability_arr, rf_arr,
        use_antithetic,
    )

    if exclude_firms_without_estimated_params:
        invalid_mask = ~has_estimated_model_params
        if np.any(invalid_mask):
            pd_out[:, invalid_mask] = np.nan
            spread_out[:, invalid_mask] = np.nan
            debt_out[:, invalid_mask] = np.nan
    
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
            'has_estimated_ms_garch_params': bool(has_estimated_model_params[firm_idx]),
            'used_default_ms_garch_inputs': bool(not has_estimated_model_params[firm_idx]),
            'mc_ms_garch_pd_1y': pd_1y[firm_idx],
            'mc_ms_garch_pd_3y': pd_3y[firm_idx],
            'mc_ms_garch_pd_5y': pd_5y[firm_idx],
            'mc_ms_garch_pd_terminal_1y': pd_1y[firm_idx],
            'mc_ms_garch_pd_terminal_3y': pd_3y[firm_idx],
            'mc_ms_garch_pd_terminal_5y': pd_5y[firm_idx],
            'mc_ms_garch_implied_spread_1y': mc_spread_1y[firm_idx],
            'mc_ms_garch_implied_spread_3y': mc_spread_3y[firm_idx],
            'mc_ms_garch_implied_spread_5y': mc_spread_5y[firm_idx],
            'mc_ms_garch_debt_value_1y': mc_debt_1y[firm_idx],
            'mc_ms_garch_debt_value_3y': mc_debt_3y[firm_idx],
            'mc_ms_garch_debt_value_5y': mc_debt_5y[firm_idx],
        })
    
    return results_list


def monte_carlo_ms_garch_1year_parallel(ms_garch_file, merton_file, gvkey_selected=None, num_simulations=1000, num_days=1260, n_jobs=-1, exclude_firms_without_estimated_params=True, use_antithetic=False):
    print(f"Loading MS-GARCH data from {ms_garch_file}...")
    df = pd.read_csv(ms_garch_file)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter firms if specified
    if gvkey_selected is not None:
        df = df[df['gvkey'].isin(gvkey_selected)]
    
    print(f"Running PARALLELIZED Monte Carlo MS-GARCH simulation:")
    print(f"  Firms: {df['gvkey'].nunique()}")
    print(f"  Dates: {df['date'].nunique() if 'date' in df.columns else 1}")
    print(f"  Simulations per firm: {num_simulations:,}")
    print(f"  Forecast horizon: {num_days} days")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Innovation distribution: Student's t per regime")
    print(f"  Antithetic variates: {use_antithetic}")
    print(f"  Exclude rows without estimated MS-GARCH params: {exclude_firms_without_estimated_params}")
    
    start_time = pd.Timestamp.now()
    
    # Prepare date groups for parallel processing
    
    # Load Merton Data for PD calculation
    merton_by_date = {}
    df_merton = pd.read_csv(merton_file)
    df_merton['date'] = pd.to_datetime(df_merton['date'])
    merton_by_date = {k: v for k, v in df_merton.groupby('date')}
    print(f"✓ Loaded Merton data for PD calculation ({len(df_merton):,} rows) from {merton_file}")

    date_groups = []
    if 'date' in df.columns:
        for date, group in df.groupby('date'):
            # Prepare merton dict
            merton_date_dict = {}
            if date in merton_by_date:
                df_m = merton_by_date[date]
                firms_on_date = group['gvkey'].unique()
                df_m_relevant = df_m[df_m['gvkey'].isin(firms_on_date)]
                # Vectorized construction: subset columns, drop duplicates (keep='last'), convert to dict
                df_m_subset = df_m_relevant[['gvkey', 'asset_value', 'liabilities_total', 'risk_free_rate']]
                df_m_subset = df_m_subset.drop_duplicates(subset='gvkey', keep='last')
                merton_date_dict = df_m_subset.set_index('gvkey').to_dict('index')
            date_groups.append((date, group, merton_date_dict))
    else:
        # Single date case
        date_groups = [(pd.Timestamp.now().date(), df, {})]
    
    print(f"\nProcessing {len(date_groups)} dates in parallel...")
    
    # OPTIMIZATION: Avoid Joblib overhead for single date / Numba interaction
    use_joblib = True

    # Parallel processing across dates
    print(f"Refuting Numba-parallelism to Joblib workers (dates={len(date_groups)})")
    results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_date_ms_garch_mc)(
            date_data,
            num_simulations,
            num_days,
            exclude_firms_without_estimated_params,
            use_antithetic,
        ) 
        for date_data in date_groups
    )
    
    # Flatten results
    results_list = []
    for date_results in results_nested:
        results_list.extend(date_results)
    
    results_df = pd.DataFrame(results_list)
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"\n{'='*80}")
    print(f"PARALLELIZED MONTE CARLO MS-GARCH COMPLETE")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    if not results_df.empty and 'used_default_ms_garch_inputs' in results_df.columns:
        excluded_share = results_df['used_default_ms_garch_inputs'].mean()
        print(f"Rows without estimated MS-GARCH params (excluded from spreads): {excluded_share:.2%}")
    print(f"{'='*80}\n")
    
    return results_df

