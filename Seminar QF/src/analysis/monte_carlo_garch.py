# monte_carlo_garch.py

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba
from joblib import Parallel, delayed
from src.analysis.cds_date_filter import load_allowed_cds_dates, filter_df_to_allowed_dates


def _horizon_vector_from_max_days(max_days: int) -> np.ndarray:
    if max_days <= 252:
        return np.array([252], dtype=np.int32)
    if max_days <= 756:
        return np.array([252, 756], dtype=np.int32)
    return np.array([252, 756, 1260], dtype=np.int32)


@numba.jit(nopython=True, fastmath=True, cache=True)
def simulate_garch_pd_spreads_t_jit(omega_arr, alpha_arr, beta_arr, 
                                    sigma_arr, nu_arr,
                                    num_simulations, num_firms,
                                    horizon_days, v0_arr, liability_arr, rf_arr,
                                    use_antithetic=False,
                                    spread_cap=0.5):
    """
    Optimized Monte Carlo GARCH simulation that computes only PD and spreads.
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
        omega = omega_arr[f]
        alpha = alpha_arr[f]
        beta = beta_arr[f]
        nu = nu_arr[f]
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
        liability_horizon = np.full(n_horizons, liability)
        log_liability_horizon = np.full(n_horizons, np.log(liability))
        if valid_rf:
            for h in range(n_horizons):
                T_years = horizon_days[h] / 252.0
                liability_T = liability * np.exp(rf_rate * T_years)
                liability_horizon[h] = liability_T
                log_liability_horizon[h] = np.log(liability_T)
        
        # State Vectors (Size: num_simulations)
        sigma2 = np.full(num_simulations, sigma_arr[f] ** 2)
        sigma = np.sqrt(np.maximum(sigma2, 1e-12))
        log_asset = np.full(num_simulations, log_v0)
        
        # Accumulators for each horizon
        default_counts = np.zeros(n_horizons)
        payoff_sums = np.zeros(n_horizons)
        
        # T-dist parameters
        check_normal = (nu >= 100)
        
        if nu > 2.05:
            t_factor = np.sqrt((nu - 2) / nu)
        else:
            safe_nu = max(nu, 2.0001)
            t_factor = np.sqrt((safe_nu - 2) / safe_nu)
        
        for day in range(max_days):
            # 1. Vectorized Random Generation
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
            
            if check_normal:
                t_sample = z
            else:
                if use_antithetic and num_simulations > 1:
                    half = num_simulations // 2
                    v_half = np.random.chisquare(nu, half)
                    v = np.empty(num_simulations)
                    v[:half] = v_half
                    v[half:(2 * half)] = v_half
                    if num_simulations % 2 == 1:
                        v[num_simulations - 1] = np.random.chisquare(nu)
                else:
                    v = np.random.chisquare(nu, num_simulations)
                v = np.maximum(v, 1e-12)
                t_sample = z / np.sqrt(v / nu) * t_factor
            
            # 2. Vectorized Updates
            eps = sigma * t_sample
            
            # TRUNCATION: Cap t_sample at 5 standard deviations (as per paper page 12)
            t_sample = np.clip(t_sample, -5.0, 5.0)
            eps = sigma * t_sample

            # Update log-asset using shocks only (no drift)
            log_asset += eps
            
            # 3. Horizon Check
            if is_horizon[day]:
                h = horizon_map[day]
                log_liability_T = log_liability_horizon[h]
                liability_T = liability_horizon[h]
                
                # Default detection in log-space
                defaults = (log_asset < log_liability_T).astype(np.float64)
                default_counts[h] = np.sum(defaults)
                
                # Compute asset values at horizon (exp only here)
                asset_T = np.exp(log_asset)
                payoffs = np.minimum(asset_T, liability_T)
                payoff_sums[h] = np.sum(payoffs)
            
            # 4. GARCH Update using sigma2 state
            sigma2 = omega + alpha * eps**2 + beta * sigma2
            sigma2 = np.maximum(sigma2, 1e-12)
            sigma = np.sqrt(sigma2)
        
        # Compute PD and spreads for each horizon
        for h in range(n_horizons):
            # PD
            pd_out[h, f] = default_counts[h] / num_simulations
            
            # Expected payoff and debt value
            expected_payoff = payoff_sums[h] / num_simulations
            
            # Compute spreads only if rf_rate is valid
            if valid_rf:
                T_years = horizon_days[h] / 252.0
                liability_T = liability_horizon[h]
                debt_val = expected_payoff * np.exp(-rf_rate * T_years)
                debt_out[h, f] = debt_val
                
                if debt_val > 0:
                    ytm = -np.log(debt_val / liability_T) / T_years
                    spread = max(ytm - rf_rate, 0.0)
                    spread_out[h, f] = min(spread, spread_cap)
    
    return pd_out, spread_out, debt_out


def _process_single_date_garch_mc(date_data, num_simulations, num_days, exclude_firms_without_estimated_garch=True, use_antithetic=False, spread_cap=0.5):
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
    
    required_garch_cols = ['garch_omega', 'garch_alpha', 'garch_beta', 'garch_volatility', 'garch_nu']
    if all(col in df_firms.columns for col in required_garch_cols):
        has_estimated_garch_params = df_firms[required_garch_cols].notna().all(axis=1).values
    else:
        has_estimated_garch_params = np.zeros(num_firms, dtype=bool)

    # Vectorized parameter extraction with conservative defaults only for numerical safety.
    # Missing estimated parameters are excluded from output when exclude_firms_without_estimated_garch=True.
    omega_arr = np.maximum(df_firms.get('garch_omega', pd.Series([1e-6]*num_firms)).fillna(1e-6).values, 1e-8)
    alpha_arr = np.maximum(df_firms.get('garch_alpha', pd.Series([0.05]*num_firms)).fillna(0.05).values, 1e-4)
    beta_arr = np.maximum(df_firms.get('garch_beta', pd.Series([0.93]*num_firms)).fillna(0.93).values, 0.0)
    sigma_arr = np.maximum(df_firms.get('garch_volatility', pd.Series([0.02]*num_firms)).fillna(0.02).values, 1e-4)
    nu_arr = df_firms.get('garch_nu', pd.Series([30.0]*num_firms)).fillna(30.0).values
    
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
    
    # Maturity-aware horizons per firm-date; default to 5Y if not provided
    if 'cds_max_horizon_days' in df_firms.columns:
        required_horizons = pd.to_numeric(df_firms['cds_max_horizon_days'], errors='coerce').fillna(1260).astype(int).values
    else:
        required_horizons = np.full(num_firms, 1260, dtype=np.int32)
    required_horizons = np.where(required_horizons <= 252, 252, np.where(required_horizons <= 756, 756, 1260)).astype(np.int32)

    pd_out = np.full((3, num_firms), np.nan)
    spread_out = np.full((3, num_firms), np.nan)
    debt_out = np.full((3, num_firms), np.nan)

    for max_days in np.unique(required_horizons):
        idx = np.where(required_horizons == max_days)[0]
        if len(idx) == 0:
            continue

        horizon_days = _horizon_vector_from_max_days(int(max_days))
        sub_pd, sub_spread, sub_debt = simulate_garch_pd_spreads_t_jit(
            omega_arr[idx], alpha_arr[idx], beta_arr[idx], sigma_arr[idx], nu_arr[idx],
            num_simulations, len(idx), horizon_days,
            v0_arr[idx], liability_arr[idx], rf_arr[idx],
            use_antithetic,
            spread_cap,
        )

        pd_out[0, idx] = sub_pd[0, :]
        spread_out[0, idx] = sub_spread[0, :]
        debt_out[0, idx] = sub_debt[0, :]
        if horizon_days.shape[0] >= 2:
            pd_out[1, idx] = sub_pd[1, :]
            spread_out[1, idx] = sub_spread[1, :]
            debt_out[1, idx] = sub_debt[1, :]
        if horizon_days.shape[0] >= 3:
            pd_out[2, idx] = sub_pd[2, :]
            spread_out[2, idx] = sub_spread[2, :]
            debt_out[2, idx] = sub_debt[2, :]

    if exclude_firms_without_estimated_garch:
        invalid_mask = ~has_estimated_garch_params
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
            'cds_max_horizon_days': int(required_horizons[firm_idx]),
            'has_estimated_garch_params': bool(has_estimated_garch_params[firm_idx]),
            'used_default_garch_inputs': bool(not has_estimated_garch_params[firm_idx]),
            'mc_garch_pd_1y': pd_1y[firm_idx],
            'mc_garch_pd_3y': pd_3y[firm_idx],
            'mc_garch_pd_5y': pd_5y[firm_idx],
            'mc_garch_pd_terminal_1y': pd_1y[firm_idx],
            'mc_garch_pd_terminal_3y': pd_3y[firm_idx],
            'mc_garch_pd_terminal_5y': pd_5y[firm_idx],
            'mc_garch_implied_spread_1y': mc_spread_1y[firm_idx],
            'mc_garch_implied_spread_3y': mc_spread_3y[firm_idx],
            'mc_garch_implied_spread_5y': mc_spread_5y[firm_idx],
            'mc_garch_debt_value_1y': mc_debt_1y[firm_idx],
            'mc_garch_debt_value_3y': mc_debt_3y[firm_idx],
            'mc_garch_debt_value_5y': mc_debt_5y[firm_idx],
        })
    
    return results_list


def monte_carlo_garch_1year_parallel(garch_file, merton_file, gvkey_selected=None, num_simulations=1000, num_days=1260, n_jobs=-1, exclude_firms_without_estimated_garch=True, use_antithetic=False, spread_cap=0.5, cds_filter_file=None):
    print(f"Loading GARCH data from {garch_file}...")
    df = pd.read_csv(garch_file)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter firms if specified
    if gvkey_selected is not None:
        df = df[df['gvkey'].isin(gvkey_selected)]

    if cds_filter_file:
        allowed_dates = load_allowed_cds_dates(cds_filter_file)
        before_rows = len(df)
        before_dates = df['date'].nunique() if 'date' in df.columns else 1
        df = filter_df_to_allowed_dates(df, allowed_dates, date_col='date')
        after_rows = len(df)
        after_dates = df['date'].nunique() if 'date' in df.columns else 1
        print(f"✓ Applied CDS clean-date filter from {cds_filter_file}")
        print(f"  Rows: {before_rows:,} -> {after_rows:,}; Dates: {before_dates} -> {after_dates}")
    
    print(f"Running PARALLELIZED Monte Carlo GARCH simulation:")
    print(f"  Firms: {df['gvkey'].nunique()}")
    print(f"  Dates: {df['date'].nunique() if 'date' in df.columns else 1}")
    print(f"  Simulations per firm: {num_simulations:,}")
    print(f"  Forecast horizon: {num_days} days")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Innovation distribution: Student's t")
    print(f"  Antithetic variates: {use_antithetic}")
    print(f"  CDS spread cap: {spread_cap:.4f} ({spread_cap*10000:.0f} bps)")
    print(f"  Exclude rows without estimated GARCH params: {exclude_firms_without_estimated_garch}")
    
    start_time = pd.Timestamp.now()
    
    # Prepare date groups for parallel processing
    
    # Load Merton Data for PD calculation
    merton_by_date = {}
    df_merton = pd.read_csv(merton_file)
    df_merton['date'] = pd.to_datetime(df_merton['date'])
    if cds_filter_file:
        df_merton = filter_df_to_allowed_dates(df_merton, allowed_dates, date_col='date')
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
        delayed(_process_single_date_garch_mc)(
            date_data,
            num_simulations,
            num_days,
            exclude_firms_without_estimated_garch,
            use_antithetic,
            spread_cap,
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
    print(f"PARALLELIZED MONTE CARLO GARCH COMPLETE")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    if not results_df.empty and 'used_default_garch_inputs' in results_df.columns:
        excluded_share = results_df['used_default_garch_inputs'].mean()
        print(f"Rows without estimated GARCH params (excluded from spreads): {excluded_share:.2%}")
    print(f"{'='*80}\n")
    
    return results_df
