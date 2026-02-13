
# monte_carlo_regime_switching.py
# Efficient regime-switching Monte Carlo with vectorization AND t-distribution

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba
from joblib import Parallel, delayed

@numba.jit(nopython=True, fastmath=True, cache=True)
def simulate_regime_switching_vectorized(
    mu_0, mu_1, ar_0, ar_1, sigma_0, sigma_1, nu_0, nu_1,
    trans_00, trans_01, trans_10, trans_11,
    num_simulations, num_firms, horizon_days, initial_regime_probs,
    v0_arr, liability_arr, rf_arr,
    use_antithetic=False,
):
    """
    Optimized regime-switching Monte Carlo simulation with T-DISTRIBUTION.
    Uses log-asset evolution and horizon-based accumulation to minimize memory and computation.
    Returns only per-firm, per-horizon aggregates (no per-simulation arrays).
    """
    max_days = np.max(horizon_days)
    n_horizons = len(horizon_days)
    
    # Output arrays: (n_horizons, num_firms)
    pd_out = np.full((n_horizons, num_firms), np.nan)
    spread_out = np.full((n_horizons, num_firms), np.nan)
    debt_out = np.full((n_horizons, num_firms), np.nan)
    
    # Regime fractions: (2, num_firms)
    regime_fractions = np.zeros((2, num_firms))
    
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
        m0, m1 = mu_0[f], mu_1[f]
        ar0, ar1 = ar_0[f], ar_1[f]
        s0, s1 = sigma_0[f], sigma_1[f]
        n0, n1 = nu_0[f], nu_1[f]
        tr01, tr10 = trans_01[f], trans_10[f]
        prob_regime1 = initial_regime_probs[f]
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
        regime_1_mask = np.random.random(num_simulations) < prob_regime1
        sigma = np.where(regime_1_mask, s1, s0)
        log_asset = np.full(num_simulations, log_v0)
        r_prev = np.zeros(num_simulations)  # AR(1) previous return
        regime_0_cnt = np.zeros(num_simulations)
        regime_1_cnt = np.zeros(num_simulations)
        
        # Accumulators for each horizon
        default_counts = np.zeros(n_horizons)
        payoff_sums = np.zeros(n_horizons)
        
        # T-dist parameters
        check_normal_0 = (n0 >= 100)
        check_normal_1 = (n1 >= 100)
        
        if n0 > 2.05:
            t_factor_0 = np.sqrt((n0 - 2) / n0)
        else:
            safe_n0 = max(n0, 2.0001)
            t_factor_0 = np.sqrt((safe_n0 - 2) / safe_n0)
        
        if n1 > 2.05:
            t_factor_1 = np.sqrt((n1 - 2) / n1)
        else:
            safe_n1 = max(n1, 2.0001)
            t_factor_1 = np.sqrt((safe_n1 - 2) / safe_n1)

        for day in range(max_days):
            # 1. Regime Transition and Counters
            regime_0_cnt += (~regime_1_mask)
            regime_1_cnt += regime_1_mask
            
            u = np.random.random(num_simulations)
            switch_to_1 = (~regime_1_mask) & (u < tr01)
            switch_to_0 = (regime_1_mask) & (u < tr10)
            regime_1_mask = (regime_1_mask | switch_to_1) & (~switch_to_0)
            
            # 2. Parameter Selection
            mu_curr = np.where(regime_1_mask, m1, m0)
            ar_curr = np.where(regime_1_mask, ar1, ar0)
            sigma = np.where(regime_1_mask, s1, s0)
            
            # 3. Vectorized Random Generation (T-dist)
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
            
            # Generate t-samples for both regimes
            if check_normal_0:
                t_sample_0 = z
            else:
                if use_antithetic and num_simulations > 1:
                    half = num_simulations // 2
                    v0_half = np.random.chisquare(n0, half)
                    v0 = np.empty(num_simulations)
                    v0[:half] = v0_half
                    v0[half:(2 * half)] = v0_half
                    if num_simulations % 2 == 1:
                        v0[num_simulations - 1] = np.random.chisquare(n0)
                else:
                    v0 = np.random.chisquare(n0, num_simulations)
                v0 = np.maximum(v0, 1e-12)
                t_sample_0 = z / np.sqrt(v0 / n0) * t_factor_0
            
            if check_normal_1:
                t_sample_1 = z
            else:
                if use_antithetic and num_simulations > 1:
                    half = num_simulations // 2
                    v1_half = np.random.chisquare(n1, half)
                    v1 = np.empty(num_simulations)
                    v1[:half] = v1_half
                    v1[half:(2 * half)] = v1_half
                    if num_simulations % 2 == 1:
                        v1[num_simulations - 1] = np.random.chisquare(n1)
                else:
                    v1 = np.random.chisquare(n1, num_simulations)
                v1 = np.maximum(v1, 1e-12)
                t_sample_1 = z / np.sqrt(v1 / n1) * t_factor_1
            
            z_t = np.where(regime_1_mask, t_sample_1, t_sample_0)
            
            # 4. Vectorized Updates
            r_curr = mu_curr + ar_curr * r_prev + sigma * z_t
            r_prev = r_curr
            
            # Update log-asset (no exp needed daily)
            log_asset += r_curr
            
            # 5. Horizon Check
            if is_horizon[day]:
                h = horizon_map[day]
                
                # Default detection in log-space
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
        
        # Regime fractions (averaged across simulations)
        regime_fractions[0, f] = np.mean(regime_0_cnt) / max_days
        regime_fractions[1, f] = np.mean(regime_1_cnt) / max_days
    
    return pd_out, spread_out, debt_out, regime_fractions


def _process_single_date_rs_mc(date_data, num_simulations, num_days, exclude_firms_without_estimated_params=True, use_antithetic=False):
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
    
    required_rs_cols = [
        'regime_0_mean', 'regime_1_mean', 'regime_0_ar', 'regime_1_ar',
        'regime_0_vol', 'regime_1_vol', 'regime_0_nu', 'regime_1_nu',
        'transition_prob_00', 'transition_prob_01', 'transition_prob_10', 'transition_prob_11'
    ]
    if all(col in df_firms.columns for col in required_rs_cols):
        has_estimated_model_params = df_firms[required_rs_cols].notna().all(axis=1).values
    else:
        has_estimated_model_params = np.zeros(num_firms, dtype=bool)

    # Vectorized parameter extraction with defaults and floors
    mu_0_arr = df_firms.get('regime_0_mean', pd.Series([0.0]*num_firms)).fillna(0.0).values
    mu_1_arr = df_firms.get('regime_1_mean', pd.Series([0.0]*num_firms)).fillna(0.0).values
    ar_0_arr = df_firms.get('regime_0_ar', pd.Series([0.0]*num_firms)).fillna(0.0).values
    ar_1_arr = df_firms.get('regime_1_ar', pd.Series([0.0]*num_firms)).fillna(0.0).values
    sigma_0_arr = np.maximum(df_firms.get('regime_0_vol', pd.Series([0.02]*num_firms)).fillna(0.02).values, 1e-4)
    sigma_1_arr = np.maximum(df_firms.get('regime_1_vol', pd.Series([0.02]*num_firms)).fillna(0.02).values, 1e-4)
    nu_0_arr = df_firms.get('regime_0_nu', pd.Series([30.0]*num_firms)).fillna(30.0).values
    nu_1_arr = df_firms.get('regime_1_nu', pd.Series([30.0]*num_firms)).fillna(30.0).values
    trans_00_arr = df_firms.get('transition_prob_00', pd.Series([0.95]*num_firms)).fillna(0.95).values
    trans_01_arr = df_firms.get('transition_prob_01', pd.Series([0.05]*num_firms)).fillna(0.05).values
    trans_10_arr = df_firms.get('transition_prob_10', pd.Series([0.05]*num_firms)).fillna(0.05).values
    trans_11_arr = df_firms.get('transition_prob_11', pd.Series([0.95]*num_firms)).fillna(0.95).values
    
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
    
    # Initial regime probabilities
    initial_regime_probs = np.full(num_firms, 0.5)
    
    # Define horizons: 1y, 3y, 5y (trading days)
    horizon_days = np.array([252, 756, 1260], dtype=np.int32)
    
    # Run simulation
    pd_out, spread_out, debt_out, regime_fractions = simulate_regime_switching_vectorized(
        mu_0_arr, mu_1_arr, ar_0_arr, ar_1_arr, sigma_0_arr, sigma_1_arr, nu_0_arr, nu_1_arr,
        trans_00_arr, trans_01_arr, trans_10_arr, trans_11_arr,
        num_simulations, num_firms, horizon_days, initial_regime_probs,
        v0_arr, liability_arr, rf_arr,
        use_antithetic,
    )

    if exclude_firms_without_estimated_params:
        invalid_mask = ~has_estimated_model_params
        if np.any(invalid_mask):
            pd_out[:, invalid_mask] = np.nan
            spread_out[:, invalid_mask] = np.nan
            debt_out[:, invalid_mask] = np.nan
            regime_fractions[:, invalid_mask] = np.nan
    
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
    
    regime_0_frac = regime_fractions[0, :]
    regime_1_frac = regime_fractions[1, :]
    
    # Assemble results
    results_list = []
    for firm_idx, firm in enumerate(firms_list):
        results_list.append({
            'gvkey': firm,
            'date': date,
            'has_estimated_rs_params': bool(has_estimated_model_params[firm_idx]),
            'used_default_rs_inputs': bool(not has_estimated_model_params[firm_idx]),
            'rs_fraction_regime_0': regime_0_frac[firm_idx],
            'rs_fraction_regime_1': regime_1_frac[firm_idx],
            'rs_probability_of_default': pd_1y[firm_idx],
            'rs_pd_1y': pd_1y[firm_idx],
            'rs_pd_3y': pd_3y[firm_idx],
            'rs_pd_5y': pd_5y[firm_idx],
            'rs_pd_terminal_1y': pd_1y[firm_idx],
            'rs_pd_terminal_3y': pd_3y[firm_idx],
            'rs_pd_terminal_5y': pd_5y[firm_idx],
            'rs_implied_spread_1y': mc_spread_1y[firm_idx],
            'rs_implied_spread_3y': mc_spread_3y[firm_idx],
            'rs_implied_spread_5y': mc_spread_5y[firm_idx],
            'rs_debt_value_1y': mc_debt_1y[firm_idx],
            'rs_debt_value_3y': mc_debt_3y[firm_idx],
            'rs_debt_value_5y': mc_debt_5y[firm_idx],
        })
    
    return results_list


def monte_carlo_regime_switching_1year_parallel(regime_params_file, merton_file, gvkey_selected=None, num_simulations=1000, num_days=1260, n_jobs=-1, exclude_firms_without_estimated_params=True, use_antithetic=False):
    print(f"Loading Regime-Switching data from {regime_params_file}...")
    df = pd.read_csv(regime_params_file)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Filter firms if specified
    if gvkey_selected is not None:
        df = df[df['gvkey'].isin(gvkey_selected)]
    
    print(f"Running PARALLELIZED Monte Carlo Regime-Switching simulation:")
    print(f"  Firms: {df['gvkey'].nunique()}")
    print(f"  Dates: {df['date'].nunique() if 'date' in df.columns else 1}")
    print(f"  Simulations per firm: {num_simulations:,}")
    print(f"  Forecast horizon: {num_days} days")
    print(f"  Parallel jobs: {n_jobs}")
    print(f"  Innovation distribution: Student's t")
    print(f"  Antithetic variates: {use_antithetic}")
    print(f"  Exclude rows without estimated RS params: {exclude_firms_without_estimated_params}")
    
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
        delayed(_process_single_date_rs_mc)(
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
    print(f"PARALLELIZED MONTE CARLO REGIME-SWITCHING COMPLETE")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    if not results_df.empty and 'used_default_rs_inputs' in results_df.columns:
        excluded_share = results_df['used_default_rs_inputs'].mean()
        print(f"Rows without estimated RS params (excluded from spreads): {excluded_share:.2%}")
    print(f"{'='*80}\n")
    
    return results_df
