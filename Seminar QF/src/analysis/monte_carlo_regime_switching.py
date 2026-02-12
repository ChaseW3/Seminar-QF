
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
    Optimized to use direct Chi-Square generation.
    """
    if nu >= 100:
        return np.random.standard_normal()
    if nu <= 2:
        return np.random.standard_normal()
    
    z = np.random.standard_normal()
    v = np.random.chisquare(nu)
    
    t_sample = z / np.sqrt(v / nu)
    scale = np.sqrt((nu - 2) / nu)
    return t_sample * scale


@numba.jit(nopython=True, fastmath=True, cache=True)
def simulate_regime_switching_vectorized(
    mu_0, mu_1, ar_0, ar_1, sigma_0, sigma_1, nu_0, nu_1,
    trans_00, trans_01, trans_10, trans_11,
    num_simulations, num_firms, horizon_days, initial_regime_probs,
    v0_arr, liability_arr
):
    """
    FULLY OPTIMIZED regime-switching Monte Carlo simulation with T-DISTRIBUTION.
    Vectorized over simulations for SIMD efficiency.
    """
    max_days = np.max(horizon_days)
    n_horizons = len(horizon_days)
    
    # Output arrays -> (Horizon, Firm, Sim)
    terminal_vols = np.zeros((n_horizons, num_firms, num_simulations))
    terminal_assets = np.zeros((n_horizons, num_firms, num_simulations))
    default_indicators = np.zeros((n_horizons, num_firms, num_simulations))
    
    # (Regime, Firm, Sim)
    regime_fractions = np.zeros((2, num_firms, num_simulations))
    
    # Statistics -> (Firm, Sim)
    vol_means = np.zeros((num_firms, num_simulations))
    vol_stds = np.zeros((num_firms, num_simulations))
    vol_maxs = np.zeros((num_firms, num_simulations))
    vol_mins = np.zeros((num_firms, num_simulations))
    
    # Horizon lookup map
    is_horizon = np.zeros(max_days, dtype=np.int32)
    horizon_map = np.full(max_days, -1, dtype=np.int32)
    for h in range(n_horizons):
        d = horizon_days[h] - 1
        if d < max_days: 
            is_horizon[d] = 1
            horizon_map[d] = h
            
    # Parallelize over FIRMS using simple loop (Joblib handles outer dates)
    for f in range(num_firms):
        # Local params
        m0, m1 = mu_0[f], mu_1[f]
        ar0, ar1 = ar_0[f], ar_1[f]
        s0, s1 = sigma_0[f], sigma_1[f]
        n0, n1 = nu_0[f], nu_1[f]
        tr01, tr10 = trans_01[f], trans_10[f]

        prob_regime1 = initial_regime_probs[f]
        v0 = v0_arr[f]
        liability = liability_arr[f]

        # 1. State Initialization (Vectorized)
        
        # Regime: 1 if random < prob, else 0
        regime_1_mask = np.random.random(num_simulations) < prob_regime1
        
        # Initial Volatility depends on regime
        sigma = np.where(regime_1_mask, s1, s0)
        
        asset = np.full(num_simulations, v0)
        r_prev = np.zeros(num_simulations) # AR(1) previous return
        
        regime_0_cnt = np.zeros(num_simulations)
        regime_1_cnt = np.zeros(num_simulations)
        
        m_old = sigma.copy()
        s_old = np.zeros(num_simulations)
        curr_max = sigma.copy()
        curr_min = sigma.copy()
        
        # Precompute t-factors
        t_fact0 = np.sqrt((n0 - 2) / n0) if n0 > 2 else 1.0
        t_fact1 = np.sqrt((n1 - 2) / n1) if n1 > 2 else 1.0

        for day in range(max_days):
            # A. Update Counters
            regime_0_cnt += (~regime_1_mask)
            regime_1_cnt += regime_1_mask
            
            # B. Regime Transition (Vectorized)
            u = np.random.random(num_simulations)
            
            # 0->1: Currently 0 (not 1) AND u < tr01
            switch_to_1 = (~regime_1_mask) & (u < tr01)
            # 1->0: Currently 1 AND u < tr10
            switch_to_0 = (regime_1_mask) & (u < tr10)
            
            # Update mask
            regime_1_mask = (regime_1_mask | switch_to_1) & (~switch_to_0)
            
            # C. Parameter Selection
            mu_curr = np.where(regime_1_mask, m1, m0)
            ar_curr = np.where(regime_1_mask, ar1, ar0)
            sigma = np.where(regime_1_mask, s1, s0)
            nu_curr_is_1 = regime_1_mask # Just use mask logic
            
            # D. Stats Update (Welford)
            curr_max = np.maximum(curr_max, sigma)
            curr_min = np.minimum(curr_min, sigma)
            
            n = day + 1
            delta = sigma - m_old
            m_new = m_old + delta / n
            s_old += delta * (sigma - m_new)
            m_old = m_new
            
            # E. Generate Innovation (T-dist)
            z = np.random.standard_normal(num_simulations)
            
            # Two method approach for mixed t-dist sampling:
            # Generate Chi2 for both nu0 and nu1
            # Protect against nu < 1e-4 or nu being zero
            n0_safe = max(n0, 1e-4)
            n1_safe = max(n1, 1e-4)

            v0 = np.random.chisquare(n0_safe, num_simulations)
            v1 = np.random.chisquare(n1_safe, num_simulations)
            
            # Protect against v=0
            v0 = np.maximum(v0, 1e-12)
            v1 = np.maximum(v1, 1e-12)
            
            t0 = z / np.sqrt(v0 / n0_safe) * t_fact0
            t1 = z / np.sqrt(v1 / n1_safe) * t_fact1
            
            # Handle normal approx cases if nu > 100
            if n0 >= 100: t0 = z
            if n1 >= 100: t1 = z
            
            z_t = np.where(regime_1_mask, t1, t0)
            
            # F. Return Process (AR1)
            # r_t = mu + phi*r_{t-1} + sigma*z
            r_curr = mu_curr + ar_curr * r_prev + sigma * z_t
            r_prev = r_curr
            
            asset *= np.exp(r_curr)
            
            # G. Horizons
            if is_horizon[day]:
                h = horizon_map[day]
                terminal_vols[h, f, :] = sigma
                terminal_assets[h, f, :] = asset
                if liability > 0:
                    default_indicators[h, f, :] = (asset < liability).astype(np.float64)
        
        # Finalize Stats
        vol_means[f, :] = m_old
        vol_stds[f, :] = np.sqrt(s_old / max_days)
        vol_maxs[f, :] = curr_max
        vol_mins[f, :] = curr_min
        
        regime_fractions[0, f, :] = regime_0_cnt / max_days
        regime_fractions[1, f, :] = regime_1_cnt / max_days
    
    return (terminal_vols, terminal_assets, default_indicators,
            vol_means, vol_stds, vol_maxs, vol_mins, regime_fractions)


def _process_single_date_rs_mc(date_data, num_simulations, num_days):
    """
    OPTIMIZED Process Monte Carlo regime-switching simulation for a single date.
    
    Now only stores terminal values at 1y/3y/5y horizons instead of all daily values!
    Memory usage reduced by ~400x.
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
    nu_0_arr = np.zeros(n_firms)
    nu_1_arr = np.zeros(n_firms)
    trans_00_arr = np.zeros(n_firms)
    trans_01_arr = np.zeros(n_firms)
    trans_10_arr = np.zeros(n_firms)
    trans_11_arr = np.zeros(n_firms)
    v0_arr = np.zeros(n_firms)
    liability_arr = np.zeros(n_firms)
    rf_rate_arr = np.full(n_firms, np.nan)
    
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
        nu_0_arr[f_idx] = params['regime_0_nu']
        nu_1_arr[f_idx] = params['regime_1_nu']
        trans_00_arr[f_idx] = params['transition_prob_00']
        trans_01_arr[f_idx] = params['transition_prob_01']
        trans_10_arr[f_idx] = params['transition_prob_10']
        trans_11_arr[f_idx] = params['transition_prob_11']
        
        # Prepare Merton parameters
        if firm in merton_data_dict:
            m_data = merton_data_dict[firm]
            v0_raw = m_data.get('asset_value', 0.0)
            liability_raw = m_data.get('liabilities_total', 0.0)
            rf = m_data.get('risk_free_rate', np.nan)
            
            # Adjust units if needed (ensure decimal)
            if not np.isnan(rf) and abs(rf) > 0.5:
                rf = rf / 100.0
            
            if not np.isnan(v0_raw) and not np.isnan(liability_raw) and v0_raw > 0:
                v0_arr[f_idx] = v0_raw
                liability_arr[f_idx] = liability_raw
                rf_rate_arr[f_idx] = rf
        
        valid_firms.append((f_idx, firm))
    
    if len(valid_firms) == 0:
        return results_list
    
    # Initial regime probabilities
    initial_regime_probs = np.full(n_firms, 0.5)
    
    # Define horizons: 1y, 3y, 5y (trading days)
    horizon_days = np.array([252, 756, 1260], dtype=np.int32)
    
    # Run FULLY OPTIMIZED vectorized Monte Carlo - calculates EVERYTHING in one pass!
    # Output shapes:
    # terminal_vols: (Horizon, Firm, Sim)
    # vol_means: (Firm, Sim)
    (terminal_vols, terminal_assets, default_indicators,
     vol_means, vol_stds, vol_maxs, vol_mins, regime_fractions) = \
        simulate_regime_switching_vectorized(
            mu_0_arr, mu_1_arr, ar_0_arr, ar_1_arr, sigma_0_arr, sigma_1_arr, nu_0_arr, nu_1_arr,
            trans_00_arr, trans_01_arr, trans_10_arr, trans_11_arr,
            num_simulations, n_firms, horizon_days, initial_regime_probs,
            v0_arr, liability_arr
        )
    
    # Extract results for valid firms
    for f_idx, firm in valid_firms:
        # Extract terminal values at each horizon for this firm (updated for vectorized shape)
        firm_vols_1y = terminal_vols[0, f_idx, :]
        firm_vols_3y = terminal_vols[1, f_idx, :]
        firm_vols_5y = terminal_vols[2, f_idx, :]
        
        firm_assets_1y = terminal_assets[0, f_idx, :]
        firm_assets_3y = terminal_assets[1, f_idx, :]
        firm_assets_5y = terminal_assets[2, f_idx, :]
        
        firm_defaulted_1y = default_indicators[0, f_idx, :]
        firm_defaulted_3y = default_indicators[1, f_idx, :]
        firm_defaulted_5y = default_indicators[2, f_idx, :]
        
        # Get summary statistics
        firm_vol_mean = np.mean(vol_means[f_idx, :])
        firm_vol_std = np.mean(vol_stds[f_idx, :])
        firm_vol_max = np.max(vol_maxs[f_idx, :])
        firm_vol_min = np.min(vol_mins[f_idx, :])
        
        # Regime fractions
        regime_0_frac = np.mean(regime_fractions[0, f_idx, :])
        regime_1_frac = np.mean(regime_fractions[1, f_idx, :])
        
        # Calculate PD from pre-computed default indicators
        pd_value_1y = np.mean(firm_defaulted_1y)
        pd_value_3y = np.mean(firm_defaulted_3y)
        pd_value_5y = np.mean(firm_defaulted_5y)
        
        # Calculate CDS spreads
        mc_spread_1y, mc_spread_3y, mc_spread_5y = np.nan, np.nan, np.nan
        mc_debt_1y, mc_debt_3y, mc_debt_5y = np.nan, np.nan, np.nan
        
        if v0_arr[f_idx] > 0 and liability_arr[f_idx] > 0 and not np.isnan(rf_rate_arr[f_idx]):
            rf = rf_rate_arr[f_idx]
            liability = liability_arr[f_idx]
            
            # 1-Year CDS Spread
            expected_payoff_1y = np.mean(np.minimum(firm_assets_1y, liability))
            T_years = 1.0
            discount_factor = np.exp(-rf * T_years)
            debt_val_1y = expected_payoff_1y * discount_factor
            if debt_val_1y > 0 and liability > 0:
                ytm = -np.log(debt_val_1y / liability) / T_years
                mc_spread_1y = max(ytm - rf, 0.0)
                mc_debt_1y = debt_val_1y
            
            # 3-Year CDS Spread
            expected_payoff_3y = np.mean(np.minimum(firm_assets_3y, liability))
            T_years = 3.0
            discount_factor = np.exp(-rf * T_years)
            debt_val_3y = expected_payoff_3y * discount_factor
            if debt_val_3y > 0 and liability > 0:
                ytm = -np.log(debt_val_3y / liability) / T_years
                mc_spread_3y = max(ytm - rf, 0.0)
                mc_debt_3y = debt_val_3y
            
            # 5-Year CDS Spread
            expected_payoff_5y = np.mean(np.minimum(firm_assets_5y, liability))
            T_years = 5.0
            discount_factor = np.exp(-rf * T_years)
            debt_val_5y = expected_payoff_5y * discount_factor
            if debt_val_5y > 0 and liability > 0:
                ytm = -np.log(debt_val_5y / liability) / T_years
                mc_spread_5y = max(ytm - rf, 0.0)
                mc_debt_5y = debt_val_5y
        
        # Legacy compatibility
        pd_value = pd_value_1y

        results_list.append({
            'gvkey': firm,
            'date': date,
            'rs_mean_daily_volatility': firm_vol_mean,
            'rs_fraction_regime_0': regime_0_frac,
            'rs_fraction_regime_1': regime_1_frac,
            'rs_probability_of_default': pd_value,
            'rs_pd_1y': pd_value_1y, # Now Terminal PD
            'rs_pd_3y': pd_value_3y,
            'rs_pd_5y': pd_value_5y,
            'rs_pd_terminal_1y': pd_value_1y, # Explicitly named
            'rs_pd_terminal_3y': pd_value_3y,
            'rs_pd_terminal_5y': pd_value_5y,
            'rs_implied_spread_1y': mc_spread_1y,
            'rs_implied_spread_3y': mc_spread_3y,
            'rs_implied_spread_5y': mc_spread_5y,
            'rs_debt_value_1y': mc_debt_1y,
            'rs_debt_value_3y': mc_debt_3y,
            'rs_debt_value_5y': mc_debt_5y,
        })
    
    return results_list


def monte_carlo_regime_switching_1year_parallel(
    garch_file, 
    regime_params_file,
    gvkey_selected=None, 
    num_simulations=1000, 
    num_days=1260,
    n_jobs=-1,
    merton_df=None
):
    """
    PARALLELIZED Monte Carlo regime-switching forecast for 1-5 years (default 1260 days/5 years).
    
    This version processes different dates in parallel for significant speedup.
    """
    start_time = pd.Timestamp.now()
    
    print(f"\n{'='*80}")
    print("PARALLELIZED MONTE CARLO REGIME-SWITCHING 1-YEAR FORECAST")
    print(f"{'='*80}\n")
    
    # Load regime parameters
    if isinstance(regime_params_file, str):
        print(f"Loading regime parameters from {regime_params_file}...")
        try:
            regime_params = pd.read_csv(regime_params_file)
        except Exception as e:
            print(f"✗ File '{regime_params_file}' not found or error loading: {e}")
            return pd.DataFrame()
    else:
        print(f"Using provided regime parameters DataFrame...")
        regime_params = regime_params_file.copy()

    # Check for new 'nu' columns from updated estimator
    if 'regime_0_nu' not in regime_params.columns:
        print("⚠ Warning: 'regime_0_nu' not found in parameters from parallel function. Using default df=100 (Normal approx).")
        regime_params['regime_0_nu'] = 100.0
        regime_params['regime_1_nu'] = 100.0

    print(f"✓ Loaded regime-switching parameters for {len(regime_params)} firms")
    
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
            'regime_0_nu': row['regime_0_nu'],
            'regime_1_nu': row['regime_1_nu'],
            'transition_prob_00': row['transition_prob_00'],
            'transition_prob_01': row['transition_prob_01'],
            'transition_prob_10': row['transition_prob_10'],
            'transition_prob_11': row['transition_prob_11'],
        }
    
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

    # Load GARCH base data
    if isinstance(garch_file, str):
        df_garch = pd.read_csv(garch_file)
    else:
        df_garch = garch_file.copy()
        
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
                    'liabilities_total': row['liabilities_total'],
                    'risk_free_rate': row['risk_free_rate']
                }
        
        date_groups.append((date, group, regime_params_dict, firms_on_date, merton_date_dict))
    
    print(f"\nProcessing {len(date_groups)} dates in parallel...")
    
    # OPTIMIZATION: Avoid Joblib overhead for single date / Numba interaction
    use_joblib = True
    if len(date_groups) == 1:
        use_joblib = False
    elif len(date_groups) < n_jobs and n_jobs > 1:
        pass

    if use_joblib:
        # Parallel processing
        print(f"Refuting Numba-parallelism to Joblib workers (dates={len(date_groups)})")
        results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_process_single_date_rs_mc)(date_data, num_simulations, num_days)
            for date_data in date_groups
        )
    else:
        # Run directly in main process - allows Numba to use all cores effectively
        print("Optimization: Running simulation directly (avoiding Joblib overhead for single/few batches)")
        results_nested = [
            _process_single_date_rs_mc(date_data, num_simulations, num_days)
            for date_data in date_groups
        ]
    
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
