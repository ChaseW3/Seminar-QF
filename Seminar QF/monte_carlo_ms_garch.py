# monte_carlo_ms_garch.py
"""
Monte Carlo Simulation for Proper MS-GARCH(1,1)
===============================================

This module simulates future volatility paths using the MS-GARCH model where:
1. Volatility follows GARCH(1,1) dynamics within EACH regime
2. Regime transitions follow a Markov chain
3. Parameters differ between regimes

Key difference from simple regime-switching MC:
- Volatility evolves with GARCH dynamics, not fixed per regime
- σ²_t = ω_k + α_k * ε²_{t-1} + β_k * σ²_{t-1} in regime k
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
import numba

os.makedirs('./intermediates/', exist_ok=True)


@numba.jit(nopython=True, parallel=True)
def simulate_ms_garch_paths_vectorized(
    omega_0, omega_1, alpha_0, alpha_1, beta_0, beta_1,
    mu_0, mu_1, p00, p11,
    initial_sigma2, initial_regime_probs,
    num_simulations, num_days, num_firms
):
    """
    Vectorized MS-GARCH Monte Carlo simulation with GARCH dynamics per regime.
    
    In each regime k:
        σ²_t = ω_k + α_k * ε²_{t-1} + β_k * σ²_{t-1}
        r_t = μ_k + σ_t * z_t
    
    Parameters:
    -----------
    omega_0, omega_1 : arrays (num_firms,)
        GARCH omega for each regime
    alpha_0, alpha_1 : arrays (num_firms,)
        GARCH alpha for each regime  
    beta_0, beta_1 : arrays (num_firms,)
        GARCH beta for each regime
    mu_0, mu_1 : arrays (num_firms,)
        Mean returns for each regime
    p00, p11 : arrays (num_firms,)
        Regime staying probabilities
    initial_sigma2 : array (num_firms,)
        Initial variance (from last observed period)
    initial_regime_probs : array (num_firms,)
        Probability of starting in regime 1
    num_simulations : int
        Number of MC paths
    num_days : int
        Forecast horizon
    num_firms : int
        Number of firms
        
    Returns:
    --------
    daily_volatilities : array (num_days, num_simulations, num_firms)
        Simulated daily volatilities (σ_t, not σ²_t)
    regime_paths : array (num_days, num_simulations, num_firms)
        Regime states (0 or 1)
    """
    
    daily_volatilities = np.zeros((num_days, num_simulations, num_firms))
    regime_paths = np.zeros((num_days, num_simulations, num_firms), dtype=np.int32)
    
    for sim in numba.prange(num_simulations):
        # Initialize regime for each firm
        regime = np.zeros(num_firms, dtype=np.int32)
        for f in range(num_firms):
            if np.random.uniform() < initial_regime_probs[f]:
                regime[f] = 1
        
        # Initialize variance from last observation
        sigma2 = initial_sigma2.copy()
        
        # Previous epsilon squared for GARCH update
        eps2_prev = np.zeros(num_firms)
        
        for day in range(num_days):
            # Store current regime
            regime_paths[day, sim, :] = regime
            
            # Generate innovations
            z = np.random.standard_normal(num_firms)
            
            for f in range(num_firms):
                # Get parameters based on current regime
                if regime[f] == 0:
                    omega = omega_0[f]
                    alpha = alpha_0[f]
                    beta = beta_0[f]
                    mu = mu_0[f]
                else:
                    omega = omega_1[f]
                    alpha = alpha_1[f]
                    beta = beta_1[f]
                    mu = mu_1[f]
                
                # GARCH(1,1) variance update
                if day > 0:
                    sigma2[f] = omega + alpha * eps2_prev[f] + beta * sigma2[f]
                
                # Ensure positive variance
                sigma2[f] = max(sigma2[f], 1e-10)
                
                # Current volatility
                sigma = np.sqrt(sigma2[f])
                daily_volatilities[day, sim, f] = sigma
                
                # Generate return and compute epsilon squared for next iteration
                eps = sigma * z[f]
                eps2_prev[f] = eps * eps
                
                # Regime transition for next period
                if regime[f] == 0:
                    # In regime 0, switch with prob (1 - p00)
                    if np.random.uniform() > p00[f]:
                        regime[f] = 1
                else:
                    # In regime 1, switch with prob (1 - p11)
                    if np.random.uniform() > p11[f]:
                        regime[f] = 0
    
    return daily_volatilities, regime_paths


def monte_carlo_ms_garch_1year(
    daily_returns_file,
    ms_garch_params_file,
    gvkey_selected=None,
    num_simulations=1000,
    num_days=252
):
    """
    Run Monte Carlo MS-GARCH forecast for 1 year (252 trading days).
    
    For each trading date and firm:
    1. Load MS-GARCH parameters (ω_k, α_k, β_k, p00, p11)
    2. Get current volatility state from data
    3. Run Monte Carlo with GARCH dynamics within each regime
    4. Calculate cumulative volatility over forecast horizon
    
    Parameters:
    -----------
    daily_returns_file : str
        Path to daily returns with MS-GARCH volatility
    ms_garch_params_file : str
        Path to MS-GARCH parameters CSV
    gvkey_selected : int or None
        If None, run for ALL firms
    num_simulations : int
        Number of MC paths (default 1000)
    num_days : int
        Forecast horizon (default 252 trading days)
    
    Returns:
    --------
    results_df : pd.DataFrame
        Cumulative volatility forecasts per firm per date
    """
    
    start_time = pd.Timestamp.now()
    
    print(f"\n{'='*80}")
    print("MONTE CARLO MS-GARCH 1-YEAR VOLATILITY FORECAST")
    print(f"{'='*80}")
    print(f"\nModel: σ²_t = ω_k + α_k * ε²_{{t-1}} + β_k * σ²_{{t-1}} in regime k")
    print(f"Simulations: {num_simulations:,}")
    print(f"Horizon: {num_days} trading days\n")
    
    # Load MS-GARCH parameters
    try:
        params_df = pd.read_csv(ms_garch_params_file)
        print(f"✓ Loaded MS-GARCH parameters for {len(params_df)} firms")
    except FileNotFoundError:
        print(f"✗ File '{ms_garch_params_file}' not found!")
        print(f"   Run MS-GARCH estimation first (ms_garch_proper.py)")
        return pd.DataFrame()
    
    # Load daily returns with MS-GARCH volatility
    df_returns = pd.read_csv(daily_returns_file)
    df_returns['date'] = pd.to_datetime(df_returns['date'])
    df_returns = df_returns.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    print(f"✓ Loaded {len(df_returns):,} daily observations")
    
    # Get unique dates and firms
    unique_dates = sorted(df_returns['date'].unique())
    firm_list = sorted(df_returns['gvkey'].unique())
    
    # Filter to firms with MS-GARCH parameters
    firms_with_params = set(params_df['gvkey'].unique())
    firm_list = [f for f in firm_list if f in firms_with_params]
    
    num_dates = len(unique_dates)
    num_firms_total = len(firm_list)
    
    print(f"✓ Unique dates: {num_dates}")
    print(f"✓ Firms with MS-GARCH params: {num_firms_total}")
    print(f"✓ Date range: {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Expected output rows: ~{num_dates * num_firms_total:,}\n")
    
    if gvkey_selected is not None:
        if gvkey_selected in firm_list:
            firm_list = [gvkey_selected]
            num_firms_total = 1
            print(f"⚠ Filtering to single firm: {gvkey_selected}\n")
        else:
            print(f"✗ Firm {gvkey_selected} not found in parameter file!")
            return pd.DataFrame()
    
    # Index parameters by gvkey for fast lookup
    params_df = params_df.set_index('gvkey')
    
    results_list = []
    
    # Process each date
    for date_idx, date in enumerate(unique_dates):
        if date_idx % max(1, num_dates // 10) == 0:
            elapsed = (pd.Timestamp.now() - start_time).total_seconds()
            print(f"Progress: {date_idx + 1}/{num_dates} ({date.strftime('%Y-%m-%d')}) - {elapsed:.0f}s elapsed")
        
        # Get firms with data on this date
        df_date = df_returns[df_returns['date'] == date]
        firms_on_date = [f for f in df_date['gvkey'].unique() if f in firm_list]
        
        if len(firms_on_date) == 0:
            continue
        
        # PREPARE PARAMETER ARRAYS
        n_firms = len(firms_on_date)
        
        omega_0_arr = np.zeros(n_firms)
        omega_1_arr = np.zeros(n_firms)
        alpha_0_arr = np.zeros(n_firms)
        alpha_1_arr = np.zeros(n_firms)
        beta_0_arr = np.zeros(n_firms)
        beta_1_arr = np.zeros(n_firms)
        mu_0_arr = np.zeros(n_firms)
        mu_1_arr = np.zeros(n_firms)
        p00_arr = np.zeros(n_firms)
        p11_arr = np.zeros(n_firms)
        initial_sigma2_arr = np.zeros(n_firms)
        initial_regime_probs_arr = np.zeros(n_firms)
        
        for f_idx, firm in enumerate(firms_on_date):
            params = params_df.loc[firm]
            
            omega_0_arr[f_idx] = params['omega_0']
            omega_1_arr[f_idx] = params['omega_1']
            alpha_0_arr[f_idx] = params['alpha_0']
            alpha_1_arr[f_idx] = params['alpha_1']
            beta_0_arr[f_idx] = params['beta_0']
            beta_1_arr[f_idx] = params['beta_1']
            mu_0_arr[f_idx] = params['mu_0']
            mu_1_arr[f_idx] = params['mu_1']
            p00_arr[f_idx] = params['p00']
            p11_arr[f_idx] = params['p11']
            
            # Get current volatility from data
            firm_data = df_date[df_date['gvkey'] == firm]
            if 'ms_garch_volatility' in firm_data.columns and not firm_data['ms_garch_volatility'].isna().all():
                current_vol = firm_data['ms_garch_volatility'].values[0]
                initial_sigma2_arr[f_idx] = current_vol ** 2
            else:
                # Use unconditional variance from low-vol regime as fallback
                uncond_var = params['omega_0'] / max(1 - params['alpha_0'] - params['beta_0'], 0.001)
                initial_sigma2_arr[f_idx] = uncond_var
            
            # Initial regime probability (use current regime prob if available)
            if 'ms_garch_regime_prob' in firm_data.columns and not firm_data['ms_garch_regime_prob'].isna().all():
                initial_regime_probs_arr[f_idx] = firm_data['ms_garch_regime_prob'].values[0]
            else:
                initial_regime_probs_arr[f_idx] = 0.5
        
        # RUN MONTE CARLO SIMULATION
        daily_vols, regime_paths = simulate_ms_garch_paths_vectorized(
            omega_0_arr, omega_1_arr, alpha_0_arr, alpha_1_arr, beta_0_arr, beta_1_arr,
            mu_0_arr, mu_1_arr, p00_arr, p11_arr,
            initial_sigma2_arr, initial_regime_probs_arr,
            num_simulations, num_days, n_firms
        )
        
        # CALCULATE STATISTICS
        # daily_vols shape: (num_days, num_simulations, num_firms)
        
        # Mean across simulations for each day
        mean_paths = np.mean(daily_vols, axis=1)  # (num_days, num_firms)
        
        # Cumulative volatility (sum of mean daily vols)
        cumulative_vols = np.sum(mean_paths, axis=0)  # (num_firms,)
        
        # Other statistics
        mean_daily_vols = np.mean(mean_paths, axis=0)
        std_daily_vols = np.std(mean_paths, axis=0)
        min_daily_vols = np.min(mean_paths, axis=0)
        max_daily_vols = np.max(mean_paths, axis=0)
        
        # Regime statistics
        frac_regime_0 = np.mean(regime_paths == 0, axis=(0, 1))
        frac_regime_1 = np.mean(regime_paths == 1, axis=(0, 1))
        
        # STORE RESULTS
        for f_idx, firm in enumerate(firms_on_date):
            results_list.append({
                'gvkey': firm,
                'date': date,
                'mc_msgarch_cumulative_volatility': cumulative_vols[f_idx],
                'mc_msgarch_mean_daily_vol': mean_daily_vols[f_idx],
                'mc_msgarch_std_daily_vol': std_daily_vols[f_idx],
                'mc_msgarch_min_daily_vol': min_daily_vols[f_idx],
                'mc_msgarch_max_daily_vol': max_daily_vols[f_idx],
                'mc_msgarch_frac_regime_0': frac_regime_0[f_idx],
                'mc_msgarch_frac_regime_1': frac_regime_1[f_idx]
            })
    
    results_df = pd.DataFrame(results_list)
    
    # SUMMARY
    print(f"\n{'='*80}")
    print("MS-GARCH MONTE CARLO COMPLETE")
    print(f"{'='*80}\n")
    
    print(f"Total output rows: {len(results_df):,}")
    print(f"Unique firms: {results_df['gvkey'].nunique()}")
    print(f"Unique dates: {results_df['date'].nunique()}")
    
    # Volatility statistics
    print(f"\nCumulative Volatility Statistics:")
    print(f"  Min: {results_df['mc_msgarch_cumulative_volatility'].min():.4f}")
    print(f"  Max: {results_df['mc_msgarch_cumulative_volatility'].max():.4f}")
    print(f"  Mean: {results_df['mc_msgarch_cumulative_volatility'].mean():.4f}")
    print(f"  Median: {results_df['mc_msgarch_cumulative_volatility'].median():.4f}")
    
    # Annualized volatility check
    annualized = results_df['mc_msgarch_cumulative_volatility'] / np.sqrt(252)
    print(f"\nAnnualized Volatility (cumulative/√252):")
    print(f"  Min: {annualized.min()*100:.2f}%")
    print(f"  Max: {annualized.max()*100:.2f}%")
    print(f"  Mean: {annualized.mean()*100:.2f}%")
    print(f"  Median: {annualized.median()*100:.2f}%")
    
    # Check for extreme values
    n_extreme = (annualized > 1.0).sum()
    if n_extreme > 0:
        print(f"\n⚠️  Warning: {n_extreme} observations with annualized vol > 100%")
        extreme_firms = results_df[annualized > 1.0]['gvkey'].unique()
        print(f"   Firms: {extreme_firms.tolist()}")
    
    # Regime statistics
    print(f"\nRegime Fraction Statistics:")
    print(f"  Regime 0 (low vol): mean={results_df['mc_msgarch_frac_regime_0'].mean():.3f}")
    print(f"  Regime 1 (high vol): mean={results_df['mc_msgarch_frac_regime_1'].mean():.3f}")
    
    total_time = (pd.Timestamp.now() - start_time).total_seconds()
    print(f"\nTotal time: {timedelta(seconds=int(total_time))}")
    print(f"Rows per second: {len(results_df) / total_time:.0f}")
    
    return results_df
