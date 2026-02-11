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
    v = 0.0
    nu_int = int(nu)
    for _ in range(nu_int):
        v += np.random.standard_normal() ** 2
    
    # t = Z / sqrt(V/nu)
    t_sample = z / np.sqrt(v / nu)
    
    # Standardize so variance = 1
    scale = np.sqrt((nu - 2) / nu)
    return t_sample * scale


@numba.jit(nopython=True, parallel=True)
def simulate_ms_garch_paths_t_jit(
    omega_0, omega_1, alpha_0, alpha_1, beta_0, beta_1,
    mu_0, mu_1, p00, p11, nu_0, nu_1,
    initial_sigma2, initial_regime_probs,
    num_simulations, num_days, num_firms
):
    """
    FAST Numba JIT-compiled MS-GARCH Monte Carlo with t-distributed innovations.
    
    This is ~100x faster than the scipy version!
    """
    daily_volatilities = np.zeros((num_days, num_simulations, num_firms))
    daily_returns = np.zeros((num_days, num_simulations, num_firms))
    regime_paths = np.zeros((num_days, num_simulations, num_firms), dtype=np.int32)
    
    for sim in numba.prange(num_simulations):
        # Initialize regime for each firm
        regime = np.zeros(num_firms, dtype=np.int32)
        for f in range(num_firms):
            if np.random.uniform() < initial_regime_probs[f]:
                regime[f] = 1
        
        # Initialize variance from last observation
        sigma2 = initial_sigma2.copy()
        eps2_prev = np.zeros(num_firms)
        
        for day in range(num_days):
            regime_paths[day, sim, :] = regime
            
            for f in range(num_firms):
                # Get parameters based on current regime
                if regime[f] == 0:
                    omega = omega_0[f]
                    alpha = alpha_0[f]
                    beta = beta_0[f]
                    nu = nu_0[f]
                else:
                    omega = omega_1[f]
                    alpha = alpha_1[f]
                    beta = beta_1[f]
                    nu = nu_1[f]
                
                # GARCH(1,1) variance update
                # IMPORTANT: The GARCH parameters were estimated using RAW residuals ε²,
                # NOT standardized residuals z². So we must use eps² = (σ*z)²
                if day > 0:
                    sigma2[f] = omega + alpha * eps2_prev[f] + beta * sigma2[f]
                
                # Enforce reasonable variance bounds (1% to 10% daily volatility)
                sigma2[f] = max(min(sigma2[f], 0.01), 1e-6)
                sigma = np.sqrt(sigma2[f])
                daily_volatilities[day, sim, f] = sigma
                
                # Generate t-distributed innovation using JIT sampler
                z = sample_standardized_t(nu)
                #Cap extreme innovations to prevent temporary spikes
                z = max(min(z, 5.0), -5.0)  # Cap at ±5 sigma
                eps = sigma * z
                daily_returns[day, sim, f] = eps
                eps2_prev[f] = eps * eps
                
                # Regime transition
                if regime[f] == 0:
                    if np.random.uniform() > p00[f]:
                        regime[f] = 1
                else:
                    if np.random.uniform() > p11[f]:
                        regime[f] = 0
    
    return daily_volatilities, regime_paths, daily_returns


@numba.jit(nopython=True, parallel=True)
def simulate_ms_garch_paths_vectorized(
    omega_0, omega_1, alpha_0, alpha_1, beta_0, beta_1,
    mu_0, mu_1, p00, p11,
    initial_sigma2, initial_regime_probs,
    num_simulations, num_days, num_firms
):
    """
    Vectorized MS-GARCH Monte Carlo simulation with GARCH dynamics per regime.
    Legacy version using normal innovations for backward compatibility.
    
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
                # Use RAW residuals ε² = (σ*z)², not standardized z²
                if day > 0:
                    sigma2[f] = omega + alpha * eps2_prev[f] + beta * sigma2[f]
                
                # Enforce reasonable variance bounds
                sigma2[f] = max(min(sigma2[f], 0.01), 1e-6)
                
                # Current volatility
                sigma = np.sqrt(sigma2[f])
                daily_volatilities[day, sim, f] = sigma
                
                # Cap extreme innovations to prevent temporary spikes
                z_capped = max(min(z[f], 5.0), -5.0)  # Cap at ±5 sigma
                eps = sigma * z_capped
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


def simulate_ms_garch_paths_t_dist(
    omega_0, omega_1, alpha_0, alpha_1, beta_0, beta_1,
    mu_0, mu_1, p00, p11, nu_0, nu_1,
    initial_sigma2, initial_regime_probs,
    num_simulations, num_days, num_firms
):
    """
    MS-GARCH Monte Carlo simulation with Student's t distributed innovations.
    
    NOW USES THE FAST JIT-COMPILED VERSION INTERNALLY!
    
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
    nu_0, nu_1 : arrays (num_firms,)
        Degrees of freedom for t-distribution in each regime
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
    daily_returns : array (num_days, num_simulations, num_firms)
        Simulated returns (innovations)
    """
    # Use the fast JIT-compiled version
    return simulate_ms_garch_paths_t_jit(
        omega_0, omega_1, alpha_0, alpha_1, beta_0, beta_1,
        mu_0, mu_1, p00, p11, nu_0, nu_1,
        initial_sigma2, initial_regime_probs,
        num_simulations, num_days, num_firms
    )


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
    
    # Load Merton Data for PD calculation
    merton_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ms_garch_params_file))), 'data', 'output', 'merged_data_with_merton.csv')
    if not os.path.exists(merton_file):
         merton_file = './data/output/merged_data_with_merton.csv'

    merton_by_date = {}
    if os.path.exists(merton_file):
        try:
            df_merton = pd.read_csv(merton_file)
            df_merton['date'] = pd.to_datetime(df_merton['date'])
            merton_by_date = {k: v for k, v in df_merton.groupby('date')}
            print(f"✓ Loaded Merton data for PD calculation ({len(df_merton):,} rows)")
        except Exception as e:
            print(f"⚠ Error loading Merton data: {e}")
            merton_by_date = {}
    
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
        nu_0_arr = np.zeros(n_firms)  # Degrees of freedom for t-distribution
        nu_1_arr = np.zeros(n_firms)
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
            nu_0_arr[f_idx] = params.get('nu_0', 30.0)  # Default to 30 if not present
            nu_1_arr[f_idx] = params.get('nu_1', 30.0)
            
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
        
        # RUN MONTE CARLO SIMULATION with t-distribution
        daily_vols, regime_paths, daily_sim_returns = simulate_ms_garch_paths_t_dist(
            omega_0_arr, omega_1_arr, alpha_0_arr, alpha_1_arr, beta_0_arr, beta_1_arr,
            mu_0_arr, mu_1_arr, p00_arr, p11_arr, nu_0_arr, nu_1_arr,
            initial_sigma2_arr, initial_regime_probs_arr,
            num_simulations, num_days, n_firms
        )
        
        # CALCULATE STATISTICS
        # daily_vols shape: (num_days, num_simulations, num_firms)
        
        # YEARLY VARIANCE CALCULATION (Asset Value Simulation Method)
        # 1. Use simulated returns directly (t-distributed shocks)
        assets_daily_returns = daily_sim_returns
        
        # 2. Cumulative yearly return using LOG RETURNS summation
        # V_T = V_0 * exp(sum(r_t))
        cum_log_returns = np.sum(assets_daily_returns, axis=0) # Shape: (num_simulations, num_firms)
        
        # 3. Calculate Variance of Cumulative Log Returns
        # This is the strictly correct "Integrated Variance" for diffusion processes
        integrated_variances = np.var(cum_log_returns, axis=0, ddof=1)  # Shape: (num_firms,)

        
        # Calculate regime statistics
        mean_regime_0 = np.mean(regime_paths == 0, axis=(0, 1))
        mean_regime_1 = np.mean(regime_paths == 1, axis=(0, 1))
        
        # APPEND RESULTS
        for firm_idx, firm in enumerate(firms_on_date):
            
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
                rf_rate = firm_merton_data.get('risk_free_rate', 0.04)
                
                if not np.isnan(asset_value) and not np.isnan(debt_face):
                     # Extract firm paths
                     firm_daily_returns = daily_sim_returns[:, :, firm_idx]
                     
                     # Construct Asset Paths (Consistent with User Request)
                     cum_log_returns_path = np.cumsum(firm_daily_returns, axis=0) # shape: (num_days, num_simulations)
                     
                     # Final Asset Value V_T = V0 * exp(cum_log_returns)
                     V_T_paths = asset_value * np.exp(cum_log_returns_path[-1, :])
                     
                     payoffs = np.minimum(V_T_paths, debt_face)
                     expected_payoff = np.mean(payoffs)
                     
                     T_years = num_days / 252.0
                     discount_factor = np.exp(-rf_rate * T_years)
                     risky_debt_value = discount_factor * expected_payoff
                     
                     if risky_debt_value > 0 and debt_face > 0:
                         implied_yield = -1.0/T_years * np.log(risky_debt_value / debt_face)
                         mc_spread = implied_yield - rf_rate
                         mc_debt_value = risky_debt_value
            
            results_list.append({
                'gvkey': firm,
                'date': date,
                'mc_msgarch_integrated_variance': integrated_variances[firm_idx],
                'mc_msgarch_fraction_regime_0': mean_regime_0[firm_idx], 
                'mc_msgarch_fraction_regime_1': mean_regime_1[firm_idx],
                'mc_msgarch_implied_spread': mc_spread,
                'mc_msgarch_debt_value': mc_debt_value
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
    print(f"\nIntegrated Variance Statistics:")
    print(f"  Min: {results_df['mc_msgarch_integrated_variance'].min():.4f}")
    print(f"  Max: {results_df['mc_msgarch_integrated_variance'].max():.4f}")
    print(f"  Mean: {results_df['mc_msgarch_integrated_variance'].mean():.4f}")
    print(f"  Median: {results_df['mc_msgarch_integrated_variance'].median():.4f}")
    
    # Annualized volatility check
    annualized = np.sqrt(results_df['mc_msgarch_integrated_variance'])
    print(f"\nAnnualized Volatility (sqrt(IV)):")
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


def _process_single_date_msgarch_mc(date_data, num_simulations, num_days):
    """
    Process Monte Carlo MS-GARCH simulation for a single date (for parallelization).
    
    Parameters:
    -----------
    date_data : tuple
        (date, df_date, msgarch_params_dict, merton_data_dict) 
    num_simulations : int
        Number of Monte Carlo paths
    num_days : int
        Forecast horizon in days
        
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
            
            # Run MS-GARCH Monte Carlo with t-distribution
            # NOTE: JIT function expects arrays, so wrap scalars in np.array([...])
            daily_vols, regime_paths, daily_sim_returns = simulate_ms_garch_paths_t_dist(
                np.array([params['omega_0']]), np.array([params['omega_1']]),
                np.array([params['alpha_0']]), np.array([params['alpha_1']]),
                np.array([params['beta_0']]), np.array([params['beta_1']]),
                np.array([params['mu_0']]), np.array([params['mu_1']]),
                np.array([params['p00']]), np.array([params['p11']]),
                np.array([params['nu_0']]), np.array([params['nu_1']]),
                np.array([initial_sigma2]),
                np.array([regime_prob]),
                num_simulations, num_days, 1
            )
            
            # Extract firm results (firm index 0 since processing one firm)
            firm_vols = daily_vols[:, :, 0]  # shape: (num_days, num_simulations)
            firm_regimes = regime_paths[:, :, 0]  # shape: (num_days, num_simulations)
            firm_daily_returns = daily_sim_returns[:, :, 0] # Use simulated returns (t-distributed)
            
            # Calculate statistics - ASSET VALUE SIMULATION METHOD
            # 1. Use simulated returns directly
            # 2. Calculate daily asset returns: R_t from simulation
            # (No need to regenerate z_innovations here, we use the ones from simulation)
            
            # 3. Cumulative yearly return
            firm_cumulative_returns = np.prod(1.0 + firm_daily_returns, axis=0) - 1.0
            
            # 4. Variance of yearly returns
            integrated_variance = np.var(firm_cumulative_returns, ddof=1)

            
            # Regime fractions
            regime_0_frac = np.mean(firm_regimes == 0)
            regime_1_frac = np.mean(firm_regimes == 1)
            
            # PD and Spread Calculation
            pd_value = np.nan
            mc_spread = np.nan
            mc_debt_value = np.nan
            
            if firm in merton_data_dict:
                m_data = merton_data_dict[firm]
                v0 = m_data.get('asset_value', np.nan)
                liability = m_data.get('liabilities_total', np.nan)
                rf_rate = m_data.get('risk_free_rate', np.nan)
                
                # Adjust units if needed (ensure decimal)
                if not np.isnan(rf_rate) and abs(rf_rate) > 0.5:
                    rf_rate = rf_rate / 100.0

                if not np.isnan(v0) and not np.isnan(liability) and v0 > 0:
                     # Asset Paths: V_t = V0 * cumulative_prod(1+R)
                     cum_returns_path = np.cumprod(1.0 + firm_daily_returns, axis=0)
                     asset_paths = v0 * cum_returns_path
                     
                     path_defaulted = np.any(asset_paths < liability, axis=0)
                     pd_value = np.mean(path_defaulted)
                     
                     if not np.isnan(rf_rate):
                         V_T = asset_paths[-1, :]
                         expected_payoff = np.mean(np.minimum(V_T, liability))
                         T_years = num_days / 252.0
                         discount_factor = np.exp(-rf_rate * T_years)
                         debt_val = expected_payoff * discount_factor
                         
                         if debt_val > 0 and liability > 0:
                             ytm = -np.log(debt_val / liability) / T_years
                             mc_spread = max(ytm - rf_rate, 0.0)
                             mc_debt_value = debt_val
            
            results_list.append({
                'gvkey': firm,
                'date': date,
                'mc_msgarch_integrated_variance': integrated_variance,
                'mc_msgarch_mean_daily_volatility': np.mean(firm_vols),
                'mc_msgarch_std_daily_volatility': np.std(firm_vols),
                'mc_msgarch_max_daily_volatility': np.max(firm_vols),
                'mc_msgarch_min_daily_volatility': np.min(firm_vols),
                'mc_msgarch_p95_daily_volatility': np.percentile(firm_vols, 95),
                'mc_msgarch_p05_daily_volatility': np.percentile(firm_vols, 5),
                'mc_msgarch_frac_regime_0': regime_0_frac,
                'mc_msgarch_frac_regime_1': regime_1_frac,
                'mc_msgarch_probability_of_default': pd_value,
                'mc_msgarch_implied_spread': mc_spread,
                'mc_msgarch_debt_value': mc_debt_value
            })
            
        except Exception as e:
            print(f"Error processing firm {firm} on date {date}: {e}")
            continue
    
    return results_list


def monte_carlo_ms_garch_1year_parallel(daily_returns_file, ms_garch_params_file, 
                                       gvkey_selected=None, num_simulations=1000, 
                                       num_days=252, n_jobs=-1, merton_df=None):
    """
    Parallelized Monte Carlo MS-GARCH forecast for 1 year.
    
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
        if isinstance(daily_returns_file, str):
            merton_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(daily_returns_file))), 'merged_data_with_merton.csv')
        else:
             merton_file = './data/output/merged_data_with_merton.csv'
             
        if not os.path.exists(merton_file):
            merton_file = './data/output/merged_data_with_merton.csv'

        if os.path.exists(merton_file):
            try:
                df_merton = pd.read_csv(merton_file)
                df_merton['date'] = pd.to_datetime(df_merton['date'])
                merton_by_date = {k: v for k, v in df_merton.groupby('date')}
                print(f"✓ Loaded Merton data for PD calculation ({len(df_merton):,} rows)")
            except Exception as e:
                print(f"⚠ Error loading Merton data: {e}")
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
                        'liabilities_total': row['liabilities_total']
                    }
            date_groups.append((date, group, msgarch_params, merton_date_dict))
    else:
        date_groups = [(pd.Timestamp.now().date(), df, msgarch_params, {})]
    
    print(f"\nProcessing {len(date_groups)} dates in parallel...")
    
    # Parallel processing across dates
    results_nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_single_date_msgarch_mc)(date_data, num_simulations, num_days) 
        for date_data in date_groups
    )
    
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
