# garch_model.py

import pandas as pd
import numpy as np
from arch import arch_model
from pathlib import Path

# Import config for output paths
try:
    from src.utils import config
    OUTPUT_DIR = config.OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "output"

def sanitize_garch_params(params_row):
    # Clean up estimated GARCH params before saving — clips/repairs values that would
    # break the Monte Carlo or produce economically nonsensical simulations
    p = params_row.copy()
    flags = []
    is_repaired = False
    is_rejected = False
    
    # Nu must be in [2.1, 200]: below 2.1 the t-distribution has infinite variance,
    # above 200 it's indistinguishable from normal so we just treat it as such
    min_nu = 2.1
    max_nu = 200.0
    if pd.isna(p['nu']):
        p['nu'] = max_nu  # treat missing nu as normal distribution
        flags.append('imputed_nu_normal')
        is_repaired = True
    elif p['nu'] < min_nu:
        p['nu'] = min_nu
        flags.append('clipped_nu_min')
        is_repaired = True
    elif p['nu'] > max_nu:
        p['nu'] = max_nu
        flags.append('clipped_nu_max')
        is_repaired = True
        
    # Clip alpha and beta to non-negative (arch usually enforces this but just in case)
    if p['alpha'] < 0:
        p['alpha'] = 0.0
        flags.append('clipped_alpha_zero')
        is_repaired = True
    
    if p['beta'] < 0:
        p['beta'] = 0.0
        flags.append('clipped_beta_zero')
        is_repaired = True
        
    # Flag very low beta — implies almost no volatility memory, likely an unstable fit
    min_beta = 0.05
    if p['beta'] < min_beta:
        flags.append('low_beta_warning')

    # Enforce stationarity: alpha + beta must be < 0.999
    # If breached, reduce beta to bring persistence below the cap
    max_persistence = 0.999
    persistence = p['alpha'] + p['beta']
    
    if persistence >= max_persistence:
        new_beta = max_persistence - p['alpha']
        if new_beta < 0:
            new_beta = 0.0
            p['alpha'] = max_persistence  # alpha itself is too large
             
        p['beta'] = new_beta
        flags.append('renormalized_persistence')
        is_repaired = True
    
    p['persistence'] = p['alpha'] + p['beta']
    
    # Floor omega to avoid numerical underflow in variance calculations
    # In decimal return scale, daily variance is typically ~1e-4, so omega is at most ~1e-6.
    # Anything below 1e-9 is effectively zero and will cause problems.
    if p['omega'] < 1e-9:
        p['omega'] = 1e-9
        flags.append('floored_omega')
        is_repaired = True
        
    p['is_repaired'] = is_repaired
    p['is_rejected'] = is_rejected
    p['repair_flags'] = ";".join(flags)
    
    return p

def run_garch_estimation(daily_returns_df):
    # Fit a rolling GARCH(1,1)-t on daily asset returns, one 252-day window per month per firm.
    # Returns are scaled ×100 before fitting for numerical stability, then parameters are
    # rescaled back to decimal units before saving.
    print("Estimating GARCH(1,1) with t-distribution on DAILY Asset Returns...")
    
    if daily_returns_df.empty:
        print("No daily returns provided for GARCH.")
        return daily_returns_df
        
    df_out = daily_returns_df.copy()
    
    firms = df_out["gvkey"].unique()
    print(f"Processing GARCH for {len(firms)} firms (Daily Data)...")
    
    SCALE_FACTOR = 100.0  # Scale returns to percentage form
    
    # Initialize list to store parameters for each firm
    params_list = []
    
    # Diagnostics Counters
    diag_total = 0
    diag_converged = 0
    diag_rejected_convergence = 0
    diag_rejected_stationarity = 0
    diag_repaired = 0
    
    for i, gvkey in enumerate(firms):
        mask = df_out["gvkey"] == gvkey
        firm_ts = df_out.loc[mask].copy()
        
        # Ensure date column is datetime
        if 'date' not in firm_ts.columns:
             if isinstance(firm_ts.index, pd.DatetimeIndex):
                 firm_ts['date'] = firm_ts.index
             else:
                 print(f"Skipping {gvkey}: No date column found")
                 continue
                 
        firm_ts['date'] = pd.to_datetime(firm_ts['date'])
        firm_ts = firm_ts.sort_values('date')
        
        # Determine rolling windows (Monthly)
        start_date = firm_ts['date'].min()
        end_date = firm_ts['date'].max()
        
        # Start 12 months (1 year) in
        try:
            estimation_start = start_date + pd.DateOffset(months=12)
            if estimation_start >= end_date:
                continue
            month_ends = pd.date_range(start=estimation_start, end=end_date, freq='ME')
        except Exception as e:
            print(f"Error defining date range for {gvkey}: {e}")
            continue
            
        if (i+1) % 10 == 0:
            print(f"[{i+1}/{len(firms)}] Processing {gvkey}...")
        
        last_params = None
        
        for date_point in month_ends:
            # Select all data up to this point
            data_up_to_point = firm_ts[firm_ts['date'] <= date_point]

            # Need at least 252 trading days to fill one full window
            if len(data_up_to_point) < 252:
                continue

            # Use the most recent 252 days as the estimation window
            window_df = data_up_to_point.iloc[-252:].copy()
            
            # Skip windows with too many gaps in returns
            if window_df['asset_return_daily'].isna().sum() > 10: 
                 continue
                 
            last_trading_date = window_df['date'].max()
  
            if "asset_return_daily_scaled" in window_df.columns:
                returns = window_df["asset_return_daily_scaled"].dropna().values
            else:
                window_df = window_df.dropna(subset=["asset_return_daily"])
                if len(window_df) < 200: continue  # need at least 200 valid obs
                returns = window_df["asset_return_daily"].values * SCALE_FACTOR
            
            try:
                diag_total += 1
                
                # GARCH(1,1) with Student-t — arch enforces alpha+beta < 1 by default
                am = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
                
                # Warm start from last window if available
                start_vals = last_params.values if last_params is not None else None
                
                res = am.fit(starting_values=start_vals, disp='off', show_warning=False)
                
                # Check whether the optimizer actually converged
                converged = (hasattr(res, 'optimization_result') and res.optimization_result.success) or \
                           (hasattr(res, 'convergence_flag') and res.convergence_flag == 0)
                
                if not converged:
                    diag_rejected_convergence += 1
                    continue
                
                diag_converged += 1
                
                # Extract parameters (still in scaled space at this point)
                omega_est = res.params['omega']
                alpha_est = res.params['alpha[1]']
                beta_est = res.params['beta[1]']
                nu_est = res.params.get('nu', np.nan) 
                mu_est = res.params.get('mu', 0.0) 

                # Rescale omega and mu back to decimal return units (omega scales with SCALE^2)
                omega = omega_est / (SCALE_FACTOR ** 2)
                mu = mu_est / SCALE_FACTOR
                alpha = alpha_est
                beta = beta_est
                nu = nu_est
                
                # Last conditional vol is in scaled units — divide to get decimal daily vol
                last_cond_vol_scaled = res.conditional_volatility[-1]
                sigma0 = last_cond_vol_scaled / SCALE_FACTOR
                
                # Store parameters for this window
                params_row = {
                    'gvkey': gvkey,
                    'date': last_trading_date,
                    'omega': omega,
                    'alpha': alpha,
                    'beta': beta,
                    'nu': nu,
                    'mu': mu,
                    'sigma0': sigma0,  # last conditional vol, used as starting vol in Monte Carlo
                    'persistence': alpha + beta,
                    'unconditional_variance': omega / (1 - alpha - beta) if (alpha + beta) < 1 else np.nan,
                    'log_likelihood': getattr(res, 'loglikelihood', getattr(res, 'llf', np.nan)),
                    'aic': getattr(res, 'aic', np.nan),
                    'bic': getattr(res, 'bic', np.nan),
                    'num_observations': len(returns)
                }
                
                # Clip/repair any out-of-range values before saving
                sanitized = sanitize_garch_params(params_row)
                
                if sanitized['is_rejected']:
                    diag_rejected_stationarity += 1
                    continue
                    
                if sanitized['is_repaired']:
                    diag_repaired += 1
                
                params_list.append(sanitized)
                
                # Pass raw estimated params as warm start (not the sanitized ones)
                # since the optimizer expects the structure of res.params
                last_params = res.params
                
            except Exception as e:
                # print(f"Error estimating GARCH for {gvkey} on {date_point}: {e}")
                continue

    print("\nGARCH ESTIMATION DIAGNOSTICS:")
    print(f"  Total Windows Attempted: {diag_total}")
    
    val_conv_rate = (diag_converged/diag_total) if diag_total > 0 else 0
    print(f"  Converged:               {diag_converged} ({val_conv_rate:.1%} of total)")
    print(f"  Rejected (Convergence):  {diag_rejected_convergence}")
    print(f"  Rejected (Logic/Other):  {diag_rejected_stationarity}")
    
    val_repaired_rate = (diag_repaired/diag_converged) if diag_converged > 0 else 0
    print(f"  Repaired (Stationarity): {diag_repaired} ({val_repaired_rate:.1%} of converged)")
    
    # Save parameters for Monte Carlo and reference
    if params_list:
        params_df = pd.DataFrame(params_list)
        output_path = OUTPUT_DIR / 'garch_parameters.csv'
        params_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved GARCH parameters to '{output_path}'")
        
        # Merge rolling parameters back into daily dataframe
        
        # Ensure date type
        df_out['date'] = pd.to_datetime(df_out['date'])
        
        # Prepare params for merge (add garch_ prefix to avoid column name collisions)
        merge_df = params_df[['gvkey', 'date', 'omega', 'alpha', 'beta', 'nu', 'mu', 'sigma0']].rename(columns={
            'omega': 'garch_omega',
            'alpha': 'garch_alpha',
            'beta': 'garch_beta',
            'nu': 'garch_nu',
            'mu': 'garch_mu_daily',
            'sigma0': 'garch_sigma0'
        })
        
        # Drop existing columns to avoid conflicts
        drop_cols = ['garch_omega', 'garch_alpha', 'garch_beta', 'garch_nu', 'garch_mu_daily', 'garch_volatility', 'garch_sigma0']
        df_out = df_out.drop(columns=[c for c in drop_cols if c in df_out.columns])
        
        # Merge on date (month-ends only match initially)
        df_out = pd.merge(df_out, merge_df, on=['gvkey', 'date'], how='left')
        
        # Forward fill parameters per firm
        df_out = df_out.sort_values(['gvkey', 'date'])
        fill_cols = ['garch_omega', 'garch_alpha', 'garch_beta', 'garch_nu', 'garch_mu_daily', 'garch_sigma0']
        df_out[fill_cols] = df_out.groupby('gvkey')[fill_cols].ffill()
        
        # Calculate daily volatility for convenience — prefer last conditional vol,
        # fall back to unconditional sqrt(omega/(1-a-b)) where sigma0 is missing
        df_out['garch_volatility'] = df_out['garch_sigma0']
        
        mask_nan = df_out['garch_volatility'].isna() & df_out['garch_omega'].notna()
        if mask_nan.any():
            df_out.loc[mask_nan, 'garch_volatility'] = np.sqrt(
                df_out.loc[mask_nan, 'garch_omega'] / 
                (1 - df_out.loc[mask_nan, 'garch_alpha'] - df_out.loc[mask_nan, 'garch_beta'])
            )
        
    return df_out
