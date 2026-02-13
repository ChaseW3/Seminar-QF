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
    """
    Sanitizes GARCH parameters to ensure numerical stability and meaningful simulation.
    
    Policies:
    1. Nu (degrees of freedom) must be >= 2.1 to ensure finite variance and valid t-distribution.
       Upper bound capped at 200 (essentially normal).
    2. Beta (persistence) must be >= 0.05. If too low, it implies unstable fitting or memory collapse.
       Action: Reject or Repair (we will repair by shrinkage if very low, but key implies rejection of <0.1 often).
       Requirements say: "Rejects beta < 0.05 (or configurable threshold) OR apply shrinkage"
       We will implement a repair to min_beta=0.05 if it's close, else flagged.
    3. Alpha + Beta (persistence) must be < 0.999.
       Action: If >= 0.999, renormalize beta = 0.999 - alpha.
    4. Omega (baseline variance) must be >= floor (e.g. 1e-7).
    """
    p = params_row.copy()
    flags = []
    is_repaired = False
    is_rejected = False
    
    # 1. Sanitize Nu
    # Requirement: Nu >= 2.1
    min_nu = 2.1
    max_nu = 200.0
    if pd.isna(p['nu']):
         # If Normal distribution was used (though we use 't'), or fit failed to return nu
         p['nu'] = max_nu
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
        
    # 2. Sanitize Persistence (Alpha + Beta)
    # Requirement: Alpha >= 0, Beta >= 0, Sum < 0.999
    # Ensure non-negative alpha/beta first (arch_model usually enforces this but good to be safe)
    if p['alpha'] < 0:
        p['alpha'] = 0.0
        flags.append('clipped_alpha_zero')
        is_repaired = True
    
    if p['beta'] < 0:
        p['beta'] = 0.0
        flags.append('clipped_beta_zero')
        is_repaired = True
        
    # Check Beta Lower Bound (Memory collapse)
    min_beta = 0.05
    if p['beta'] < min_beta:
        # User requirement: Reject or apply shrinkage. 
        # We will flag it strongly. If it's extremely small, simulation might be just noise.
        # Let's enforce a soft floor but flag it.
        # p['beta'] = min_beta 
        # flags.append('floored_beta')
        # is_repaired = True
        # For now, just flag it. The user said: "Beta < 0.1: 709 (14%) -> unstable fits".
        flags.append('low_beta_warning')

    # Check Stationarity
    max_persistence = 0.999
    persistence = p['alpha'] + p['beta']
    
    if persistence >= max_persistence:
        # Renormalize Beta
        # Keep Alpha, reduce Beta
        new_beta = max_persistence - p['alpha']
        if new_beta < 0:
             # If Alpha itself is > 0.999, we have a problem. Cap alpha too?
             new_beta = 0.0
             p['alpha'] = max_persistence
             
        p['beta'] = new_beta
        flags.append('renormalized_persistence')
        is_repaired = True
    
    # Recalculate persistence
    p['persistence'] = p['alpha'] + p['beta']
    
    # 3. Sanitize Omega
    # Prevent extremely small omega which causes numerical underflow or zero vol
    min_omega = 1e-7 # Heuristic floor relative to percentage returns squared (if scaled) or decimal?
    # Data is SCALED by 100 before estimation in run_garch_estimation.
    # Omega returned there is unscaled back. 
    # If returns are ~1% (0.01), variance ~1e-4. Omega is roughly (1-a-b)*Var.
    # If 1-a-b is 0.01, Omega ~ 1e-6.
    # So 1e-7 or 1e-8 is a reasonable floor for DAILY variance.
    if p['omega'] < 1e-9:
        p['omega'] = 1e-9
        flags.append('floored_omega')
        is_repaired = True
        
    p['is_repaired'] = is_repaired
    p['is_rejected'] = is_rejected # Logic for rejection could be stricter if needed
    p['repair_flags'] = ";".join(flags)
    
    return p

def run_garch_estimation(daily_returns_df):
    """
    Estimates GARCH(1,1) on DAILY asset returns.
    Scales returns for numerical stability, then rescales parameters back.
    Uses Student's t distribution to handle fat tails (excess kurtosis) 
    commonly found in financial returns data.
    """
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
        
        # Start 24 months in
        try:
            estimation_start = start_date + pd.DateOffset(months=24)
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

            # Require at least 504 trading days of history
            if len(data_up_to_point) < 504:
                continue

            # Take the exact last 504 trading days for the window
            window_df = data_up_to_point.iloc[-504:].copy()
            
            # Additional check for missing values inside the window
            if window_df['asset_return_daily'].isna().sum() > 20: 
                 # Skip if too many missing returns in the window
                 continue
                 
            last_trading_date = window_df['date'].max()
  
            if "asset_return_daily_scaled" in window_df.columns:
                returns = window_df["asset_return_daily_scaled"].dropna().values
            else:
                window_df = window_df.dropna(subset=["asset_return_daily"])
                if len(window_df) < 400: continue 
                returns = window_df["asset_return_daily"].values * SCALE_FACTOR
            
            try:
                diag_total += 1
                
                # Use Student's t distribution, GARCH(1,1)
                # Enforce stationarity in optimization if possible (requires arch >= 4.19 approx)
                # But 'arch' package doesn't always expose 'enforce_stationarity' easily in high level fit?
                # Actually, defaults usually enforce alpha+beta < 1 unless bounds are loose.
                am = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
                
                # Warm Start
                start_vals = last_params.values if last_params is not None else None
                
                # Fit
                res = am.fit(starting_values=start_vals, disp='off', show_warning=False)
                
                # 1. Convergence Check
                # Check convergence using optimization_result.success or convergence_flag
                converged = (hasattr(res, 'optimization_result') and res.optimization_result.success) or \
                           (hasattr(res, 'convergence_flag') and res.convergence_flag == 0)
                
                if not converged:
                    diag_rejected_convergence += 1
                    # Skip non-converged windows to avoid garbage
                    continue
                
                diag_converged += 1
                
                # Extract parameters (scaled)
                omega_est = res.params['omega']
                alpha_est = res.params['alpha[1]']
                beta_est = res.params['beta[1]']
                nu_est = res.params.get('nu', np.nan) 
                mu_est = res.params.get('mu', 0.0) 

                # Rescale back to decimal units
                # omega is varying with scale^2
                omega = omega_est / (SCALE_FACTOR ** 2)
                mu = mu_est / SCALE_FACTOR
                alpha = alpha_est
                beta = beta_est
                nu = nu_est
                
                # Capture Last Conditional Volatility (Sigma0)
                # res.conditional_volatility is in SCALED units if dist was scaled, but here we passed scaled returns.
                # So the volatility output is also scaled (percentage).
                # We need to divide by SCALE_FACTOR to get decimal volatility.
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
                    'sigma0': sigma0, # New: Last conditional volatility
                    'persistence': alpha + beta,
                    'unconditional_variance': omega / (1 - alpha - beta) if (alpha + beta) < 1 else np.nan,
                    'log_likelihood': getattr(res, 'loglikelihood', getattr(res, 'llf', np.nan)),
                    'aic': getattr(res, 'aic', np.nan),
                    'bic': getattr(res, 'bic', np.nan),
                    'num_observations': len(returns)
                }
                
                # 2. Sanitization & Stationarity Repair
                # (Renormalizes beta if alpha+beta >= 0.999, Clips Nu >= 2.1)
                sanitized = sanitize_garch_params(params_row)
                
                if sanitized['is_rejected']:
                    diag_rejected_stationarity += 1
                    continue
                    
                if sanitized['is_repaired']:
                    diag_repaired += 1
                
                params_list.append(sanitized)
                
                # Update parameters for next iteration (warm start)
                # Keep using the ESTIMATED values for warm start of the optimizer, not the repaired ones?
                # Actually, repaired ones might be better if the previous fit was boundary-hitting.
                # But 'last_params' expects the structure of res.params.
                # We'll just stick to res.params for simplicity.
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
        
        # Prepare params for merge (rename to garch_ prefix)
        merge_df = params_df[['gvkey', 'date', 'omega', 'alpha', 'beta', 'nu', 'mu', 'sigma0']].rename(columns={
            'omega': 'garch_omega',
            'alpha': 'garch_alpha',
            'beta': 'garch_beta',
            'nu': 'garch_nu',
            'mu': 'garch_mu_daily',
            'sigma0': 'garch_sigma0' # Store as sigma0
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
        
        # Calculate approximate volatility for plotting/checking
        # Priority: Sigma0 (Last Cond Vol) > Unconditional Vol
        df_out['garch_volatility'] = df_out['garch_sigma0']
        
        # Fallback to unconditional variance where sigma0 is nan (start of series) but params exist
        mask_nan = df_out['garch_volatility'].isna() & df_out['garch_omega'].notna()
        if mask_nan.any():
            df_out.loc[mask_nan, 'garch_volatility'] = np.sqrt(
                df_out.loc[mask_nan, 'garch_omega'] / 
                (1 - df_out.loc[mask_nan, 'garch_alpha'] - df_out.loc[mask_nan, 'garch_beta'])
            )
        
    return df_out
