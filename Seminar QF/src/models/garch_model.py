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
    df_out["garch_volatility"] = np.nan
    df_out["garch_omega"] = np.nan
    df_out["garch_alpha"] = np.nan
    df_out["garch_beta"] = np.nan
    df_out["garch_nu"] = np.nan  # Degrees of freedom for t-distribution
    
    firms = df_out["gvkey"].unique()
    print(f"Processing GARCH for {len(firms)} firms (Daily Data)...")
    
    SCALE_FACTOR = 100  # Scale returns to percentage form
    
    # Initialize list to store parameters for each firm
    params_list = []
    
    for i, gvkey in enumerate(firms):
        mask = df_out["gvkey"] == gvkey
        firm_ts = df_out.loc[mask].copy()
        
        # Ensure date column is datetime
        if 'date' not in firm_ts.columns:
             # Try index if it is datetime
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
        
        # Start 12 months in to ensure full window
        try:
            estimation_start = start_date + pd.DateOffset(months=24)
            if estimation_start >= end_date:
                continue
            month_ends = pd.date_range(start=estimation_start, end=end_date, freq='ME')
        except Exception as e:
            print(f"Error defining date range for {gvkey}: {e}")
            continue

            
        print(f"[{i+1}/{len(firms)}] Processing {gvkey} (Rolling 24M Window)...")

            
        print(f"[{i+1}/{len(firms)}] Processing {gvkey} (Rolling 24M Window)...")
        
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
            if window_df['asset_return_daily'].isna().sum() > 0: 
                 # Skip if too many missing returns in the 252-day window
                 print(f"Skipping {gvkey} on {date_point.date()}: Missing returns in window")
                 continue
                 
            last_trading_date = window_df['date'].max()
  
            if "asset_return_daily_scaled" in window_df.columns:
                returns = window_df["asset_return_daily_scaled"].dropna().values
            else:
                window_df = window_df.dropna(subset=["asset_return_daily"])
                # After dropping NaNs, ensure we still have enough data points for estimation
                if len(window_df) < 200: continue 
                # Scale returns to percentage locally if not already scaled
                returns = window_df["asset_return_daily"].values * SCALE_FACTOR
            
            try:
                # Use Student's t distribution to handle fat tails
                am = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
                
                # Use parameters from previous month as starting values (warm start)
                start_vals = last_params.values if last_params is not None else None
                res = am.fit(starting_values=start_vals, disp='off', show_warning=False)
                
                # Update parameters for next iteration (warm start)
                last_params = res.params
                
                # Extract parameters (scaled)
                omega = res.params['omega']
                alpha = res.params['alpha[1]']
                beta = res.params['beta[1]']
                nu = res.params.get('nu', np.nan)  # Degrees of freedom for t-distribution
                mu_scaled = res.params.get('mu', 0.0) # Mean/Drift (scaled % units)

                # Rescale omega back
                omega = omega / (SCALE_FACTOR ** 2)
                # Rescale mu back (percent -> decimal)
                mu = mu_scaled / SCALE_FACTOR
                
                # Store parameters for this window
                params_row = {
                    'gvkey': gvkey,
                    'date': last_trading_date,  # Use LAST TRADING DATE to ensure merge match
                    'omega': omega,
                    'alpha': alpha,
                    'beta': beta,
                    'nu': nu,
                    'mu': mu, # Decimal daily log return drift
                    'persistence': alpha + beta,
                    'unconditional_variance': omega / (1 - alpha - beta) if (alpha + beta) < 1 else np.nan,
                    'log_likelihood': getattr(res, 'loglikelihood', getattr(res, 'llf', np.nan)),
                    'aic': getattr(res, 'aic', np.nan),
                    'bic': getattr(res, 'bic', np.nan),
                    'num_observations': len(returns)
                }
                params_list.append(params_row)
                
            except Exception as e:
                print(f"Error estimating GARCH for {gvkey} on {date_point}: {e}")
                continue

    print("GARCH estimation complete.")
    
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
        merge_df = params_df[['gvkey', 'date', 'omega', 'alpha', 'beta', 'nu', 'mu']].rename(columns={
            'omega': 'garch_omega',
            'alpha': 'garch_alpha',
            'beta': 'garch_beta',
             'nu': 'garch_nu',
            'mu': 'garch_mu_daily'
        })
        
        # Drop existing columns to avoid conflicts
        drop_cols = ['garch_omega', 'garch_alpha', 'garch_beta', 'garch_nu', 'garch_mu_daily', 'garch_volatility']
        df_out = df_out.drop(columns=[c for c in drop_cols if c in df_out.columns])
        
        # Merge on date (month-ends only match initially)
        df_out = pd.merge(df_out, merge_df, on=['gvkey', 'date'], how='left')
        
        # Forward fill parameters per firm
        df_out = df_out.sort_values(['gvkey', 'date'])
        fill_cols = ['garch_omega', 'garch_alpha', 'garch_beta', 'garch_nu', 'garch_mu_daily']
        df_out[fill_cols] = df_out.groupby('gvkey')[fill_cols].ffill()
        
        # Calculate approximate volatility (unconditional) for plotting/checking
        # Valid only where params exist
        df_out['garch_volatility'] = np.sqrt(
            df_out['garch_omega'] / (1 - df_out['garch_alpha'] - df_out['garch_beta'])
        )
        
    return df_out
