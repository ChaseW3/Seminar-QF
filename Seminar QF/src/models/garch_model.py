# garch_model.py

import pandas as pd
import numpy as np
from arch import arch_model

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
    
    for i, gvkey in enumerate(firms):
        mask = df_out["gvkey"] == gvkey
        firm_ts = df_out.loc[mask].dropna(subset=["asset_return_daily"])
        
        if len(firm_ts) < 50:
            continue
            
        # Scale returns to percentage
        returns = firm_ts["asset_return_daily"].values * SCALE_FACTOR
        
        try:
            # Use Student's t distribution to handle fat tails
            am = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            
            # Extract parameters (scaled)
            omega = res.params['omega']
            alpha = res.params['alpha[1]']
            beta = res.params['beta[1]']
            nu = res.params.get('nu', np.nan)  # Degrees of freedom for t-distribution
            
            # Get conditional volatility (scaled)
            cond_vol_scaled = np.asarray(res.conditional_volatility)
            
            # Rescale volatility back to original scale
            cond_vol = cond_vol_scaled / SCALE_FACTOR
            
            # Rescale omega back
            omega = omega / (SCALE_FACTOR ** 2)
            
            if np.any(np.isnan(cond_vol)) or np.any(cond_vol < 0):
                continue
            
            df_out.loc[firm_ts.index, "garch_volatility"] = cond_vol
            df_out.loc[firm_ts.index, "garch_omega"] = omega
            df_out.loc[firm_ts.index, "garch_alpha"] = alpha
            df_out.loc[firm_ts.index, "garch_beta"] = beta
            df_out.loc[firm_ts.index, "garch_nu"] = nu  # Save degrees of freedom
            
        except Exception as e:
            continue
            
        if (i + 1) % 10 == 0:
            print(f"Processed GARCH for {i+1} firms...")

    print("GARCH estimation complete.")
    return df_out
