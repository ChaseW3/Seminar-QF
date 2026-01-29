# garch_model.py

import pandas as pd
import numpy as np
from arch import arch_model

def run_garch_estimation(monthly_returns_df):
    """
    Estimates GARCH(1,1) on the provided monthly asset returns dataframe.
    Returns:
        pd.DataFrame: A copy of the input dataframe with a new 'garch_volatility' column.
    """
    print("Estimating GARCH(1,1) on COMPUTED MONTHLY Asset Returns...")
    
    if monthly_returns_df.empty:
        print("No monthly returns provided for GARCH.")
        return monthly_returns_df
        
    df_out = monthly_returns_df.copy()
    df_out["garch_volatility"] = np.nan
    
    firms = df_out["gvkey"].unique()
    print(f"Processing GARCH for {len(firms)} firms (Monthly Data)...")
    
    for i, gvkey in enumerate(firms):
        mask = df_out["gvkey"] == gvkey
        firm_ts = df_out.loc[mask].dropna(subset=["asset_return_monthly"])
        
        # GARCH usually needs > 50 points roughly
        if len(firm_ts) < 50:
            continue
            
        # Scale returns
        returns = firm_ts["asset_return_monthly"] * 100
        
        try:
            am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
            res = am.fit(disp='off', show_warning=False)
            print(res.optimization_result)
            
            cond_vol = res.conditional_volatility
            
            # Re-scale: This is monthly volatility
            monthly_garch_vol = (cond_vol / 100)
            
            # Annualize: * sqrt(12)
            annualized_garch_vol = monthly_garch_vol * np.sqrt(12)
            
            df_out.loc[firm_ts.index, "garch_volatility"] = annualized_garch_vol
            
        except Exception as e:
            # print(f"GARCH failed for {gvkey}: {e}")
            continue
            
        if (i + 1) % 10 == 0:
            print(f"Processed GARCH for {i+1} firms...")

    print("GARCH estimation complete.")
    return df_out
