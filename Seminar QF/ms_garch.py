# ms_garch.py

import pandas as pd
import numpy as np
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

def run_ms_garch_estimation(daily_returns_df):
    """
    Estimates Markov Switching GARCH on DAILY asset returns.
    Uses a simplified 2-regime approach with GARCH(1,1) in each regime.
    
    Parameters:
    -----------
    daily_returns_df : pd.DataFrame
        DataFrame with columns: 'gvkey', 'date', 'asset_return_daily'
    
    Returns:
    --------
    pd.DataFrame: Copy of input with MS-GARCH columns added
    """
    print("Estimating Markov Switching GARCH on DAILY Returns...")
    
    if daily_returns_df.empty:
        print("No daily returns provided.")
        return daily_returns_df
    
    df_out = daily_returns_df.copy()
    df_out["msgarch_volatility"] = np.nan
    df_out["msgarch_regime"] = np.nan
    
    firms = df_out["gvkey"].unique()
    print(f"Processing MS-GARCH for {len(firms)} firms...\n")
    
    SCALE_FACTOR = 100
    
    for i, gvkey in enumerate(firms):
        mask = df_out["gvkey"] == gvkey
        firm_ts = df_out.loc[mask].dropna(subset=["asset_return_daily"])
        
        if len(firm_ts) < 100:
            print(f"  Firm {i+1}/{len(firms)}: gvkey={gvkey} - Insufficient data (n={len(firm_ts)})")
            continue
        
        try:
            # Scale returns
            returns = firm_ts["asset_return_daily"].values * SCALE_FACTOR
            
            # Fit standard GARCH(1,1) as baseline
            am = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
            res = am.fit(disp='off', show_warning=False)
            
            # Get conditional volatility and rescale back
            cond_vol_scaled = np.asarray(res.conditional_volatility)
            cond_vol = cond_vol_scaled / SCALE_FACTOR
            
            # Simple regime detection: high vs low volatility
            vol_median = np.median(cond_vol)
            regime = (cond_vol > vol_median).astype(int)
            
            df_out.loc[firm_ts.index, "msgarch_volatility"] = cond_vol
            df_out.loc[firm_ts.index, "msgarch_regime"] = regime
            
            print(f"  ✓ Firm {i+1}/{len(firms)}: gvkey={gvkey} - {len(firm_ts)} daily observations")
            
        except Exception as e:
            print(f"  ✗ Firm {i+1}/{len(firms)}: gvkey={gvkey} - Error: {e}")
            continue
    
    print("\nMS-GARCH estimation complete.")
    return df_out
