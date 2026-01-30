# regime_switching.py

import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def run_regime_switching_estimation(daily_returns_df):
    """
    Estimates a 2-regime Hamilton filter on DAILY asset returns.
    
    Parameters:
    -----------
    daily_returns_df : pd.DataFrame
        DataFrame with columns: 'gvkey', 'date', 'asset_return_daily'
    
    Returns:
    --------
    pd.DataFrame: Copy of input with regime state columns added
    """
    print("Estimating Regime Switching Model (Hamilton Filter) on DAILY Returns...")
    
    if daily_returns_df.empty:
        print("No daily returns provided.")
        return daily_returns_df
    
    df_out = daily_returns_df.copy()
    df_out["regime_state"] = np.nan
    df_out["regime_probability_0"] = np.nan
    df_out["regime_probability_1"] = np.nan
    
    firms = df_out["gvkey"].unique()
    print(f"Processing Regime Switching for {len(firms)} firms...\n")
    
    for i, gvkey in enumerate(firms):
        mask = df_out["gvkey"] == gvkey
        firm_df = df_out.loc[mask].copy()
        
        # Ensure time sorted
        firm_df = firm_df.sort_values("date")
        
        # Get returns series (drop NaNs)
        returns = firm_df["asset_return_daily"].dropna()
        
        if len(returns) < 100:
            print(f"  Firm {i+1}/{len(firms)}: gvkey={gvkey} - Insufficient data (n={len(returns)})")
            continue
        
        try:
            # Fit 2-regime Markov Switching model
            mod = MarkovRegression(returns.values, k_regimes=2, trend='c')
            res = mod.fit(disp=False)
            
            # Extract regime states and probabilities
            regime_state = res.predicted_state
            regime_probs = res.predicted_state_prob
            
            # Map back to original indices
            firm_indices = firm_df[firm_df["asset_return_daily"].notna()].index
            
            df_out.loc[firm_indices, "regime_state"] = regime_state
            df_out.loc[firm_indices, "regime_probability_0"] = regime_probs[:, 0]
            df_out.loc[firm_indices, "regime_probability_1"] = regime_probs[:, 1]
            
            print(f"  ✓ Firm {i+1}/{len(firms)}: gvkey={gvkey} - {len(returns)} daily observations")
            
        except Exception as e:
            print(f"  ✗ Firm {i+1}/{len(firms)}: gvkey={gvkey} - Error: {e}")
            continue
    
    print("\nRegime Switching estimation complete.")
    return df_out
