# regime_switching.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings

def run_regime_switching_estimation(monthly_returns_df):
    """
    Estimates a 2-regime Markov Switching Model (Hamilton Filter) on monthly asset returns.
    Model: switching mean and switching variance.
    
    Args:
        monthly_returns_df (pd.DataFrame): DataFrame containing 'gvkey', 'date', 'asset_return_monthly'.
        
    Returns:
        pd.DataFrame: Copy of input DF with additional columns for regime probabilities and parameters.
    """
    print("Estimating Regime Switching Model (Hamilton Filter)...")
    
    if monthly_returns_df.empty:
        print("No monthly returns provided for Regime Switching.")
        return monthly_returns_df
        
    df_out = monthly_returns_df.copy()
    
    # Initialize columns
    df_out["regime_0_prob"] = np.nan
    df_out["regime_1_prob"] = np.nan
    df_out["sigma2_0"] = np.nan # Variance of regime 0
    df_out["sigma2_1"] = np.nan # Variance of regime 1
    
    firms = df_out["gvkey"].unique()
    print(f"Processing Regime Switching for {len(firms)} firms...")
    
    for i, gvkey in enumerate(firms):
        mask = df_out["gvkey"] == gvkey
        firm_df = df_out.loc[mask]
        
        # Ensure time sorted
        firm_df = firm_df.sort_values("month_year")
        
        # Get returns series (drop NaNs)
        returns = firm_df["asset_return_monthly"].dropna()
        
        # Need sufficient data
        if len(returns) < 40: 
            continue
            
        # Scale returns for better numerical stability (percentage returns)
        y = returns * 100
        
        try:
            # Fit Markov Switching Model
            # 2 regimes, switching trend (mean), switching variance
            # This corresponds to the standard Hamilton model for financial returns
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model = MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True)
                res = model.fit(disp=False, search_reps=5)
            
            # Extract smoothed probabilities (P(S_t = j | all data))
            # Or filtered (P(S_t = j | data up to t)) - "Hamilton Filter" specific
            # We will use smoothed probabilities as they are the standard "inference" result, 
            # but the results object contains both. 'smoothed_marginal_probabilities'
            
            probs = res.smoothed_marginal_probabilities
            
            # Get estimated variances to identify regimes
            # params contains: const[0], const[1], sigma2[0], sigma2[1], p[0->0], p[1->0]
            # The order depends on statsmodels version but usually params are named.
            
            sigma2_0 = res.params['sigma2[0]'] / 10000 # rescale back (since we scaled *100, var is *10000)
            sigma2_1 = res.params['sigma2[1]'] / 10000
            
            # Align indices (probs is aligned with y's index)
            df_out.loc[y.index, "regime_0_prob"] = probs[0]
            df_out.loc[y.index, "regime_1_prob"] = probs[1]
            df_out.loc[y.index, "sigma2_0"] = sigma2_0
            df_out.loc[y.index, "sigma2_1"] = sigma2_1
            
        except Exception as e:
            # print(f"Error for firm {gvkey}: {e}")
            continue
            
        if (i + 1) % 10 == 0:
            print(f"Processed RS for {i+1} firms...")
            
    print("Regime Switching estimation complete.")
    return df_out
