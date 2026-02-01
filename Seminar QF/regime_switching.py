# regime_switching.py

import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

def run_regime_switching_estimation(daily_returns_df):
    """
    Estimates a 2-regime Hamilton filter on DAILY asset returns using MarkovRegression.
    
    Theory:
    -------
    Hidden Markov Model with 2 regimes (low volatility vs. high volatility).
    
    Parameters:
    -----------
    daily_returns_df : pd.DataFrame
        DataFrame with columns: 'gvkey', 'date', 'asset_return_daily'
    
    Returns:
    --------
    pd.DataFrame: Copy of input with regime state columns added
    """
    print("Estimating Regime Switching Model (2-Regime Markov) on DAILY Returns...")
    print("(Hamilton Filter)\n")
    
    if daily_returns_df.empty:
        print("No daily returns provided.")
        return daily_returns_df
    
    df_out = daily_returns_df.copy().reset_index(drop=True)
    df_out["regime_state"] = np.nan
    df_out["regime_probability_0"] = np.nan
    df_out["regime_probability_1"] = np.nan
    
    firms = df_out["gvkey"].unique()
    print(f"Processing Regime Switching for {len(firms)} firms...\n")
    
    # Initialize params list
    params_list = []
    
    for i, gvkey in enumerate(firms):
        # Get firm data using boolean mask
        firm_mask = df_out["gvkey"] == gvkey
        firm_df = df_out.loc[firm_mask].copy()
        
        # Sort by date
        firm_df = firm_df.sort_values("date")
        
        # Get the actual indices in df_out for this firm
        firm_indices = firm_df.index.tolist()
        
        # Get returns (drop NaNs but keep track of which indices are valid)
        valid_mask = firm_df["asset_return_daily"].notna()
        valid_indices = [firm_indices[j] for j in range(len(firm_indices)) if valid_mask.iloc[j]]
        returns = firm_df.loc[valid_mask, "asset_return_daily"].values
        
        if len(returns) < 150:
            print(f"  Firm {i+1}/{len(firms)}: gvkey={gvkey} - Insufficient data (n={len(returns)})")
            continue
        
        try:
            # Fit 2-regime Markov Regression
            mod = MarkovRegression(
                returns, 
                k_regimes=2, 
                trend='c'
            )
            res = mod.fit(disp=False, maxiter=200)
            
            # Get regime states using smoothed_marginal_probabilities
            smoothed_probs = res.smoothed_marginal_probabilities
            
            # Convert to numpy array
            if hasattr(smoothed_probs, 'values'):
                regime_probs = smoothed_probs.values
            else:
                regime_probs = np.array(smoothed_probs)
            
            regime_state = regime_probs.argmax(axis=1)
            
            # Assign back to df_out using valid_indices
            for j, idx in enumerate(valid_indices):
                df_out.loc[idx, "regime_state"] = regime_state[j]
                df_out.loc[idx, "regime_probability_0"] = regime_probs[j, 0]
                df_out.loc[idx, "regime_probability_1"] = regime_probs[j, 1]
            
            # Extract parameters
            params_values = res.params.values if hasattr(res.params, 'values') else np.array(res.params)
            
            # Get means (intercepts)
            regime_0_mean = float(params_values[0]) if len(params_values) > 0 else 0.0
            regime_1_mean = float(params_values[1]) if len(params_values) > 1 else 0.0
            
            # Get volatilities - calculate from data based on regime assignment
            regime_0_vol = float(np.std(returns[regime_state == 0])) if np.sum(regime_state == 0) > 0 else 0.2
            regime_1_vol = float(np.std(returns[regime_state == 1])) if np.sum(regime_state == 1) > 0 else 0.2
            
            # Compute transition matrix from data
            trans = np.zeros((2, 2))
            for t in range(len(regime_state) - 1):
                trans[int(regime_state[t]), int(regime_state[t + 1])] += 1
            row_sums = trans.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            trans = trans / row_sums
            
            # Store parameters
            params_row = {
                'gvkey': gvkey,
                'regime_0_mean': regime_0_mean,
                'regime_1_mean': regime_1_mean,
                'regime_0_ar': 0.0,
                'regime_1_ar': 0.0,
                'regime_0_vol': regime_0_vol,
                'regime_1_vol': regime_1_vol,
                'transition_prob_00': float(trans[0, 0]),
                'transition_prob_01': float(trans[0, 1]),
                'transition_prob_10': float(trans[1, 0]),
                'transition_prob_11': float(trans[1, 1]),
                'log_likelihood': float(res.llf) if hasattr(res, 'llf') else np.nan,
                'aic': float(res.aic) if hasattr(res, 'aic') else np.nan,
                'bic': float(res.bic) if hasattr(res, 'bic') else np.nan
            }
            params_list.append(params_row)
            
            print(f"  ✓ Firm {i+1}/{len(firms)}: gvkey={gvkey}")
            print(f"      Observations: {len(returns)}")
            print(f"      Regime 0: μ={regime_0_mean:.6f}, σ={regime_0_vol:.6f}")
            print(f"      Regime 1: μ={regime_1_mean:.6f}, σ={regime_1_vol:.6f}")
            print(f"      Transition: P(0→0)={trans[0, 0]:.3f}, P(1→1)={trans[1, 1]:.3f}\n")
            
        except Exception as e:
            print(f"  ✗ Firm {i+1}/{len(firms)}: gvkey={gvkey} - Error: {str(e)[:80]}")
            continue
    
    print("Regime Switching estimation complete.")
    
    # Save parameters for Monte Carlo
    if params_list:
        params_df = pd.DataFrame(params_list)
        params_df.to_csv('regime_switching_parameters.csv', index=False)
        print(f"\n✓ Saved regime-switching parameters to 'regime_switching_parameters.csv'")
        print(f"  Successfully estimated {len(params_list)} firms")
    
    return df_out
