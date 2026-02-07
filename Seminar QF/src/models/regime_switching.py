# regime_switching.py

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Import config for output paths
try:
    from src.utils import config
    OUTPUT_DIR = config.OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "output"

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
        # Use centrally scaled returns if available
        scaled_available = "asset_return_daily_scaled" in firm_df.columns
        target_col = "asset_return_daily_scaled" if scaled_available else "asset_return_daily"
        
        valid_mask = firm_df[target_col].notna()
        valid_indices = [firm_indices[j] for j in range(len(firm_indices)) if valid_mask.iloc[j]]
        returns = firm_df.loc[valid_mask, target_col].values
        
        # Scale if fallback to unscaled data was necessary
        SCALE_FACTOR = 100.0
        used_scaled = scaled_available
        if not used_scaled:
            returns = returns * SCALE_FACTOR
            used_scaled = True
        
        if len(returns) < 150:
            print(f"  Firm {i+1}/{len(firms)}: gvkey={gvkey} - Insufficient data (n={len(returns)})")
            continue
        
        try:
            # Fit 2-regime Markov Regression (on SCALED returns)
            mod = MarkovRegression(
                returns, 
                k_regimes=2, 
                trend='c',
                switching_variance=True
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

            # Create a dictionary of parameters for easy access by name
            param_names = res.model.param_names
            params_values = res.params
            if hasattr(params_values, 'values'):
                params_values = params_values.values
            
            params_dict = dict(zip(param_names, params_values))
            
            # Extract transition probabilities
            p_00 = float(params_dict.get('p[0->0]', np.nan))
            p_10 = float(params_dict.get('p[1->0]', np.nan))
            
            # Calculate complements
            p_01 = 1.0 - p_00 if not np.isnan(p_00) else np.nan
            p_11 = 1.0 - p_10 if not np.isnan(p_10) else np.nan

            # Extract means (const) and Unscale
            regime_0_mean = float(params_dict.get('const[0]', 0.0)) / SCALE_FACTOR
            regime_1_mean = float(params_dict.get('const[1]', 0.0)) / SCALE_FACTOR

            # Extract volatilities (sigma2 -> sqrt) and Unscale
            # param is sigma2 of scaled returns. 
            # Real sigma2 = param / (SCALE^2). Real sigma = sqrt(param) / SCALE.
            sigma2_0_scaled = float(params_dict.get('sigma2[0]', 0.04 * (SCALE_FACTOR**2)))
            sigma2_1_scaled = float(params_dict.get('sigma2[1]', 0.04 * (SCALE_FACTOR**2)))
            
            regime_0_vol = (np.sqrt(sigma2_0_scaled) / SCALE_FACTOR) if sigma2_0_scaled > 0 else 0.2
            regime_1_vol = (np.sqrt(sigma2_1_scaled) / SCALE_FACTOR) if sigma2_1_scaled > 0 else 0.2
            
            # CORRECT REGIME LABELING: Ensure regime 0 = HIGH vol, regime 1 = LOW vol
            if regime_0_vol < regime_1_vol:
                # Swap regime states in output df
                for j, idx in enumerate(valid_indices):
                     df_out.loc[idx, "regime_state"] = 1 - df_out.loc[idx, "regime_state"]
                     # Swap probabilities
                     p0_val = df_out.loc[idx, "regime_probability_0"]
                     df_out.loc[idx, "regime_probability_0"] = df_out.loc[idx, "regime_probability_1"]
                     df_out.loc[idx, "regime_probability_1"] = p0_val

                # Swap parameters
                regime_0_vol, regime_1_vol = regime_1_vol, regime_0_vol
                regime_0_mean, regime_1_mean = regime_1_mean, regime_0_mean
                
                # Swap transitions:
                # New Regime 0 is old Regime 1. New Regime 1 is old Regime 0.
                p_00, p_11 = p_11, p_00
                p_01, p_10 = p_10, p_01
            
            # Create trans matrix for printing
            trans = np.array([[p_00, p_01], [p_10, p_11]])
            
            # Store parameters
            params_row = {
                'gvkey': gvkey,
                'regime_0_mean': regime_0_mean,
                'regime_1_mean': regime_1_mean,
                'regime_0_ar': 0.0,
                'regime_1_ar': 0.0,
                'regime_0_vol': regime_0_vol,
                'regime_1_vol': regime_1_vol,
                'transition_prob_00': p_00,
                'transition_prob_01': p_01,
                'transition_prob_10': p_10,
                'transition_prob_11': p_11,
                'log_likelihood': float(res.llf) if hasattr(res, 'llf') else np.nan,
                'aic': float(res.aic) if hasattr(res, 'aic') else np.nan,
                'bic': float(res.bic) if hasattr(res, 'bic') else np.nan
            }
            params_list.append(params_row)
            
            print(f"  ✓ Firm {i+1}/{len(firms)}: gvkey={gvkey}")
            print(f"      Observations: {len(returns)}")
            print(f"      Regime 0 (HIGH vol/stress): μ={regime_0_mean:.6f}, σ={regime_0_vol:.6f}")
            print(f"      Regime 1 (LOW vol/calm): μ={regime_1_mean:.6f}, σ={regime_1_vol:.6f}")
            print(f"      Transition: P(0→0)={trans[0, 0]:.3f}, P(1→1)={trans[1, 1]:.3f}\n")
            
        except Exception as e:
            print(f"  ✗ Firm {i+1}/{len(firms)}: gvkey={gvkey} - Error: {str(e)[:80]}")
            continue
    
    print("Regime Switching estimation complete.")
    
    # Save parameters for Monte Carlo
    if params_list:
        params_df = pd.DataFrame(params_list)
        output_path = OUTPUT_DIR / 'regime_switching_parameters.csv'
        params_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved regime-switching parameters to '{output_path}'")
        print(f"  Successfully estimated {len(params_list)} firms")
    
    return df_out
