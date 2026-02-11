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
        firm_ts = df_out.loc[mask]
        
        # Check for sufficient data
        # Prefer scaled returns if available
        if "asset_return_daily_scaled" in firm_ts.columns:
            firm_ts = firm_ts.dropna(subset=["asset_return_daily_scaled"])
            if len(firm_ts) < 50:
                continue
            returns = firm_ts["asset_return_daily_scaled"].values
        else:
            firm_ts = firm_ts.dropna(subset=["asset_return_daily"])
            if len(firm_ts) < 50:
                continue
            # Scale returns to percentage locally if not already scaled
            returns = firm_ts["asset_return_daily"].values * SCALE_FACTOR
        
        # WINSORIZATION: Clip extreme returns to prevent optimization failure
        # Threshold: 30% daily return (approx 30 sigma event)
        # Prevents single outliers from exploding volatility persistence
        returns = np.clip(returns, -30.0, 30.0)
        
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
            
            # Store parameters for this firm
            params_row = {
                'gvkey': gvkey,
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'nu': nu,
                'persistence': alpha + beta,
                'unconditional_variance': omega / (1 - alpha - beta) if (alpha + beta) < 1 else np.nan,
                'log_likelihood': float(res.llf) if hasattr(res, 'llf') else np.nan,
                'aic': float(res.aic) if hasattr(res, 'aic') else np.nan,
                'bic': float(res.bic) if hasattr(res, 'bic') else np.nan,
                'num_observations': len(returns)
            }
            params_list.append(params_row)
            
        except Exception as e:
            continue
            
        if (i + 1) % 10 == 0:
            print(f"Processed GARCH for {i+1} firms...")

    print("GARCH estimation complete.")
    
    # Save parameters for Monte Carlo and reference
    if params_list:
        params_df = pd.DataFrame(params_list)
        output_path = OUTPUT_DIR / 'garch_parameters.csv'
        params_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved GARCH parameters to '{output_path}'")
        print(f"  Successfully estimated {len(params_list)} firms")
        print(f"  Mean persistence (α+β): {params_df['persistence'].mean():.4f}")
        print(f"  Mean degrees of freedom (ν): {params_df['nu'].mean():.2f}")
    
    return df_out
