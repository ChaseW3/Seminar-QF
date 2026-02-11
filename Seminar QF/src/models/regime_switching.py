# regime_switching.py

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from numba import njit
import sys

# Import config for output paths
try:
    from src.utils import config
    OUTPUT_DIR = config.OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "output"

# =============================================================================
# NUMBA JIT-COMPILED HELPER FUNCTIONS
# =============================================================================

@njit(cache=True)
def _numba_gammaln(x):
    """Numba-compatible log-gamma function."""
    g = 7
    c = np.array([
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7
    ])
    if x < 0.5: return np.log(np.pi / np.sin(np.pi * x)) - _numba_gammaln(1 - x)
    x = x - 1
    a = c[0]
    for i in range(1, g + 2): a += c[i] / (x + i)
    t = x + g + 0.5
    return 0.5 * np.log(2 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)

@njit(cache=True)
def _t_log_likelihood(x, nu, sigma2):
    """Compute log-likelihood of t-distribution (Numba JIT compiled)."""
    if sigma2 <= 0 or nu <= 2: return -1e10
    const = _numba_gammaln((nu + 1) / 2) - _numba_gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi * sigma2)
    kernel = -((nu + 1) / 2) * np.log(1 + x**2 / ((nu - 2) * sigma2))
    return const + kernel

@njit(cache=True)
def hamilton_filter_t_jit(returns, mu_0, mu_1, sigma2_0, sigma2_1, p00, p11, nu_0, nu_1):
    """
    Hamilton filter for Regime Switching with t-distribution and CONSTANT volatility per regime.
    """
    T = len(returns)
    filtered_prob = np.zeros((T, 2))
    log_likelihood = 0.0
    
    P = np.array([[p00, 1 - p00], [1 - p11, p11]])
    
    denom = (2 - p00 - p11)
    if abs(denom) < 1e-10:
        pi_stat = np.array([0.5, 0.5]) 
    else:
        pi_stat = np.array([(1 - p11) / denom, (1 - p00) / denom])
        
    prev_filtered = pi_stat.copy()
    
    for t in range(T):
        r = returns[t]
        
        eps_0 = r - mu_0
        eps_1 = r - mu_1
        
        ll_0 = _t_log_likelihood(eps_0, nu_0, sigma2_0)
        ll_1 = _t_log_likelihood(eps_1, nu_1, sigma2_1)
        
        max_ll = max(ll_0, ll_1)
        if max_ll < -500:
            lik_0 = 1e-200
            lik_1 = 1e-200
        else:
            lik_0 = np.exp(ll_0 - max_ll)
            lik_1 = np.exp(ll_1 - max_ll)
            
        pred_prob = P.T @ prev_filtered
        joint_0 = lik_0 * pred_prob[0]
        joint_1 = lik_1 * pred_prob[1]
        
        marginal = joint_0 + joint_1
        
        if marginal < 1e-300:
            marginal = 1e-300
            
        filtered_prob[t, 0] = joint_0 / marginal
        filtered_prob[t, 1] = joint_1 / marginal
        
        log_likelihood += np.log(marginal) + max_ll
        
        prev_filtered = filtered_prob[t, :]
        
    return log_likelihood, filtered_prob

# =============================================================================
# ESTIMATION CLASS
# =============================================================================

class MarkovSwitchingTDist:
    """Custom Estimator for Markov Switching Model with t-distribution."""
    
    def __init__(self):
        self.params = {}
        self.probs = None
        
    def fit(self, returns):
        var = np.var(returns)
        mean = np.mean(returns)
        
        # [mu0, mu1, log(sigma2_0), log(sigma2_1), logit(p00), logit(p11), log(nu0-2), log(nu1-2)]
        x0 = np.array([
            mean, mean, 
            np.log(var * 0.5), np.log(var * 2.0),
            2.0, 2.0,
            np.log(8.0), np.log(4.0)
        ])
        
        returns_arr = np.ascontiguousarray(returns)
        
        def neg_log_lik(params):
            mu_0, mu_1 = params[0], params[1]
            sigma2_0, sigma2_1 = np.exp(params[2]), np.exp(params[3])
            p00, p11 = expit(params[4]), expit(params[5])
            nu_0 = 2 + np.exp(params[6])
            nu_1 = 2 + np.exp(params[7])
            
            if p00 < 0.01 or p11 < 0.01: return 1e10
            if p00 > 0.999 or p11 > 0.999: return 1e10
            
            ll, _ = hamilton_filter_t_jit(returns_arr, mu_0, mu_1, sigma2_0, sigma2_1, p00, p11, nu_0, nu_1)
            return -ll if np.isfinite(ll) else 1e10

        res = minimize(neg_log_lik, x0, method='L-BFGS-B', options={'disp': False, 'maxiter': 500})
        
        p = res.x
        self.params = {
            'mu_0': p[0], 'mu_1': p[1],
            'sigma2_0': np.exp(p[2]), 'sigma2_1': np.exp(p[3]),
            'p00': expit(p[4]), 'p11': expit(p[5]),
            'nu_0': 2 + np.exp(p[6]), 'nu_1': 2 + np.exp(p[7]),
            'log_likelihood': -res.fun
        }
        
        _, probs = hamilton_filter_t_jit(
            returns_arr, 
            self.params['mu_0'], self.params['mu_1'], 
            self.params['sigma2_0'], self.params['sigma2_1'],
            self.params['p00'], self.params['p11'],
            self.params['nu_0'], self.params['nu_1']
        )
        self.probs = probs
        return self.params, self.probs

def run_regime_switching_estimation(daily_returns_df):
    """
    Estimates a 2-regime Hamilton filter on DAILY asset returns using CUSTOM T-DISTRIBUTION ESTIMATOR.
    """
    print("Estimating Regime Switching Model (2-Regime Markov T-Dist) on DAILY Returns...")
    
    if daily_returns_df.empty:
        print("No daily returns provided.")
        return daily_returns_df
    
    df_out = daily_returns_df.copy().reset_index(drop=True)
    df_out["regime_state"] = np.nan
    df_out["regime_probability_0"] = np.nan
    df_out["regime_probability_1"] = np.nan
    
    firms = df_out["gvkey"].unique()
    print(f"Processing Regime Switching for {len(firms)} firms...\\n")
    
    params_list = []
    
    for i, gvkey in enumerate(firms):
        firm_mask = df_out["gvkey"] == gvkey
        firm_df = df_out.loc[firm_mask].sort_values("date")
        
        scaled_available = "asset_return_daily_scaled" in firm_df.columns
        target_col = "asset_return_daily_scaled" if scaled_available else "asset_return_daily"
        
        valid_mask = firm_df[target_col].notna()
        returns = firm_df.loc[valid_mask, target_col].values
        valid_indices = firm_df.loc[valid_mask].index
        
        # Scale factor for converting parameters back to decimal scale
        # If using pre-scaled data (×100), need to divide parameters by 100
        # If using raw data, scale it by 100 for estimation, then divide by 100
        calc_scale_factor = 100.0
        if not scaled_available:
             returns = returns * calc_scale_factor
        
        # WINSORIZATION: Clip extreme returns
        # Threshold: 30% daily return (returns are scaled by 100)
        returns = np.clip(returns, -30.0, 30.0)
        
        if len(returns) < 150:
            print(f"Skipping {gvkey}: insufficient data ({len(returns)})")
            continue
            
        try:
            model = MarkovSwitchingTDist()
            params, probs = model.fit(returns)
            
            # Map regimes: 0 = LOW, 1 = HIGH Volatility
            p_sigma0 = params['sigma2_0']
            p_sigma1 = params['sigma2_1']
            
            if p_sigma0 > p_sigma1:
                 params['sigma2_0'], params['sigma2_1'] = params['sigma2_1'], params['sigma2_0']
                 params['mu_0'], params['mu_1'] = params['mu_1'], params['mu_0']
                 params['nu_0'], params['nu_1'] = params['nu_1'], params['nu_0']
                 params['p00'], params['p11'] = params['p11'], params['p00']
                 probs[:, [0, 1]] = probs[:, [1, 0]]
            
            count = 0
            for idx in valid_indices:
                df_out.loc[idx, "regime_probability_0"] = probs[count, 0]
                df_out.loc[idx, "regime_probability_1"] = probs[count, 1]
                df_out.loc[idx, "regime_state"] = 0 if probs[count, 0] > 0.5 else 1
                count += 1
            
            params_list.append({
                'gvkey': gvkey,
                'regime_0_mean': params['mu_0'] / calc_scale_factor,
                'regime_1_mean': params['mu_1'] / calc_scale_factor,
                'regime_0_vol': np.sqrt(params['sigma2_0']) / calc_scale_factor,
                'regime_1_vol': np.sqrt(params['sigma2_1']) / calc_scale_factor,
                'regime_0_nu': params['nu_0'],
                'regime_1_nu': params['nu_1'], 
                'transition_prob_00': params['p00'],
                'transition_prob_01': 1 - params['p00'],
                'transition_prob_10': 1 - params['p11'],
                'transition_prob_11': params['p11'],
                'regime_0_ar': 0.0,
                'regime_1_ar': 0.0
            })
            
            print(f"  Processed {gvkey} (Reg0 Vol: {np.sqrt(params['sigma2_0'])/calc_scale_factor:.4f}, Nu: {params['nu_0']:.2f})")
            
        except Exception as e:
            print(f"Error {gvkey}: {e}")
            continue

    if params_list:
        pd.DataFrame(params_list).to_csv(OUTPUT_DIR / "regime_switching_parameters.csv", index=False)
        print("Saved regime_switching_parameters.csv")
    
    return df_out
