# ms_garch.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

class MSGARCH_Model:
    """
    Implements a 2-Regime Markov-Switching GARCH(1,1) model.
    Specification: Haas, Mittnik, and Paolella (2004) - Diagonal process.
    The variance processes are decoupled.
    """
    def __init__(self, returns):
        # Returns should be 1D array of percent returns (e.g. 1.5 for 1.5%)
        self.y = returns
        self.T = len(returns)
        
    def diff_log_likelihood(self, params):
        # Unpack parameters
        # p00, p11: transition probs
        # regime 0: omega0, alpha0, beta0
        # regime 1: omega1, alpha1, beta1
        
        p00 = params[0]
        p11 = params[1]
        
        omega0, alpha0, beta0 = params[2], params[3], params[4]
        omega1, alpha1, beta1 = params[5], params[6], params[7]
        
        # Constraints check (return high value if violated)
        if any(x < 1e-6 for x in [omega0, alpha0, beta0, omega1, alpha1, beta1]): return 1e10
        if any(x > 1 - 1e-6 for x in [p00, p11]): return 1e10
        if (alpha0 + beta0 >= 1.0) or (alpha1 + beta1 >= 1.0): return 1e10

        # Transition Matrix
        # P[j, i] = Prob(S_t = i | S_t-1 = j)
        P = np.array([[p00, 1 - p00],
                      [1 - p11, p11]])
        
        # Steady state probabilities (Unconditional P(S_t))
        # Pi = P.T * Pi
        # Solver for 2x2: pi0 = (1-p11) / (2 - p00 - p11)
        denom = (2 - p00 - p11)
        if denom == 0: return 1e10
        pi0 = (1 - p11) / denom
        pi1 = 1 - pi0
        
        xi_pred = np.array([pi0, pi1]) # xi_{t|t-1}
        
        # Initial Variances (Unconditional variance of each GARCH)
        h0 = omega0 / (1 - alpha0 - beta0)
        h1 = omega1 / (1 - alpha1 - beta1)
        h = np.array([h0, h1])
        
        log_lik = 0.0
        
        # Loop mainly for log-likelihood
        # Optimized implementation would do this in Cython/Numba, but Python is ok for monthly data
        
        for t in range(self.T):
            y_t = self.y[t]
            y_sq = y_t**2
            
            # 1. Densities f(y_t | S_t=k, I_{t-1})
            # Assume zero mean or y is already demeaned
            # Normal distribution N(0, h_k)
            
            densities = np.empty(2)
            densities[0] = (1.0 / np.sqrt(2 * np.pi * h[0])) * np.exp(-0.5 * y_sq / h[0])
            densities[1] = (1.0 / np.sqrt(2 * np.pi * h[1])) * np.exp(-0.5 * y_sq / h[1])
            
            # Avoid numerical issues
            densities = np.maximum(densities, 1e-20)
            
            # 2. Likelihood of observation t: sum_k xi_{k|t-1} * density_k
            lik_t = np.sum(xi_pred * densities)
            log_lik += np.log(lik_t)
            
            # 3. Filtered probabilities xi_{t|t}
            xi_filt = (xi_pred * densities) / lik_t
            
            # 4. Predicted probabilities for next step xi_{t+1|t} = xi_{t|t} * P
            # Note: xi vectors are row vs column conventions.
            # Here xi_pred[i] is Prob(S=i).
            # xi_{t+1|t}[j] = sum_i xi_{t|t}[i] * P[i, j]
            xi_pred = np.dot(xi_filt, P)
            
            # 5. Update Variances for next step (Haas et al decoupled)
            # h_{k, t+1}
            h[0] = omega0 + alpha0 * y_sq + beta0 * h[0]
            h[1] = omega1 + alpha1 * y_sq + beta1 * h[1]
            
        return -log_lik # Minimize negative log likelihood

    def fit(self):
        # Initial Guess
        # p00=0.9, p11=0.7 (persistent regimes)
        # GARCH parameters: mild persistence
        # omega ~ VAR * (1-a-b)
        
        global_var = np.var(self.y)
        if global_var == 0: global_var = 1.0
        
        guess = [0.8, 0.8, 
                 0.1 * global_var, 0.1, 0.8, # Low Vol Regime
                 0.5 * global_var, 0.2, 0.6] # High Vol Regime (higher alpha, lower beta?)
        
        # Bounds
        bounds = [(0.01, 0.99), (0.01, 0.99),
                  (1e-6, None), (0.01, 0.99), (0.01, 0.99),
                  (1e-6, None), (0.01, 0.99), (0.01, 0.99)]
        
        res = minimize(self.diff_log_likelihood, guess, method='SLSQP', bounds=bounds, tol=1e-4)
        
        return res

    def get_conditional_stats(self, params):
        # Re-run filter to get probabilities and volatilities
        p00, p11 = params[0], params[1]
        omega0, alpha0, beta0 = params[2], params[3], params[4]
        omega1, alpha1, beta1 = params[5], params[6], params[7]
        
        P = np.array([[p00, 1 - p00], [1 - p11, p11]])
        denom = (2 - p00 - p11)
        pi0 = (1 - p11) / denom if denom != 0 else 0.5
        xi_pred = np.array([pi0, 1-pi0])
        
        h = np.array([omega0 / (1 - alpha0 - beta0), omega1 / (1 - alpha1 - beta1)])
        
        probs_0 = []
        vol_0 = []
        vol_1 = []
        cond_vol = [] # Weighted average volatility
        
        for t in range(self.T):
            y_t = self.y[t]
            y_sq = y_t**2
            
            # Store values
            probs_0.append(xi_pred[0]) # Store PREDICTED or Filtered? Ideally Filtered for analysis.
            # Let's calculate filtered
            
            densities = np.empty(2)
            densities[0] = (1.0 / np.sqrt(2 * np.pi * h[0])) * np.exp(-0.5 * y_sq / h[0])
            densities[1] = (1.0 / np.sqrt(2 * np.pi * h[1])) * np.exp(-0.5 * y_sq / h[1])
            densities = np.maximum(densities, 1e-20)
            lik_t = np.sum(xi_pred * densities)
            xi_filt = (xi_pred * densities) / lik_t
            
            # Update storage to be filtered probabilities
            probs_0[-1] = xi_filt[0]
            
            vol_0.append(np.sqrt(h[0]))
            vol_1.append(np.sqrt(h[1]))
            # Overall conditional volatility: E[sigma_t^2] = sum p_k * sigma_k^2 + sum p_k * mu_k^2 - (sum p_k mu_k)^2
            # Assuming zero mean, it is sqrt( sum p_k * sigma_k^2 )
            curr_cond_var = xi_filt[0]*h[0] + xi_filt[1]*h[1]
            cond_vol.append(np.sqrt(curr_cond_var))
            
            # Next step
            xi_pred = np.dot(xi_filt, P)
            h[0] = omega0 + alpha0 * y_sq + beta0 * h[0]
            h[1] = omega1 + alpha1 * y_sq + beta1 * h[1]
            
        return probs_0, vol_0, vol_1, cond_vol


def run_ms_garch_estimation(monthly_returns_df):
    """
    Estimates a 2-regime MS-GARCH(1,1) (Haas et al. 2004) on monthly asset returns.
    """
    print("Estimating MS-GARCH(1,1) Model...")
    
    if monthly_returns_df.empty:
        print("No monthly returns provided for MS-GARCH.")
        return monthly_returns_df
        
    df_out = monthly_returns_df.copy()
    
    # Initialize columns
    df_out["msgarch_prob_0"] = np.nan
    df_out["msgarch_vol_0"] = np.nan
    df_out["msgarch_vol_1"] = np.nan
    df_out["msgarch_cond_vol"] = np.nan
    
    firms = df_out["gvkey"].unique()
    print(f"Processing MS-GARCH for {len(firms)} firms...")
    
    for i, gvkey in enumerate(firms):
        mask = df_out["gvkey"] == gvkey
        firm_df = df_out.loc[mask]
        firm_df = firm_df.sort_values("month_year")
        
        returns = firm_df["asset_return_monthly"].dropna()
        
        # Need sufficient data (more than GARCH)
        if len(returns) < 50: 
            continue
            
        # Demean returns and scale
        y = (returns - returns.mean()) * 100
        y_vals = y.values
        
        model = MSGARCH_Model(y_vals)
        try:
            res = model.fit()
            if res.success:
                probs_0, vol_0, vol_1, cond_vol = model.get_conditional_stats(res.x)
                
                # Assign back (rescale vols back to decimal)
                # Vol is for percentage returns, so divide by 100
                # Annualize? User usually wants annualized. 
                # Monthly vol -> * sqrt(12)
                
                indices = returns.index
                df_out.loc[indices, "msgarch_prob_0"] = probs_0
                df_out.loc[indices, "msgarch_vol_0"] = (np.array(vol_0) / 100) * np.sqrt(12)
                df_out.loc[indices, "msgarch_vol_1"] = (np.array(vol_1) / 100) * np.sqrt(12)
                df_out.loc[indices, "msgarch_cond_vol"] = (np.array(cond_vol) / 100) * np.sqrt(12)
        except Exception as e:
            # print(f"MS-GARCH failed for {gvkey}: {e}")
            continue
            
        if (i + 1) % 10 == 0:
            print(f"Processed MS-GARCH for {i+1} firms...")
            
    print("MS-GARCH estimation complete.")
    return df_out
