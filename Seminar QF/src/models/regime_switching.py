# regime_switching.py

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import expit
from numba import njit
import sys
# Parallel imports removed for performance optimization (overhead > benefit for this model)
# from joblib import Parallel, delayed
# import multiprocessing

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
    if sigma2 <= 0 or nu <= 2.001: return -1e10 # Strict check
    const = _numba_gammaln((nu + 1) / 2) - _numba_gammaln(nu / 2) - 0.5 * np.log((nu - 2) * np.pi * sigma2)
    kernel = -((nu + 1) / 2) * np.log(1 + x**2 / ((nu - 2) * sigma2))
    return const + kernel

@njit(cache=True)
def hamilton_filter_t_details_jit(returns, mu_0, mu_1, sigma2_0, sigma2_1, p00, p11, nu_0, nu_1):
    """
    Hamilton filter for Regime Switching with t-distribution.
    Returns likelihood, filtered probabilities, AND predicted probabilities (needed for smoothing).
    """
    T = len(returns)
    filtered_prob = np.zeros((T, 2))
    predicted_prob = np.zeros((T, 2))
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
        
        # 1. Prediction Step
        pred_prob = P.T @ prev_filtered
        predicted_prob[t, :] = pred_prob
        
        # 2. Update Step
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
            
        joint_0 = lik_0 * pred_prob[0]
        joint_1 = lik_1 * pred_prob[1]
        
        marginal = joint_0 + joint_1
        
        if marginal < 1e-300:
            marginal = 1e-300
            
        filtered_prob[t, 0] = joint_0 / marginal
        filtered_prob[t, 1] = joint_1 / marginal
        
        log_likelihood += np.log(marginal) + max_ll
        
        prev_filtered = filtered_prob[t, :]
        
    return log_likelihood, filtered_prob, predicted_prob

@njit(cache=True)
def hamilton_filter_t_jit(returns, mu_0, mu_1, sigma2_0, sigma2_1, p00, p11, nu_0, nu_1):
    """Wrapper for backward compatibility if needed, returns just LL and filtered."""
    ll, filt, _ = hamilton_filter_t_details_jit(returns, mu_0, mu_1, sigma2_0, sigma2_1, p00, p11, nu_0, nu_1)
    return ll, filt

@njit(cache=True)
def kim_smoother_t_jit(filtered_prob, predicted_prob, p00, p11):
    """
    Kim Smoother (Backward pass) for Regime Switching.
    Calculates P(S_t = i | Y_T) (smoothed probabilities).
    """
    T = filtered_prob.shape[0]
    smoothed_prob = np.zeros((T, 2))
    
    # Initialize with last filtered probability
    smoothed_prob[T-1, :] = filtered_prob[T-1, :]
    
    P = np.array([[p00, 1 - p00], [1 - p11, p11]])
    
    # Backward recursion
    for t in range(T - 2, -1, -1):
        for i in range(2):
            sum_val = 0.0
            for j in range(2):
                # Avoid division by zero
                denom = predicted_prob[t+1, j]
                if denom < 1e-100: 
                    denom = 1e-100
                
                # Formula: smoothed[t, i] = filtered[t, i] * sum_j ( P_ij * smoothed[t+1, j] / predicted[t+1, j] )
                sum_val += P[i, j] * smoothed_prob[t+1, j] / denom
                
            smoothed_prob[t, i] = filtered_prob[t, i] * sum_val
            
    return smoothed_prob

@njit(cache=True)
def calculate_expected_transitions_jit(smoothed_prob, filtered_prob, predicted_prob, p00, p11):
    """
    Calculate expected transitions (soft counts) for M-Step update of Transition Matrix.
    xi_{t}(i,j) = P(S_t=i, S_{t+1}=j | Y)
    Returns numerator accumulators for p00 and p11.
    """
    T = smoothed_prob.shape[0]
    P = np.array([[p00, 1 - p00], [1 - p11, p11]])
    
    sum_xi_00 = 0.0
    sum_xi_0_all = 0.0
    sum_xi_11 = 0.0
    sum_xi_1_all = 0.0
    
    # We need to compute xi_t for t = 0 to T-2
    for t in range(T - 1):
        # Calculate xi matrix for time t
        for j in range(2):
            denom = predicted_prob[t+1, j]
            if denom < 1e-100: denom = 1e-100
            
            term_j = smoothed_prob[t+1, j] / denom
            
            # For i=0
            xi_0j = term_j * P[0, j] * filtered_prob[t, 0]
            sum_xi_0_all += xi_0j
            if j == 0:
                sum_xi_00 += xi_0j
                
            # For i=1
            xi_1j = term_j * P[1, j] * filtered_prob[t, 1]
            sum_xi_1_all += xi_1j
            if j == 1:
                sum_xi_11 += xi_1j
                
    return sum_xi_00, sum_xi_0_all, sum_xi_11, sum_xi_1_all

@njit(cache=True)
def _t_log_likelihood_sum(params_arr, returns, weights):
    """
    Weighted Negative Log-Likelihood for a Single Regime (t-distribution).
    params_arr: [mu, log_sigma2, log(nu-2)]
    """
    mu = params_arr[0]
    # Bound checks hard-coded or handled by optimizer bounds, but here for safety in JIT
    sigma2 = np.exp(params_arr[1])
    nu = 2.0 + np.exp(params_arr[2])
    
    total_wll = 0.0
    for t in range(len(returns)):
        if weights[t] > 1e-8: # Optimization: skip tiny weights
            ll = _t_log_likelihood(returns[t] - mu, nu, sigma2)
            total_wll += weights[t] * ll
            
    return -total_wll # Return negative for minimization

# =============================================================================
# ESTIMATION CLASS
# =============================================================================

class MarkovSwitchingTDist:
    """Custom Estimator for Markov Switching Model with t-distribution using EM Algorithm."""
    
    def __init__(self):
        self.params = {}
        self.probs = None
        
    def fit(self, returns, max_iter=500, tol=1e-5, n_init=1, init_params=None):
        """
        Fits the model using Expectation Maximization (EM).
        """
        returns_arr = np.ascontiguousarray(returns)
        var = np.var(returns_arr)
        mean = np.mean(returns_arr)
        std = np.std(returns_arr)
        
        # 1. Initialization
        if init_params is not None:
             current_params = init_params.copy()
        else:
            try:
                n = len(returns_arr)
                window = min(20, max(5, n // 10))
                rolling_std = pd.Series(returns_arr).rolling(window=window).std().fillna(std)
                vol_low = np.percentile(rolling_std, 20)
                vol_high = np.percentile(rolling_std, 80)
                sigma2_0_init = max(vol_low**2, 1e-6)
                sigma2_1_init = max(vol_high**2, 1e-6)
            except Exception:
                sigma2_0_init = var * 0.5
                sigma2_1_init = var * 2.0
                
            # Initial guess
            # Perturb means slightly to distinguish regimes during EM and improve convergence
            current_params = {
                'mu_0': mean - 0.2 * std, 
                'mu_1': mean + 0.2 * std,
                'sigma2_0': sigma2_0_init, 'sigma2_1': sigma2_1_init,
                'p00': 0.9, 'p11': 0.9,
                'nu_0': 10.0, 'nu_1': 6.0
            }
        
        best_overall_ll = -np.inf
        
        prev_ll = -np.inf
        
        for iteration in range(max_iter):
            # ==================
            # E-STEP
            # ==================
            
            # 1. Run Filter
            ll_curr, filt_prob, pred_prob = hamilton_filter_t_details_jit(
                returns_arr,
                current_params['mu_0'], current_params['mu_1'],
                current_params['sigma2_0'], current_params['sigma2_1'],
                current_params['p00'], current_params['p11'],
                current_params['nu_0'], current_params['nu_1']
            )
            
            # Check convergence
            if abs(ll_curr - prev_ll) < tol and iteration > 0:
                # if iteration > 5:
                #     print(f"      Converged in {iteration} iterations (LL: {ll_curr:.2f})")
                break
            
            if iteration > 0 and iteration % 50 == 0:
                 print(f"      EM Iteration {iteration} (LL: {ll_curr:.4f})")
                
            prev_ll = ll_curr
            
            # 2. Run Smoother
            smoothed_prob = kim_smoother_t_jit(
                filt_prob, pred_prob, 
                current_params['p00'], current_params['p11']
            )
            
            # ==================
            # M-STEP
            # ==================
            
            # 1. Update Transition Probabilities
            sum_xi_00, sum_xi_0_all, sum_xi_11, sum_xi_1_all = calculate_expected_transitions_jit(
                smoothed_prob, filt_prob, pred_prob,
                current_params['p00'], current_params['p11']
            )
            
            # Add priors/smoothing to transitions to prevent 0 or 1
            new_p00 = (sum_xi_00 + 0.1) / (sum_xi_0_all + 0.2)
            new_p11 = (sum_xi_11 + 0.1) / (sum_xi_1_all + 0.2)
            
            current_params['p00'] = min(max(new_p00, 0.01), 0.999)
            current_params['p11'] = min(max(new_p11, 0.01), 0.999)
            
            # 2. Update Emission Parameters (Weighted MLE for t-dist)
            # Optimization method needed to wrap JIT
            def optimize_regime(weights, current_mu, current_sigma2, current_nu):
                # Initial guess for this regime
                x0_regime = np.array([
                    current_mu, 
                    np.log(current_sigma2), 
                    np.log(max(current_nu - 2.0, 1e-4))
                ])
                
                # Bounds
                b_regime = [
                    (None, None), # mu
                    (-15, 10),    # log_sigma2
                    (-10, 6)      # log(nu-2)
                ]
                
                # Wrapper for JIT objective
                def obj_func(x):
                    return _t_log_likelihood_sum(x, returns_arr, weights)
                
                try:
                    res = minimize(obj_func, x0_regime, method='L-BFGS-B', bounds=b_regime)
                    if res.success or res.fun < 1e10:
                        return res.x
                except:
                    pass
                return x0_regime # Fallback
            
            # Update Regime 0
            p0 = optimize_regime(smoothed_prob[:, 0], current_params['mu_0'], current_params['sigma2_0'], current_params['nu_0'])
            current_params['mu_0'] = p0[0]
            current_params['sigma2_0'] = np.exp(p0[1])
            current_params['nu_0'] = 2.0 + np.exp(p0[2])
            
            # Update Regime 1
            p1 = optimize_regime(smoothed_prob[:, 1], current_params['mu_1'], current_params['sigma2_1'], current_params['nu_1'])
            current_params['mu_1'] = p1[0]
            current_params['sigma2_1'] = np.exp(p1[1])
            current_params['nu_1'] = 2.0 + np.exp(p1[2])
            
        # Final pass to get probabilities with final params
        final_ll, final_filt, final_pred = hamilton_filter_t_details_jit(
            returns_arr,
            current_params['mu_0'], current_params['mu_1'],
            current_params['sigma2_0'], current_params['sigma2_1'],
            current_params['p00'], current_params['p11'],
            current_params['nu_0'], current_params['nu_1']
        )
        final_smoothed = kim_smoother_t_jit(final_filt, final_pred, current_params['p00'], current_params['p11'])
        
        current_params['log_likelihood'] = final_ll
        self.params = current_params
        self.probs = final_smoothed
        
        return self.params, self.probs

def _process_single_firm(gvkey, firm_df):
    """
    Helper function to process a single firm for regime switching estimation.
    Designed for parallel execution.
    """
    # Work on a copy to avoid Shared Memory issues
    firm_df = firm_df.copy()
    
    # Ensure date format
    if 'date' not in firm_df.columns:
        if isinstance(firm_df.index, pd.DatetimeIndex):
            firm_df['date'] = firm_df.index
    firm_df['date'] = pd.to_datetime(firm_df['date'])
    firm_df = firm_df.sort_values("date")
    
    scaled_available = "asset_return_daily_scaled" in firm_df.columns
    target_col = "asset_return_daily_scaled" if scaled_available else "asset_return_daily"
    
    # Monthly Rolling Window
    start_date = firm_df['date'].min()
    end_date = firm_df['date'].max()
    params_list = []
    
    try:
        estimation_start = start_date + pd.DateOffset(months=12)
        if estimation_start >= end_date:
            return firm_df, params_list
        month_ends = pd.date_range(start=estimation_start, end=end_date, freq='ME')
    except Exception:
        return firm_df, params_list

    last_params = None

    for date_point in month_ends:
        # Select all data up to this point
        data_up_to_point = firm_df[firm_df['date'] <= date_point]

        # Require at least 252 trading days of history
        if len(data_up_to_point) < 252:
            continue

        if len(params_list) % 12 == 0:
            print(f"    > Date: {date_point.date()} (Window {len(params_list)})")

        # Take the exact last 252 trading days for the window
        window_df = data_up_to_point.iloc[-252:].copy()
        
        # Check for missing values
        if window_df[target_col].isna().sum() > 20: 
             continue
        
        valid_mask = window_df[target_col].notna()
        if valid_mask.sum() < 200:
            continue
            
        returns = window_df.loc[valid_mask, target_col].values

        # Use the last actual trading date
        last_trading_date = window_df['date'].max()
        
        # Scale for calculation if needed
        calc_scale_factor = 100.0
        if not scaled_available:
             returns = returns * calc_scale_factor
        
        try:
            model = MarkovSwitchingTDist()
            params, probs = model.fit(returns, init_params=last_params)
            
            # Map regimes: 0 = LOW, 1 = HIGH Volatility
            p_sigma0 = params['sigma2_0']
            p_sigma1 = params['sigma2_1']
            
            if p_sigma0 > p_sigma1:
                 params['sigma2_0'], params['sigma2_1'] = params['sigma2_1'], params['sigma2_0']
                 params['mu_0'], params['mu_1'] = params['mu_1'], params['mu_0']
                 params['nu_0'], params['nu_1'] = params['nu_1'], params['nu_0']
                 params['p00'], params['p11'] = params['p11'], params['p00']
                 # Swap probabilities
                 probs[:, [0, 1]] = probs[:, [1, 0]]
            
            last_params = params

            # Stitch probabilities into firm_df (for the newest month)
            prev_month_end = date_point - pd.DateOffset(months=1)
            
            # Map back to indices (in firm_df)
            window_indices = window_df.index[valid_mask]
            window_dates = window_df.loc[valid_mask, 'date']
            
            new_month_mask = window_dates > prev_month_end
            
            if new_month_mask.any():
                update_indices = window_indices[new_month_mask]
                update_probs = probs[new_month_mask.values]
                
                firm_df.loc[update_indices, "regime_probability_0"] = update_probs[:, 0]
                firm_df.loc[update_indices, "regime_probability_1"] = update_probs[:, 1]
                firm_df.loc[update_indices, "regime_state"] = np.where(update_probs[:, 0] > 0.5, 0, 1)

            # Store params
            params_list.append({
                'gvkey': gvkey,
                'date': last_trading_date, 
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
            
        except Exception:
            continue
            
    print(f"  > Finished Firm {gvkey} ({len(params_list)} windows)")
    return firm_df, params_list

def run_regime_switching_estimation(daily_returns_df):
    """
    Estimates a 2-regime Hamilton filter on DAILY asset returns using Custom T-Dist, Parallelized.
    """
    print("Estimating Regime Switching Model (2-Regime Markov T-Dist) on DAILY Returns...")
    
    if daily_returns_df.empty:
        print("No daily returns provided.")
        return daily_returns_df
    
    # Initialize Output Structure
    df_out = daily_returns_df.copy().reset_index(drop=True)
    df_out["regime_state"] = np.nan
    df_out["regime_probability_0"] = np.nan
    df_out["regime_probability_1"] = np.nan
    
    firms = df_out["gvkey"].unique()
    # n_cores = multiprocessing.cpu_count()
    # print(f"Processing Regime Switching for {len(firms)} firms using {n_cores} cores...\\n")
    print(f"Processing Regime Switching for {len(firms)} firms (Sequential execution)...\\n")
    
    # Run Sequential Execution
    processed_dfs = []
    all_params = []
    
    for i, gvkey in enumerate(firms):
        print(f"Processing firm {gvkey} ({i+1}/{len(firms)})...")
        try:
            res_df, res_params = _process_single_firm(gvkey, df_out[df_out["gvkey"] == gvkey])
            processed_dfs.append(res_df)
            all_params.extend(res_params)
        except Exception as e:
            print(f"Error processing firm {gvkey}: {e}")
            continue

    # Reassemble main dataframe
    if processed_dfs:
        df_out = pd.concat(processed_dfs).sort_index()
    else:
        print("Warning: No firms processed successfully.")

    # Save Parameters and Merge
    if all_params:
        params_df = pd.DataFrame(all_params)
        params_df.to_csv(OUTPUT_DIR / "regime_switching_parameters.csv", index=False)
        print("Saved regime_switching_parameters.csv")
        
        # Merge rolling parameters back into daily dataframe
        # Ensure date type
        df_out['date'] = pd.to_datetime(df_out['date'])
        
        merge_cols = [c for c in params_df.columns if c not in ['gvkey', 'date', 'regime_0_ar', 'regime_1_ar']]
        
        # Drop existing if any
        df_out = df_out.drop(columns=[c for c in merge_cols if c in df_out.columns])
        
        # Merge on date (month-ends)
        merge_df = params_df[['gvkey', 'date'] + merge_cols]
        df_out = pd.merge(df_out, merge_df, on=['gvkey', 'date'], how='left')
        
        # Forward fill per firm
        df_out = df_out.sort_values(['gvkey', 'date'])
        df_out[merge_cols] = df_out.groupby('gvkey')[merge_cols].ffill()
        
        print(f"  Merged rolling RS parameters into daily returns data (Forward Filled)")
    
    return df_out
