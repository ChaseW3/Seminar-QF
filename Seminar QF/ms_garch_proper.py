"""
Proper MS-GARCH(1,1) Model Implementation

This module implements a true Markov-Switching GARCH model where each regime
has its own GARCH(1,1) dynamics:

    σ²_t|S_t = ω_{S_t} + α_{S_t} * ε²_{t-1} + β_{S_t} * σ²_{t-1}

where S_t ∈ {0, 1} follows a Markov chain with transition probabilities:
    P(S_t = j | S_{t-1} = i) = p_ij

The model is estimated using Maximum Likelihood with the Hamilton filter.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit  # logistic function for probability bounds
import warnings
warnings.filterwarnings('ignore')


class MSGARCH:
    """
    Markov-Switching GARCH(1,1) model with 2 regimes.
    
    Parameters are estimated via MLE with Hamilton filter.
    
    Regime 0: Low volatility regime
    Regime 1: High volatility regime
    
    Each regime has its own GARCH parameters:
        - omega (constant term)
        - alpha (ARCH effect)
        - beta (GARCH effect)
        - mu (mean return, optional)
    """
    
    def __init__(self, n_regimes=2):
        """
        Initialize MS-GARCH model.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes (fixed at 2 for this implementation)
        """
        self.n_regimes = n_regimes
        self.params = None
        self.filtered_probs = None
        self.smoothed_probs = None
        self.returns = None
        self.conditional_vol = None
        self.log_likelihood = None
        self.aic = None
        self.bic = None
        
    def fit(self, returns, verbose=True):
        """
        Fit the MS-GARCH model using MLE.
        
        Parameters:
        -----------
        returns : np.ndarray
            Array of returns (not in percentage)
        verbose : bool
            Whether to print fitting progress
            
        Returns:
        --------
        dict : Estimated parameters
        """
        self.returns = np.asarray(returns).flatten()
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(self.returns)
        if not np.all(valid_mask):
            if verbose:
                print(f"  Removing {np.sum(~valid_mask)} non-finite values")
            self.returns = self.returns[valid_mask]
        
        if len(self.returns) < 50:
            raise ValueError("Insufficient data points for MS-GARCH estimation")
        
        return self._fit_mle(verbose)
    
    def _get_initial_regime_classification(self, returns):
        """
        Get initial regime classification based on rolling volatility.
        Uses strict low/high classification with median as threshold.
        
        Parameters:
        -----------
        returns : np.ndarray
            Return series
            
        Returns:
        --------
        np.ndarray : Binary regime classification (0=low vol, 1=high vol)
        """
        # Calculate rolling volatility
        window = min(20, len(returns) // 10)
        window = max(window, 5)
        
        # Use pandas for rolling calculation
        returns_series = pd.Series(returns)
        rolling_vol = returns_series.rolling(window=window, min_periods=5).std()
        
        # Fill NaN with overall std
        overall_vol = rolling_vol.fillna(returns_series.std()).values
        
        # Strict classification: median split (no smoothing)
        vol_50 = np.percentile(overall_vol, 50)
        regimes = np.zeros(len(returns), dtype=int)
        regimes[overall_vol > vol_50] = 1  # High volatility regime
        
        return regimes
    
    def _fit_mle(self, verbose=True):
        """
        Fit MS-GARCH using Maximum Likelihood Estimation with Hamilton filter.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress
            
        Returns:
        --------
        dict : Estimated parameters
        """
        returns = self.returns
        T = len(returns)
        var_ret = np.var(returns)
        
        if verbose:
            print("  Fitting MS-GARCH via MLE with Hamilton filter...")
        
        # Get initial regime classification for parameter initialization
        initial_regimes = self._get_initial_regime_classification(returns)
        
        # Calculate initial parameters from regimes
        low_vol_returns = returns[initial_regimes == 0]
        high_vol_returns = returns[initial_regimes == 1]
        
        # Ensure we have returns in both regimes
        if len(low_vol_returns) < 10:
            low_vol_returns = returns[returns**2 < np.median(returns**2)]
        if len(high_vol_returns) < 10:
            high_vol_returns = returns[returns**2 >= np.median(returns**2)]
        
        var_low = np.var(low_vol_returns) if len(low_vol_returns) > 0 else var_ret * 0.5
        var_high = np.var(high_vol_returns) if len(high_vol_returns) > 0 else var_ret * 2.0
        
        # Initial parameter guess
        # [omega_0, alpha_0, beta_0, omega_1, alpha_1, beta_1, mu_0, mu_1, p00, p11]
        # Use transformed parameters for unconstrained optimization
        
        omega_0_init = var_low * 0.05
        omega_1_init = var_high * 0.05
        alpha_init = 0.08
        beta_init = 0.85
        
        # Transform initial values
        x0 = np.array([
            np.log(omega_0_init),      # log(omega_0)
            -2.0,                       # logit-like for alpha_0
            2.0,                        # logit-like for beta_0
            np.log(omega_1_init),      # log(omega_1)
            -1.5,                       # logit-like for alpha_1
            1.5,                        # logit-like for beta_1
            np.mean(low_vol_returns),  # mu_0
            np.mean(high_vol_returns), # mu_1
            2.0,                        # logit(p00) - high persistence
            2.0                         # logit(p11) - high persistence
        ])
        
        def neg_log_likelihood(params):
            """Negative log-likelihood for optimization."""
            try:
                # Transform parameters back to constrained space
                omega_0 = np.exp(params[0])
                alpha_0 = 0.3 * expit(params[1])  # bounded (0, 0.3)
                beta_0 = 0.6 + 0.39 * expit(params[2])  # bounded (0.6, 0.99)
                
                omega_1 = np.exp(params[3])
                alpha_1 = 0.5 * expit(params[4])  # bounded (0, 0.5)
                beta_1 = 0.5 + 0.49 * expit(params[5])  # bounded (0.5, 0.99)
                
                mu_0 = params[6]
                mu_1 = params[7]
                
                p00 = 0.5 + 0.49 * expit(params[8])  # bounded (0.5, 0.99)
                p11 = 0.5 + 0.49 * expit(params[9])  # bounded (0.5, 0.99)
                
                # Check stationarity constraints
                if alpha_0 + beta_0 >= 0.999 or alpha_1 + beta_1 >= 0.999:
                    return 1e10
                
                # Hamilton filter
                ll = self._hamilton_filter_likelihood(
                    returns, omega_0, alpha_0, beta_0, omega_1, alpha_1, beta_1,
                    mu_0, mu_1, p00, p11
                )
                
                if np.isnan(ll) or np.isinf(ll):
                    return 1e10
                    
                return -ll
                
            except Exception:
                return 1e10
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'disp': False}
        )
        
        # Extract final parameters
        params_opt = result.x
        omega_0 = np.exp(params_opt[0])
        alpha_0 = 0.3 * expit(params_opt[1])
        beta_0 = 0.6 + 0.39 * expit(params_opt[2])
        omega_1 = np.exp(params_opt[3])
        alpha_1 = 0.5 * expit(params_opt[4])
        beta_1 = 0.5 + 0.49 * expit(params_opt[5])
        mu_0 = params_opt[6]
        mu_1 = params_opt[7]
        p00 = 0.5 + 0.49 * expit(params_opt[8])
        p11 = 0.5 + 0.49 * expit(params_opt[9])
        
        # Store parameters
        self.params = {
            'omega_0': omega_0, 'alpha_0': alpha_0, 'beta_0': beta_0,
            'omega_1': omega_1, 'alpha_1': alpha_1, 'beta_1': beta_1,
            'mu_0': mu_0, 'mu_1': mu_1,
            'p00': p00, 'p11': p11
        }
        
        # Calculate filtered probabilities and conditional volatility
        self._calculate_filtered_probs()
        self._calculate_conditional_vol()
        
        # Calculate log-likelihood and information criteria
        self.log_likelihood = -result.fun
        n_params = 10
        self.aic = 2 * n_params - 2 * self.log_likelihood
        self.bic = np.log(T) * n_params - 2 * self.log_likelihood
        
        if verbose:
            print(f"  MLE converged: {result.success}")
            print(f"  Log-likelihood: {self.log_likelihood:.2f}")
            print(f"  Regime 0 (low vol): omega={omega_0:.6f}, alpha={alpha_0:.4f}, beta={beta_0:.4f}")
            print(f"  Regime 1 (high vol): omega={omega_1:.6f}, alpha={alpha_1:.4f}, beta={beta_1:.4f}")
            print(f"  Persistence: p00={p00:.4f}, p11={p11:.4f}")
        
        return self.params
    
    def _hamilton_filter_likelihood(self, returns, omega_0, alpha_0, beta_0,
                                    omega_1, alpha_1, beta_1, mu_0, mu_1, p00, p11):
        """
        Calculate log-likelihood using Hamilton filter.
        
        Parameters:
        -----------
        returns : np.ndarray
            Return series
        omega_0, alpha_0, beta_0 : float
            GARCH parameters for regime 0
        omega_1, alpha_1, beta_1 : float
            GARCH parameters for regime 1
        mu_0, mu_1 : float
            Mean returns for each regime
        p00, p11 : float
            Transition probabilities
            
        Returns:
        --------
        float : Log-likelihood value
        """
        T = len(returns)
        
        # Transition matrix
        P = np.array([[p00, 1 - p00],
                      [1 - p11, p11]])
        
        # Unconditional (ergodic) probabilities
        p01 = 1 - p00
        p10 = 1 - p11
        pi_0 = p10 / (p01 + p10)
        pi_1 = p01 / (p01 + p10)
        
        # Initialize
        filtered_prob = np.array([pi_0, pi_1])
        
        # Unconditional variance for initialization
        uncond_var_0 = omega_0 / max(1 - alpha_0 - beta_0, 0.001)
        uncond_var_1 = omega_1 / max(1 - alpha_1 - beta_1, 0.001)
        
        sigma2_0 = uncond_var_0
        sigma2_1 = uncond_var_1
        
        log_likelihood = 0.0
        
        for t in range(T):
            # Predicted probabilities
            pred_prob = P.T @ filtered_prob
            
            # Likelihood for each regime (Gaussian)
            eps_0 = returns[t] - mu_0
            eps_1 = returns[t] - mu_1
            
            # Ensure positive variance
            sigma2_0 = max(sigma2_0, 1e-10)
            sigma2_1 = max(sigma2_1, 1e-10)
            
            lik_0 = np.exp(-0.5 * eps_0**2 / sigma2_0) / np.sqrt(2 * np.pi * sigma2_0)
            lik_1 = np.exp(-0.5 * eps_1**2 / sigma2_1) / np.sqrt(2 * np.pi * sigma2_1)
            
            # Marginal likelihood
            marg_lik = pred_prob[0] * lik_0 + pred_prob[1] * lik_1
            
            if marg_lik < 1e-300:
                marg_lik = 1e-300
            
            log_likelihood += np.log(marg_lik)
            
            # Update filtered probabilities
            filtered_prob[0] = pred_prob[0] * lik_0 / marg_lik
            filtered_prob[1] = pred_prob[1] * lik_1 / marg_lik
            
            # Update GARCH variances for next period
            # Using regime-specific innovations
            sigma2_0_new = omega_0 + alpha_0 * eps_0**2 + beta_0 * sigma2_0
            sigma2_1_new = omega_1 + alpha_1 * eps_1**2 + beta_1 * sigma2_1
            
            sigma2_0 = sigma2_0_new
            sigma2_1 = sigma2_1_new
        
        return log_likelihood
    
    def _calculate_filtered_probs(self):
        """Calculate filtered regime probabilities using current parameters."""
        if self.params is None or self.returns is None:
            raise ValueError("Model not fitted yet")
        
        returns = self.returns
        T = len(returns)
        
        omega_0 = self.params['omega_0']
        alpha_0 = self.params['alpha_0']
        beta_0 = self.params['beta_0']
        omega_1 = self.params['omega_1']
        alpha_1 = self.params['alpha_1']
        beta_1 = self.params['beta_1']
        mu_0 = self.params['mu_0']
        mu_1 = self.params['mu_1']
        p00 = self.params['p00']
        p11 = self.params['p11']
        
        # Transition matrix
        P = np.array([[p00, 1 - p00],
                      [1 - p11, p11]])
        
        # Ergodic probabilities
        p01 = 1 - p00
        p10 = 1 - p11
        pi_0 = p10 / (p01 + p10)
        pi_1 = p01 / (p01 + p10)
        
        # Initialize
        filtered_probs = np.zeros((T, 2))
        filtered_prob = np.array([pi_0, pi_1])
        
        uncond_var_0 = omega_0 / max(1 - alpha_0 - beta_0, 0.001)
        uncond_var_1 = omega_1 / max(1 - alpha_1 - beta_1, 0.001)
        
        sigma2_0 = uncond_var_0
        sigma2_1 = uncond_var_1
        
        for t in range(T):
            pred_prob = P.T @ filtered_prob
            
            eps_0 = returns[t] - mu_0
            eps_1 = returns[t] - mu_1
            
            sigma2_0 = max(sigma2_0, 1e-10)
            sigma2_1 = max(sigma2_1, 1e-10)
            
            lik_0 = np.exp(-0.5 * eps_0**2 / sigma2_0) / np.sqrt(2 * np.pi * sigma2_0)
            lik_1 = np.exp(-0.5 * eps_1**2 / sigma2_1) / np.sqrt(2 * np.pi * sigma2_1)
            
            marg_lik = pred_prob[0] * lik_0 + pred_prob[1] * lik_1
            if marg_lik < 1e-300:
                marg_lik = 1e-300
            
            filtered_prob[0] = pred_prob[0] * lik_0 / marg_lik
            filtered_prob[1] = pred_prob[1] * lik_1 / marg_lik
            
            filtered_probs[t] = filtered_prob.copy()
            
            sigma2_0 = omega_0 + alpha_0 * eps_0**2 + beta_0 * sigma2_0
            sigma2_1 = omega_1 + alpha_1 * eps_1**2 + beta_1 * sigma2_1
        
        self.filtered_probs = filtered_probs
    
    def _calculate_conditional_vol(self):
        """Calculate conditional volatility series."""
        if self.params is None or self.returns is None:
            raise ValueError("Model not fitted yet")
        
        returns = self.returns
        T = len(returns)
        
        omega_0 = self.params['omega_0']
        alpha_0 = self.params['alpha_0']
        beta_0 = self.params['beta_0']
        omega_1 = self.params['omega_1']
        alpha_1 = self.params['alpha_1']
        beta_1 = self.params['beta_1']
        mu_0 = self.params['mu_0']
        mu_1 = self.params['mu_1']
        
        # Initialize variance arrays
        sigma2_0 = np.zeros(T)
        sigma2_1 = np.zeros(T)
        
        uncond_var_0 = omega_0 / max(1 - alpha_0 - beta_0, 0.001)
        uncond_var_1 = omega_1 / max(1 - alpha_1 - beta_1, 0.001)
        
        sigma2_0[0] = uncond_var_0
        sigma2_1[0] = uncond_var_1
        
        for t in range(1, T):
            eps_0 = returns[t-1] - mu_0
            eps_1 = returns[t-1] - mu_1
            
            sigma2_0[t] = omega_0 + alpha_0 * eps_0**2 + beta_0 * sigma2_0[t-1]
            sigma2_1[t] = omega_1 + alpha_1 * eps_1**2 + beta_1 * sigma2_1[t-1]
        
        # Probability-weighted conditional variance
        if self.filtered_probs is not None:
            sigma2 = self.filtered_probs[:, 0] * sigma2_0 + self.filtered_probs[:, 1] * sigma2_1
        else:
            sigma2 = 0.5 * (sigma2_0 + sigma2_1)
        
        self.conditional_vol = np.sqrt(sigma2)
        self.sigma2_0 = sigma2_0
        self.sigma2_1 = sigma2_1
    
    def get_volatility_series(self):
        """Get the conditional volatility series."""
        if self.conditional_vol is None:
            raise ValueError("Model not fitted yet")
        return self.conditional_vol
    
    def get_regime_probabilities(self):
        """Get filtered regime probabilities."""
        if self.filtered_probs is None:
            raise ValueError("Model not fitted yet")
        return self.filtered_probs
    
    def forecast_volatility(self, n_ahead=1):
        """
        Forecast volatility n periods ahead.
        
        Parameters:
        -----------
        n_ahead : int
            Number of periods to forecast
            
        Returns:
        --------
        np.ndarray : Forecasted volatilities
        """
        if self.params is None:
            raise ValueError("Model not fitted yet")
        
        omega_0 = self.params['omega_0']
        alpha_0 = self.params['alpha_0']
        beta_0 = self.params['beta_0']
        omega_1 = self.params['omega_1']
        alpha_1 = self.params['alpha_1']
        beta_1 = self.params['beta_1']
        p00 = self.params['p00']
        p11 = self.params['p11']
        
        # Transition matrix
        P = np.array([[p00, 1 - p00],
                      [1 - p11, p11]])
        
        # Last filtered probability
        prob = self.filtered_probs[-1].copy()
        
        # Last variances
        sigma2_0 = self.sigma2_0[-1]
        sigma2_1 = self.sigma2_1[-1]
        
        forecasts = np.zeros(n_ahead)
        
        for h in range(n_ahead):
            # Forecast probability
            prob = P.T @ prob
            
            # Forecast variance (using unconditional expectation for innovation)
            uncond_var_0 = omega_0 / max(1 - alpha_0 - beta_0, 0.001)
            uncond_var_1 = omega_1 / max(1 - alpha_1 - beta_1, 0.001)
            
            sigma2_0 = omega_0 + (alpha_0 + beta_0) * sigma2_0
            sigma2_1 = omega_1 + (alpha_1 + beta_1) * sigma2_1
            
            # Probability-weighted forecast
            sigma2 = prob[0] * sigma2_0 + prob[1] * sigma2_1
            forecasts[h] = np.sqrt(sigma2)
        
        return forecasts


def run_ms_garch_estimation(data, gvkey_column='gvkey', return_column='asset_return_daily',
                            output_file='ms_garch_parameters.csv', verbose=True):
    """
    Run MS-GARCH estimation for all firms in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with firm data including returns
    gvkey_column : str
        Column name for firm identifier (default 'gvkey')
    return_column : str
        Column name for returns (default 'asset_return_daily')
    output_file : str
        Path for output CSV
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    pd.DataFrame : data with MS-GARCH volatility columns added
    """
    if verbose:
        print("\n" + "="*60)
        print("MS-GARCH PARAMETER ESTIMATION (MLE)")
        print("="*60)
    
    gvkeys = data[gvkey_column].unique()
    all_params = []
    data_with_vol = data.copy()
    data_with_vol['ms_garch_volatility'] = np.nan
    data_with_vol['ms_garch_regime_prob'] = np.nan
    
    for i, gvkey in enumerate(gvkeys):
        if verbose:
            print(f"\n[{i+1}/{len(gvkeys)}] Processing {gvkey}")
        
        firm_data = data[data[gvkey_column] == gvkey].copy()
        returns = firm_data[return_column].dropna().values
        
        if len(returns) < 50:
            if verbose:
                print(f"  Skipping {gvkey}: insufficient data ({len(returns)} obs)")
            continue
        
        try:
            model = MSGARCH()
            params = model.fit(returns, verbose=verbose)
            
            # Get volatility and regime probabilities
            vol_series = model.get_volatility_series()
            regime_probs = model.get_regime_probabilities()
            
            # Store parameters with gvkey
            params['gvkey'] = gvkey
            params['log_likelihood'] = model.log_likelihood
            params['aic'] = model.aic
            params['bic'] = model.bic
            params['n_obs'] = len(returns)
            all_params.append(params)
            
            # Add to data
            valid_idx = firm_data[return_column].notna()
            firm_idx = data_with_vol[gvkey_column] == gvkey
            
            # Align lengths
            vol_idx = np.where(firm_idx)[0]
            valid_positions = np.where(valid_idx.values)[0]
            
            for j, pos in enumerate(valid_positions):
                if j < len(vol_series):
                    data_with_vol.loc[data_with_vol.index[vol_idx[pos]], 'ms_garch_volatility'] = vol_series[j]
                    data_with_vol.loc[data_with_vol.index[vol_idx[pos]], 'ms_garch_regime_prob'] = regime_probs[j, 1]
            
            if verbose:
                print(f"  Successfully estimated MS-GARCH for {gvkey}")
                
        except Exception as e:
            if verbose:
                print(f"  Error estimating MS-GARCH for {gvkey}: {str(e)}")
            continue
    
    # Create parameters DataFrame
    params_df = pd.DataFrame(all_params)
    
    if len(params_df) > 0:
        # Reorder columns
        cols = ['gvkey', 'omega_0', 'alpha_0', 'beta_0', 'omega_1', 'alpha_1', 'beta_1',
                'mu_0', 'mu_1', 'p00', 'p11', 'log_likelihood', 'aic', 'bic', 'n_obs']
        params_df = params_df[[c for c in cols if c in params_df.columns]]
        
        # Save parameters
        params_df.to_csv(output_file, index=False)
        if verbose:
            print(f"\nMS-GARCH parameters saved to {output_file}")
    
    if verbose:
        print("\n" + "="*60)
        print("MS-GARCH ESTIMATION COMPLETE")
        print(f"Successfully estimated: {len(params_df)} firms")
        print("="*60)
    
    return data_with_vol

