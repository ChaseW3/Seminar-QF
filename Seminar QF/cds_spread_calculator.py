"""
CDS Spread Calculator
=====================
Calculates model-implied CDS spreads based on structural credit risk models.

Theory (Section 2.4.2):
-----------------------
Following Malone et al. (2009), model-implied credit spreads are computed from the 
valuation of risky debt under each structural model specification.

Market value of risky debt:
    D^(m)_i,t(τ) = K_i(τ) e^(-r_t τ) - P^(m)_i,t(τ)

Model-implied credit spread (risk-neutral):
    s^(m)_i,t(τ) = -(1/τ) ln(1 - P^(m)_i,t(τ) / (K_i(τ) e^(-r_t τ)))
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import timedelta
from scipy.special import ndtr

CACHE_DIR = './intermediates/'
os.makedirs(CACHE_DIR, exist_ok=True)


class CDSSpreadCalculator:
    """
    Calculate model-implied CDS spreads from structural credit risk models.
    
    Follows the methodology of Malone et al. (2009) and Section 2.4.2 of the paper.
    
    Attributes:
    -----------
    maturity_horizons : list
        CDS maturities τ in years. Default: [1, 3, 5]
    """
    
    def __init__(self, maturity_horizons=[1, 3, 5]):
        """
        Initialize CDS Spread Calculator.
        
        Parameters:
        -----------
        maturity_horizons : list
            CDS maturities in years (1Y, 3Y, 5Y per Section 2.4.2)
        """
        self.maturity_horizons = maturity_horizons
        
        print(f"\nCDS Spread Calculator initialized (Section 2.4.2):")
        print(f"  CDS Maturities: {self.maturity_horizons} years")
        print(f"  Methodology: Malone et al. (2009) - Risk-neutral valuation\n")
    
    
    def black_scholes_put_value(self, V_t, K, r, sigma_V, tau):
        """
        Compute Black-Scholes put option value on firm assets.
        
        This is the implicit put option value P^(m)_i,t(τ) in the risky debt formula.
        
        P^(m)_i,t(τ) = K e^(-rτ) Φ(-d2) - V Φ(-d1)
        
        Parameters:
        -----------
        V_t : float or array
            Current firm asset value V_i,t
        K : float
            Promised debt repayment K_i(τ)
        r : float
            Risk-free rate r_t
        sigma_V : float or array
            Model-implied asset volatility σ^(m)_V,i,t(τ) (annualized)
        tau : float
            Time to maturity τ (years)
        
        Returns:
        --------
        P : float or array
            Black-Scholes put option value
        """
        
        V_t = np.asarray(V_t, dtype=np.float64)
        sigma_V = np.asarray(sigma_V, dtype=np.float64)
        
        # Ensure positive values
        V_t = np.maximum(V_t, 1e-6)
        sigma_V = np.maximum(sigma_V, 1e-4)
        K = np.maximum(K, 1e-6)
        
        # Compute d1, d2
        sig_sqrt_tau = sigma_V * np.sqrt(tau)
        d1 = (np.log(V_t / K) + (r + 0.5 * sigma_V**2) * tau) / sig_sqrt_tau
        d2 = d1 - sig_sqrt_tau
        
        # Use scipy.special.ndtr for exact normal CDF
        Nd1 = ndtr(d1)
        Nd2 = ndtr(d2)
        
        # Black-Scholes put: P = K e^(-rτ) Φ(-d2) - V Φ(-d1)
        # Note: Φ(-x) = 1 - Φ(x)
        P = K * np.exp(-r * tau) * (1 - Nd2) - V_t * (1 - Nd1)
        
        return P
    
    
    def credit_spread_from_put_value(self, V_t, K, r, sigma_V, tau):
        """
        Compute model-implied credit spread (risk-neutral).
        
        From Black-Scholes put option value P:
            s^(m)_i,t(τ) = -(1/τ) ln(1 - P / (K e^(-r τ)))
        
        This is directly comparable to observed CDS spreads.
        
        Parameters:
        -----------
        V_t : float or array
            Firm asset value V_i,t
        K : float
            Promised debt repayment K_i(τ)
        r : float
            Risk-free rate r_t
        sigma_V : float or array
            Asset volatility σ^(m)_V,i,t(τ) (annualized)
        tau : float
            Time to maturity τ (years)
        
        Returns:
        --------
        spread : float or array
            Model-implied credit spread (annualized, decimal form)
        spread_bps : float or array
            Credit spread in basis points (100 bps = 1%)
        """
        
        # Put option value
        P = self.black_scholes_put_value(V_t, K, r, sigma_V, tau)
        
        # Denominator: K e^(-rτ)
        denominator = K * np.exp(-r * tau)
        
        # Avoid division by zero and log of negative/zero
        P_ratio = P / np.maximum(denominator, 1e-6)
        P_ratio = np.clip(P_ratio, 0, 0.9999)  # Ensure valid range for ln
        
        # Credit spread formula: s = -(1/τ) ln(1 - P/(K e^(-rτ)))
        spread = -(1.0 / tau) * np.log(np.maximum(1 - P_ratio, 1e-6))
        
        # Convert to basis points
        spread_bps = spread * 10000
        
        return spread, spread_bps
    
    
    def calculate_cds_spreads_single_model(self, df_model, model_name, r_t=None, 
                                           volatility_column='asset_volatility'):
        """
        Calculate model-implied CDS spreads for a single model at all maturities.
        
        Parameters:
        -----------
        df_model : pd.DataFrame
            DataFrame with columns:
            - 'gvkey': Firm identifier
            - 'date': Date
            - 'asset_value': V_i,t (from Merton model)
            - volatility_column: σ^(m)_V,i,t (model-implied, annualized)
            - 'liabilities_total': K_i (debt face value)
            - 'risk_free_rate': r_t (risk-free rate, optional)
        model_name : str
            Name of the model (e.g., 'GARCH', 'Regime Switching', 'MS-GARCH')
        r_t : float, optional
            Risk-free rate. If None, tries to use 'risk_free_rate' column or default (0.05)
        volatility_column : str
            Name of the volatility column in df_model
        
        Returns:
        --------
        df_spreads : pd.DataFrame
            DataFrame with model-implied CDS spreads for each maturity
        """
        
        df_spreads = df_model[['gvkey', 'date']].copy()
        
        # Handle risk-free rate
        if r_t is None:
            if 'risk_free_rate' in df_model.columns:
                r_t = df_model['risk_free_rate'].values
            else:
                r_t = 0.05  # Default fallback
        
        # Get asset values and debt
        V_t = df_model['asset_value'].values
        K = df_model['liabilities_total'].values * 1_000_000  # Convert to actual values
        sigma_V = df_model[volatility_column].values
        
        # Calculate spreads for each maturity
        for tau in self.maturity_horizons:
            spread, spread_bps = self.credit_spread_from_put_value(
                V_t=V_t,
                K=K,
                r=r_t if isinstance(r_t, (int, float)) else r_t,
                sigma_V=sigma_V,
                tau=tau
            )
            
            df_spreads[f'cds_spread_{tau}y'] = spread
            df_spreads[f'cds_spread_{tau}y_bps'] = spread_bps
        
        return df_spreads
    
    
    def calculate_cds_spreads_multimodel(self, pd_garch_file, pd_regime_file, pd_msgarch_file,
                                        daily_returns_file, output_file=None):
        """
        Calculate model-implied CDS spreads for all three volatility models.
        
        Parameters:
        -----------
        pd_garch_file : str
            Path to CSV with GARCH model PD results
        pd_regime_file : str
            Path to CSV with Regime Switching model PD results
        pd_msgarch_file : str
            Path to CSV with MS-GARCH model PD results
        daily_returns_file : str
            Path to CSV with daily returns and basic asset/liability data
        output_file : str, optional
            Where to save combined results. If None, uses default.
        
        Returns:
        --------
        df_cds_all : pd.DataFrame
            Combined CDS spreads for all three models and all maturities
        """
        
        overall_start = time.time()
        
        print(f"\n{'='*80}")
        print("MODEL-IMPLIED CDS SPREADS (Section 2.4.2)")
        print(f"{'='*80}\n")
        
        print("Loading model results...\n")
        
        # Load base daily returns data
        df_base = pd.read_csv(daily_returns_file)
        df_base['date'] = pd.to_datetime(df_base['date'])
        
        print(f"✓ Loaded base data: {len(df_base):,} observations")
        
        # Load GARCH results
        df_garch = pd.read_csv(pd_garch_file)
        df_garch['date'] = pd.to_datetime(df_garch['date'])
        df_garch_merged = pd.merge(df_base, df_garch[['gvkey', 'date', 'garch_volatility']], 
                                   on=['gvkey', 'date'], how='left')
        print(f"✓ Loaded GARCH volatility: {df_garch_merged['garch_volatility'].notna().sum():,} observations")
        
        # Load Regime Switching results
        df_regime = pd.read_csv(pd_regime_file)
        df_regime['date'] = pd.to_datetime(df_regime['date'])
        df_regime_merged = pd.merge(df_base, df_regime[['gvkey', 'date', 'garch_volatility']], 
                                    on=['gvkey', 'date'], how='left')
        df_regime_merged = df_regime_merged.rename(columns={'garch_volatility': 'regime_volatility'})
        print(f"✓ Loaded Regime Switching volatility: {df_regime_merged['regime_volatility'].notna().sum():,} observations")
        
        # Load MS-GARCH results
        df_msgarch = pd.read_csv(pd_msgarch_file)
        df_msgarch['date'] = pd.to_datetime(df_msgarch['date'])
        df_msgarch_merged = pd.merge(df_base, df_msgarch[['gvkey', 'date', 'msgarch_volatility']], 
                                     on=['gvkey', 'date'], how='left')
        print(f"✓ Loaded MS-GARCH volatility: {df_msgarch_merged['msgarch_volatility'].notna().sum():,} observations\n")
        
        # Calculate CDS spreads for each model
        print("Calculating model-implied CDS spreads...\n")
        
        df_spreads_garch = self.calculate_cds_spreads_single_model(
            df_garch_merged, 'GARCH', volatility_column='garch_volatility'
        )
        df_spreads_garch = df_spreads_garch.rename(columns={
            col: col.replace('cds_spread_', 'cds_spread_garch_')
            for col in df_spreads_garch.columns if 'cds_spread_' in col
        })
        print(f"✓ GARCH CDS spreads calculated")
        
        df_spreads_regime = self.calculate_cds_spreads_single_model(
            df_regime_merged, 'Regime Switching', volatility_column='regime_volatility'
        )
        df_spreads_regime = df_spreads_regime.rename(columns={
            col: col.replace('cds_spread_', 'cds_spread_regime_')
            for col in df_spreads_regime.columns if 'cds_spread_' in col
        })
        print(f"✓ Regime Switching CDS spreads calculated")
        
        df_spreads_msgarch = self.calculate_cds_spreads_single_model(
            df_msgarch_merged, 'MS-GARCH', volatility_column='msgarch_volatility'
        )
        df_spreads_msgarch = df_spreads_msgarch.rename(columns={
            col: col.replace('cds_spread_', 'cds_spread_msgarch_')
            for col in df_spreads_msgarch.columns if 'cds_spread_' in col
        })
        print(f"✓ MS-GARCH CDS spreads calculated\n")
        
        # Merge all results
        print("Combining all model results...\n")
        
        df_cds_all = df_spreads_garch.copy()
        df_cds_all = pd.merge(df_cds_all, df_spreads_regime, on=['gvkey', 'date'], how='outer')
        df_cds_all = pd.merge(df_cds_all, df_spreads_msgarch, on=['gvkey', 'date'], how='outer')
        
        # Print summary statistics
        print(f"{'='*80}")
        print("CDS SPREAD SUMMARY STATISTICS (basis points)")
        print(f"{'='*80}\n")
        
        for tau in self.maturity_horizons:
            print(f"Maturity: {tau} years")
            print(f"{'─'*80}")
            
            for model in ['garch', 'regime', 'msgarch']:
                col = f'cds_spread_{model}_{tau}y_bps'
                if col in df_cds_all.columns:
                    mean_spread = df_cds_all[col].mean()
                    median_spread = df_cds_all[col].median()
                    std_spread = df_cds_all[col].std()
                    min_spread = df_cds_all[col].min()
                    max_spread = df_cds_all[col].max()
                    n_obs = df_cds_all[col].notna().sum()
                    
                    print(f"  {model.upper():15s} | Mean: {mean_spread:7.2f} | Median: {median_spread:7.2f} | "
                          f"Std: {std_spread:7.2f} | Range: [{min_spread:7.2f}, {max_spread:7.2f}] | N={n_obs:,}")
            print()
        
        # Save results
        if output_file is None:
            output_file = 'cds_spreads_multimodel.csv'
        
        df_cds_all.to_csv(output_file, index=False)
        
        overall_time = time.time() - overall_start
        
        print(f"{'='*80}")
        print("CDS Spread Calculation Complete")
        print(f"{'='*80}")
        print(f"Total time: {timedelta(seconds=int(overall_time))}")
        print(f"Total observations: {len(df_cds_all):,}")
        print(f"Unique firms: {df_cds_all['gvkey'].nunique()}")
        print(f"Date range: {df_cds_all['date'].min().strftime('%Y-%m-%d')} to "
              f"{df_cds_all['date'].max().strftime('%Y-%m-%d')}")
        print(f"Saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return df_cds_all
    
    def calculate_cds_spreads_from_mc_garch(self, mc_garch_file, daily_returns_file, 
                                        merton_file, output_file=None):
        """
        Calculate model-implied CDS spreads based on Monte Carlo GARCH volatility forecasts.
        Uses only mean volatility from simulations, not individual paths.
        """
        
        overall_start = time.time()
        
        print(f"\n{'='*80}")
        print("MODEL-IMPLIED CDS SPREADS FROM MONTE CARLO GARCH (Section 2.4.2)")
        print(f"{'='*80}\n")
        
        print("Loading data files...\n")
        
        # Load Monte Carlo GARCH results
        df_mc_garch = pd.read_csv(mc_garch_file)
        df_mc_garch['date'] = pd.to_datetime(df_mc_garch['date'])
        print(f"✓ Loaded Monte Carlo GARCH: {len(df_mc_garch):,} observations")
        print(f"  Columns: {list(df_mc_garch.columns)}\n")
        
        # Prefer the most appropriate volatility column (cumulative over forecast period)
        preferred = [
            'mc_garch_cumulative_volatility',  # ← FIRST priority: cumulative
            'mc_garch_volatility_forecast',
            'mc_garch_mean_daily_volatility',
            'mc_garch_mean_volatility',
            'mc_garch_volatility',
            'mc_garch_mean_vol',
        ]
        volatility_cols = [c for c in df_mc_garch.columns if ('volatility' in c.lower() or 'vol' in c.lower())]
        chosen = None
        for p in preferred:
            if p in df_mc_garch.columns:
                chosen = p
                break
        if chosen is None and volatility_cols:
            # avoid choosing a 'cumulative' column if possible
            non_cum = [c for c in volatility_cols if 'cumul' not in c.lower()]
            chosen = non_cum[0] if non_cum else volatility_cols[0]
        
        if not chosen:
            raise ValueError(f"No volatility-like column found. Available columns: {list(df_mc_garch.columns)}")
        
        volatility_column = chosen
        print(f"✓ Using volatility column: '{volatility_column}' (cumulative volatility over 252-day forecast)\n")
        
        # --- DO NOT AGGREGATE ACROSS DATES ---
        # Keep per-(gvkey,date) forecast and merge on both keys below.
        # If the MC file contains multiple simulation rows per (gvkey,date), take the mean per (gvkey,date).
        df_mc_garch_daily = df_mc_garch.groupby(['gvkey', 'date'])[[volatility_column]].mean().reset_index()
        print(f"✓ Aggregated MC simulations to {len(df_mc_garch_daily):,} unique (gvkey,date) forecast rows\n")
        
        # Load daily returns with asset values
        df_daily_returns = pd.read_csv(daily_returns_file)
        df_daily_returns['date'] = pd.to_datetime(df_daily_returns['date'])
        print(f"✓ Loaded daily returns: {len(df_daily_returns):,} observations")
        
        # Load Merton results with liabilities
        df_merton = pd.read_csv(merton_file)
        df_merton['date'] = pd.to_datetime(df_merton['date'])
        print(f"✓ Loaded Merton data: {len(df_merton):,} observations\n")
        
        # Merge all data on gvkey + date so MC forecast is date-specific
        df_merged = pd.merge(df_daily_returns, df_merton[['gvkey', 'date', 'liabilities_total']], 
                             on=['gvkey', 'date'], how='left')
        df_merged = pd.merge(df_merged, df_mc_garch_daily[['gvkey', 'date', volatility_column]], 
                             on=['gvkey', 'date'], how='left')
        
        initial_rows = len(df_merged)
        df_merged = df_merged.dropna(subset=[volatility_column, 'liabilities_total', 'asset_value'])
        final_rows = len(df_merged)
        
        print(f"✓ Merged data: {initial_rows:,} → {final_rows:,} valid observations\n")
        print("Calculating CDS spreads (date-specific mean volatility)...\n")
        
        # Extract needed columns
        V_t = df_merged['asset_value'].values
        K = df_merged['liabilities_total'].values * 1_000_000
        sigma_cumulative = df_merged[volatility_column].values
        # Cumulative volatility is already the sum; normalize to daily equivalent for Black-Scholes
        # sigma_V should be annualized daily volatility
        sigma_V = sigma_cumulative / np.sqrt(252.0)  # Convert cumulative to annualized daily vol
        r_t = 0.05  # risk-free rate
        
        print(f"✓ Converted cumulative volatility to annualized daily volatility (÷√252)\n")
        
        # Calculate spreads for each maturity (vectorized)
        results_data = {'gvkey': df_merged['gvkey'].values, 
                        'date': df_merged['date'].values}
        
        for tau in self.maturity_horizons:
            spread, spread_bps = self.credit_spread_from_put_value(V_t, K, r_t, sigma_V, tau)
            results_data[f'cds_spread_garch_mc_{tau}y'] = spread
            results_data[f'cds_spread_garch_mc_{tau}y_bps'] = spread_bps
        
        df_cds_spreads = pd.DataFrame(results_data)
        
        print(f"✓ CDS spreads calculated\n")
        
        # Print summary statistics
        print(f"{'='*80}")
        print("CDS SPREAD SUMMARY STATISTICS (basis points)")
        print(f"{'='*80}\n")
        
        for tau in self.maturity_horizons:
            col_bps = f'cds_spread_garch_mc_{tau}y_bps'
            
            if col_bps in df_cds_spreads.columns:
                mean_spread = df_cds_spreads[col_bps].mean()
                median_spread = df_cds_spreads[col_bps].median()
                std_spread = df_cds_spreads[col_bps].std()
                min_spread = df_cds_spreads[col_bps].min()
                max_spread = df_cds_spreads[col_bps].max()
                n_obs = df_cds_spreads[col_bps].notna().sum()
                
                print(f"Maturity {tau}Y:")
                print(f"  Mean:   {mean_spread:8.2f} bps")
                print(f"  Median: {median_spread:8.2f} bps")
                print(f"  Std:    {std_spread:8.2f} bps")
                print(f"  Range:  [{min_spread:7.2f}, {max_spread:7.2f}] bps")
                print(f"  N:      {n_obs:,}\n")
        
        # Save results
        if output_file is None:
            output_file = 'cds_spreads_garch_mc.csv'
        
        df_cds_spreads.to_csv(output_file, index=False)
        
        overall_time = time.time() - overall_start
        
        print(f"{'='*80}")
        print("CDS Spread Calculation Complete")
        print(f"{'='*80}")
        print(f"Total time: {timedelta(seconds=int(overall_time))}")
        print(f"Total observations: {len(df_cds_spreads):,}")
        print(f"Unique firms: {df_cds_spreads['gvkey'].nunique()}")
        print(f"File size: ~{len(df_cds_spreads) * 0.0001:.1f} MB")
        print(f"Saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return df_cds_spreads
