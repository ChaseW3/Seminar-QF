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

where K_i(τ) is promised debt repayment and P^(m)_i,t(τ) is the value of the 
implicit put option on firm assets (Black-Scholes form).

Model-implied credit spread (risk-neutral) with recovery rate R:
    s^(m)_i,t(τ) = y^(m)_i,t(τ) - r_t
                 = -(1/τ) ln(1 - (1-R) × P^(m)_i,t(τ) / (K_i(τ) e^(-r_t τ)))

where R is the recovery rate (default: 25% for senior unsecured debt).
The loss given default (LGD) = 1 - R scales the expected loss.
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
    
    def __init__(self, maturity_horizons=[1, 3, 5], recovery_rate=0.25):
        """
        Initialize CDS Spread Calculator.
        
        Parameters:
        -----------
        maturity_horizons : list
            CDS maturities in years (1Y, 3Y, 5Y per Section 2.4.2)
        recovery_rate : float
            Recovery rate on defaulted debt (default: 0.25 or 25%)
            Typical market convention: 25-40% for senior unsecured debt
        """
        self.maturity_horizons = maturity_horizons
        self.recovery_rate = recovery_rate
        
        print(f"\nCDS Spread Calculator initialized (Section 2.4.2):")
        print(f"  CDS Maturities: {self.maturity_horizons} years")
        print(f"  Recovery Rate: {self.recovery_rate*100:.0f}% (Loss Given Default: {(1-self.recovery_rate)*100:.0f}%)")
        print(f"  Methodology: Malone et al. (2009) - Risk-neutral valuation\n")
    
    
    def _add_regime_weighted_volatility_for_cds(self, df_base, df_regime):
        """
        Add regime-weighted volatility for Regime Switching model CDS calculation.
        
        Loads regime parameters and computes:
            sigma_t = P(regime=0) * sigma_0 + P(regime=1) * sigma_1
        
        Parameters:
        -----------
        df_base : pd.DataFrame
            Base DataFrame with gvkey, date, asset_value, liabilities_total
        df_regime : pd.DataFrame
            Regime switching results with regime_probability_0, regime_probability_1
            
        Returns:
        --------
        pd.DataFrame with 'regime_volatility' column
        """
        # Try to load regime parameters
        try:
            from src.utils import config
            params_file = config.OUTPUT_DIR / 'regime_switching_parameters.csv'
        except ImportError:
            params_file = os.path.join(os.path.dirname(__file__), '..', '..', 
                                       'data', 'output', 'regime_switching_parameters.csv')
        
        # Merge base with regime data
        regime_cols = ['gvkey', 'date', 'regime_probability_0', 'regime_probability_1']
        available_cols = [c for c in regime_cols if c in df_regime.columns]
        df_merged = pd.merge(df_base, df_regime[available_cols], on=['gvkey', 'date'], how='left')
        
        # Load parameters
        try:
            params_df = pd.read_csv(params_file)
            print(f"    ✓ Loaded regime parameters from {os.path.basename(str(params_file))}")
            
            # Create mapping
            params_dict = {}
            for _, row in params_df.iterrows():
                params_dict[row['gvkey']] = (row['regime_0_vol'], row['regime_1_vol'])
            
            # Compute regime-weighted volatility
            def compute_vol(row):
                gvkey = row['gvkey']
                if gvkey not in params_dict:
                    return np.nan
                sigma_0, sigma_1 = params_dict[gvkey]
                prob_0 = row.get('regime_probability_0', 0.5)
                prob_1 = row.get('regime_probability_1', 0.5)
                if pd.isna(prob_0) or pd.isna(prob_1):
                    return np.nan
                return prob_0 * sigma_0 + prob_1 * sigma_1
            
            df_merged['regime_volatility'] = df_merged.apply(compute_vol, axis=1)
            
        except FileNotFoundError:
            print(f"    ⚠ Regime parameters file not found, using asset_volatility as fallback")
            if 'asset_volatility' in df_regime.columns:
                df_merged = pd.merge(df_merged, df_regime[['gvkey', 'date', 'asset_volatility']], 
                                    on=['gvkey', 'date'], how='left', suffixes=('', '_regime'))
                df_merged['regime_volatility'] = df_merged['asset_volatility']
            else:
                df_merged['regime_volatility'] = np.nan
        
        return df_merged
    
    
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
        PD : float or array
            Risk-neutral probability of default: PD = Φ(-d2) = 1 - Φ(d2)
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
        
        # Risk-neutral probability of default: PD = Φ(-d2) = 1 - Φ(d2)
        PD = 1 - Nd2
        
        # Black-Scholes put: P = K e^(-rτ) Φ(-d2) - V Φ(-d1)
        # Note: Φ(-x) = 1 - Φ(x)
        P = K * np.exp(-r * tau) * PD - V_t * (1 - Nd1)
        
        return P, PD
    
    
    def credit_spread_from_put_value(self, V_t, K, r, sigma_V, tau):
        """
        Compute model-implied credit spread (risk-neutral) with recovery rate.
        
        From Black-Scholes put option value P and recovery rate R:
            s^(m)_i,t(τ) = -(1/τ) ln(1 - (1-R) × P / (K e^(-r τ)))
        
        The recovery rate R reduces the loss given default (LGD = 1-R), which
        reduces the expected loss and therefore the credit spread.
        
        Market conventions:
        - Senior unsecured debt: R ≈ 25-40%
        - Subordinated debt: R ≈ 10-25%
        - Our default: R = 25% (LGD = 75%)
        
        This is directly comparable to observed CDS spreads.
        
        IMPORTANT NUMERICAL ISSUE:
        -------------------------
        When (1-R)×P/(K*e^(-rτ)) approaches 1 (extreme insolvency: V << K, high volatility),
        the logarithm explodes: ln(1 - 0.9999) = -9.21 → CDS spread = 9,210 bps (1Y)
        
        With R=25% (LGD=75%), the effective put value is 0.75×P, which reduces extreme spreads.
        
        Clipping P_ratio to 0.9999 prevents crashes but:
        - Caps spreads at ~9,210 bps (1Y), ~3,070 bps (3Y), ~1,842 bps (5Y)
        - Hides model breakdown (unrealistic leverage or volatility)
        - Makes all extreme cases look identical
        
        Solution: Flag problematic observations for manual review/filtering.
        
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
        PD : float or array
            Risk-neutral probability of default over horizon tau
        """
        
        # Put option value and probability of default
        P, PD = self.black_scholes_put_value(V_t, K, r, sigma_V, tau)
        
        # Denominator: K e^(-rτ)
        denominator = K * np.exp(-r * tau)
        
        # Apply recovery rate: effective loss = (1 - R) × P
        # This is the key adjustment for CDS spreads
        LGD = 1 - self.recovery_rate  # Loss Given Default
        effective_put = LGD * P
        
        # Calculate P_ratio with recovery adjustment
        P_ratio = effective_put / np.maximum(denominator, 1e-6)
        
        # CRITICAL: Identify problematic observations where P ≥ K*e^(-rτ)
        # This indicates extreme insolvency (V << K) and/or unrealistic volatility
        problematic_mask = P_ratio >= 0.95  # Flag observations approaching or exceeding theoretical limit
        
        if np.any(problematic_mask):
            n_problematic = np.sum(problematic_mask)
            pct_problematic = 100 * n_problematic / len(P_ratio)
            
            # Calculate leverage for problematic cases (if arrays)
            if isinstance(V_t, np.ndarray) and isinstance(K, np.ndarray):
                leverage_problematic = K[problematic_mask] / V_t[problematic_mask]
                mean_leverage = np.mean(leverage_problematic)
                print(f"\n⚠️  WARNING: {n_problematic} observations ({pct_problematic:.1f}%) have (1-R)×P/(K*e^(-rτ)) ≥ 0.95")
                print(f"    This indicates severe insolvency or model breakdown")
                print(f"    Recovery rate R = {self.recovery_rate*100:.0f}%, LGD = {LGD*100:.0f}%")
                print(f"    Mean leverage (K/V) for these cases: {mean_leverage:.2f}x")
                print(f"    These observations will have capped CDS spreads (~{-(1.0/tau)*np.log(0.05):.0f} bps for {tau}Y)")
                print(f"    Recommendation: Filter firms with extreme leverage or volatility\n")
        
        # Cap P_ratio to prevent mathematical errors (ln of negative/zero)
        # BUT: This caps CDS spreads at ~9,210 bps (1Y), ~3,070 bps (3Y), ~1,842 bps (5Y)
        P_ratio_capped = np.clip(P_ratio, 0, 0.9999)
        
        # Credit spread formula with recovery: s = -(1/τ) ln(1 - (1-R) × P/(K e^(-rτ)))
        spread = -(1.0 / tau) * np.log(np.maximum(1 - P_ratio_capped, 1e-6))
        
        # Convert to basis points
        spread_bps = spread * 10000
        
        # Mark capped observations with NaN (optional - uncomment to exclude from analysis)
        # spread[problematic_mask] = np.nan
        # spread_bps[problematic_mask] = np.nan
        
        return spread, spread_bps, PD
    
    
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
        K = df_model['liabilities_total'].values  # Already scaled in data_processing.py
        sigma_V = df_model[volatility_column].values
        
        # Calculate spreads and PD for each maturity
        for tau in self.maturity_horizons:
            spread, spread_bps, PD = self.credit_spread_from_put_value(
                V_t=V_t,
                K=K,
                r=r_t if isinstance(r_t, (int, float)) else r_t,
                sigma_V=sigma_V,
                tau=tau
            )
            
            df_spreads[f'cds_spread_{tau}y'] = spread
            df_spreads[f'cds_spread_{tau}y_bps'] = spread_bps
            df_spreads[f'pd_{tau}y'] = PD
        
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
        
        # For Regime Switching: compute regime-weighted volatility from parameters
        df_regime_merged = self._add_regime_weighted_volatility_for_cds(df_base, df_regime)
        print(f"✓ Loaded Regime Switching volatility: {df_regime_merged['regime_volatility'].notna().sum():,} observations")
        
        # Load MS-GARCH results (correct column name: ms_garch_volatility)
        df_msgarch = pd.read_csv(pd_msgarch_file)
        df_msgarch['date'] = pd.to_datetime(df_msgarch['date'])
        df_msgarch_merged = pd.merge(df_base, df_msgarch[['gvkey', 'date', 'ms_garch_volatility']], 
                                     on=['gvkey', 'date'], how='left')
        print(f"✓ Loaded MS-GARCH volatility: {df_msgarch_merged['ms_garch_volatility'].notna().sum():,} observations\n")
        
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
            df_msgarch_merged, 'MS-GARCH', volatility_column='ms_garch_volatility'
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
                                        merton_file, output_file=None, volatility_column=None,
                                        model_name='GARCH'):
        """
        Calculate model-implied CDS spreads based on Monte Carlo Integrated Variance forecasts.
        
        Uses annualized volatility = sqrt(Integrated Variance).
        
        Parameters:
        -----------
        mc_garch_file : str
            Path to Monte Carlo results file
        daily_returns_file : str
            Path to daily returns with asset values
        merton_file : str
            Path to Merton results with liabilities
        output_file : str, optional
            Path to save output
        volatility_column : str, optional
            Override column name for integrated variance. If None, auto-detect.
        model_name : str, optional
            Name of the model for column naming (default: 'GARCH')
            Use 'MS-GARCH' or 'Regime Switching' for those models.
        
        Returns:
        --------
        DataFrame with CDS spreads
        """
        
        overall_start = time.time()
        
        print(f"\n{'='*80}")
        print("MODEL-IMPLIED CDS SPREADS FROM MONTE CARLO (Section 2.4.2)")
        print(f"{'='*80}\n")
        
        print("Loading data files...\n")
        
        # Load Monte Carlo results
        df_mc_garch = pd.read_csv(mc_garch_file)
        df_mc_garch['date'] = pd.to_datetime(df_mc_garch['date'])
        print(f"✓ Loaded Monte Carlo results: {len(df_mc_garch):,} observations")
        print(f"  Columns: {list(df_mc_garch.columns)}\n")
        
        # Determine volatility column
        if volatility_column is not None:
            if volatility_column not in df_mc_garch.columns:
                raise ValueError(f"Column '{volatility_column}' not found! Available: {list(df_mc_garch.columns)}")
        else:
            # Auto-detect: prioritize Integrated Variance columns
            if 'mc_garch_integrated_variance' in df_mc_garch.columns:
                volatility_column = 'mc_garch_integrated_variance'
            elif 'rs_integrated_variance' in df_mc_garch.columns:
                volatility_column = 'rs_integrated_variance'
            elif 'mc_msgarch_integrated_variance' in df_mc_garch.columns:
                volatility_column = 'mc_msgarch_integrated_variance'
            # Legacy fallback
            elif 'mc_garch_cumulative_volatility' in df_mc_garch.columns:
                 volatility_column = 'mc_garch_cumulative_volatility'
                 print("⚠ Using LEGACY cumulative volatility column (Sum of Sigma).")
            else:
                # Fallback: find any variance column
                vol_cols = [c for c in df_mc_garch.columns if 'integrated' in c.lower() and 'variance' in c.lower()]
                if vol_cols:
                    volatility_column = vol_cols[0]
                else:
                    raise ValueError(f"No integrated variance column found! Available: {list(df_mc_garch.columns)}")
        
        print(f"✓ Using volatility column: '{volatility_column}'\n")
        
        # Aggregate MC simulations to mean per (gvkey,date)
        df_mc_garch_daily = df_mc_garch.groupby(['gvkey', 'date'])[[volatility_column]].mean().reset_index()
        print(f"✓ Aggregated to {len(df_mc_garch_daily):,} (gvkey,date) pairs\n")
        
        # Load daily returns with asset values
        df_daily_returns = pd.read_csv(daily_returns_file)
        df_daily_returns['date'] = pd.to_datetime(df_daily_returns['date'])
        print(f"✓ Loaded daily returns: {len(df_daily_returns):,} observations")
        
        # Load Merton results with liabilities
        df_merton = pd.read_csv(merton_file)
        df_merton['date'] = pd.to_datetime(df_merton['date'])
        print(f"✓ Loaded Merton data: {len(df_merton):,} observations\n")
        
        # Merge on both gvkey and date
        df_merged = pd.merge(df_daily_returns, df_merton[['gvkey', 'date', 'liabilities_total']], 
                             on=['gvkey', 'date'], how='left')
        df_merged = pd.merge(df_merged, df_mc_garch_daily[['gvkey', 'date', volatility_column]], 
                             on=['gvkey', 'date'], how='left')
        
        initial_rows = len(df_merged)
        df_merged = df_merged.dropna(subset=[volatility_column, 'liabilities_total', 'asset_value'])
        final_rows = len(df_merged)
        
        print(f"✓ Merged data: {initial_rows:,} → {final_rows:,} valid observations\n")
        
        # TRIPLE CHECK: Print sample before calculation
        print("SAMPLE DATA (first 5 rows):")
        print(df_merged[['gvkey', 'date', 'asset_value', 'liabilities_total', 
                          volatility_column]].head())
        print()
        
        # Extract needed columns
        V_t = df_merged['asset_value'].values.astype(np.float64)
        K = df_merged['liabilities_total'].values.astype(np.float64)  # Already scaled in data_processing.py
        integrated_variance = df_merged[volatility_column].values.astype(np.float64)
        gvkeys = df_merged['gvkey'].values
        
        # CORRECT CONVERSION: 
        # Volatility = sqrt(Integrated Variance)
        # Handle legacy "cumulative volatility" if detected (backward compatibility)
        if 'cumulative' in volatility_column:
             # Legacy approximation: sigma_annual approx sum_sigma / sqrt(252)
             # But let's assume if the column is cumulative (Sum Sigma), user knows what they are doing.
             # However, since we updated the code to produce Integrated Variance, we standard path is:
             sigma_V = integrated_variance / np.sqrt(252) # Just for legacy fallback (Sum Sigma / sqrt(T))
             print("⚠ Applying LEGACY conversion: Sigma = Sum(Sigma_Daily) / sqrt(252)")
        else:
             # STANDARD PATH for Integrated Variance
             # IV = Sum(E[Sigma^2])
             # Annualized Volatility = sqrt(IV)
             sigma_V = np.sqrt(integrated_variance)
        
        # NO CAPPING - Instead, identify and report problematic observations
        n_extreme_high = np.sum(sigma_V > 1.0)
        n_extreme_low = np.sum(sigma_V < 0.01)
        
        # Identify which firms have extreme volatility
        extreme_high_mask = sigma_V > 1.0
        if np.any(extreme_high_mask):
            extreme_firms = np.unique(gvkeys[extreme_high_mask])
            print(f"⚠️  WARNING: {len(extreme_firms)} firms have annualized volatility > 100%:")
            print(f"    Firms: {extreme_firms.tolist()}")
            print(f"    Total extreme observations: {n_extreme_high:,}")
            # print(f"    These firms should be investigated via volatility_diagnostics.py\n")
        
        print("VOLATILITY CONVERSION CHECK:")
        print(f"  Input Metric ({volatility_column}):")
        print(f"    Min: {integrated_variance.min():.6f}")
        print(f"    Max: {integrated_variance.max():.6f}")
        print(f"    Mean: {integrated_variance.mean():.6f}")
        print(f"    Median: {np.median(integrated_variance):.6f}")
        print(f"  Annualized volatility (sqrt(IV)):")
        print(f"    Min: {sigma_V.min():.4f} ({sigma_V.min()*100:.2f}%)")
        print(f"    Max: {sigma_V.max():.4f} ({sigma_V.max()*100:.2f}%)")
        print(f"    Mean: {sigma_V.mean():.4f} ({sigma_V.mean()*100:.2f}%)")
        print(f"    Median: {np.median(sigma_V):.4f} ({np.median(sigma_V)*100:.2f}%)")
        print(f"  Observations with extreme volatility: {n_extreme_high:,} high (>100%), {n_extreme_low:,} low (<1%)")
        print(f"  [Typical equity volatility: 20-50% annually]\n")
        
        # NOTE: No capping applied - use volatility_diagnostics.py to filter problematic firms
        
        # TRIPLE CHECK: Asset values and liabilities
        print("ASSET/LIABILITY CHECK:")
        print(f"  Asset values: {V_t.min():.2e} to {V_t.max():.2e}")
        print(f"  Liabilities (×10^6): {(K/1e6).min():.2e} to {(K/1e6).max():.2e}")
        print(f"  Leverage (K/V): {(K/V_t).min():.4f} to {(K/V_t).max():.4f}\n")
        
        # Load Risk-Free Rates (Euribor 1Y)
        try:
            from src.utils import config
        except ImportError:
            from src.utils import config
        rf_file = config.INTEREST_RATES_FILE
        
        if os.path.exists(rf_file):
            print(f"Loading risk-free rates from {os.path.basename(rf_file)}...")
            df_rf = pd.read_csv(rf_file)
            
            # Identify rate column (looks for 'Euribor' or uses 3rd column)
            rate_col = [c for c in df_rf.columns if 'Euribor' in c]
            if not rate_col:
                rate_col = df_rf.columns[2] # Fallback
            else:
                rate_col = rate_col[0]
                
            df_rf = df_rf.rename(columns={'DATE': 'date', rate_col: 'risk_free_rate'})
            
            # Process dates and rates
            df_rf['date'] = pd.to_datetime(df_rf['date'])
            # Convert % to decimal (e.g. 6.34 -> 0.0634)
            df_rf['risk_free_rate'] = pd.to_numeric(df_rf['risk_free_rate'], errors='coerce') / 100.0
            
            # Prepare for merge - drop incomplete rows and sort
            df_rf = df_rf[['date', 'risk_free_rate']].dropna().sort_values('date')
            
            # Map rates to the existing data (df_merged) while preserving order
            # 1. Create mapping helper
            df_mapping = df_merged[['date']].copy()
            df_mapping['orig_idx'] = df_mapping.index
            df_mapping = df_mapping.sort_values('date')
            
            # 2. Merge asof (backward: finds latest rate <= current date)
            df_rates = pd.merge_asof(
                df_mapping, 
                df_rf, 
                on='date', 
                direction='backward'
            )
            
            # 3. Restore original order to match V_t, sigma_V
            df_rates = df_rates.sort_values('orig_idx')
            
            # 4. Extract aligned rates (fill gaps with ffill or default)
            r_t = df_rates['risk_free_rate'].ffill().fillna(0.05).values
            
            print(f"✓ Loaded dynamic risk-free rates:")
            print(f"    Mean: {np.mean(r_t)*100:.2f}% | Min: {np.min(r_t)*100:.2f}% | Max: {np.max(r_t)*100:.2f}%")
            
        else:
            print(f"⚠ Risk-free rate file not found. Defaulting to 5%.")
            r_t = 0.05
        
        print("Calculating CDS spreads...\n")
        
        # Sanitize model name for column names (lowercase, replace spaces with underscores)
        model_tag = model_name.lower().replace('-', '').replace(' ', '_')
        
        # Determine suffix: don't add '_mc' if model name already contains 'mc'
        if 'mc' in model_tag or 'monte' in model_tag.lower():
            suffix = ''  # Model name already indicates Monte Carlo (e.g., 'merton_mc', 'garch_mc')
        else:
            suffix = '_mc'  # Add '_mc' for models like 'garch', 'regime_switching', 'msgarch'
        
        # Calculate spreads and PD for each maturity
        results_data = {'gvkey': df_merged['gvkey'].values, 
                        'date': df_merged['date'].values}
        
        for tau in self.maturity_horizons:
            spread, spread_bps, PD = self.credit_spread_from_put_value(V_t, K, r_t, sigma_V, tau)
            results_data[f'cds_spread_{model_tag}{suffix}_{tau}y'] = spread
            results_data[f'cds_spread_{model_tag}{suffix}_{tau}y_bps'] = spread_bps
            results_data[f'pd_{model_tag}{suffix}_{tau}y'] = PD
        
        df_cds_spreads = pd.DataFrame(results_data)
        
        print(f"✓ CDS spreads and PD calculated\n")
        
        # Print summary statistics
        print(f"{'='*80}")
        print("CDS SPREAD SUMMARY STATISTICS (basis points)")
        print(f"{'='*80}\n")
        
        for tau in self.maturity_horizons:
            col_bps = f'cds_spread_{model_tag}{suffix}_{tau}y_bps'
            
            if col_bps in df_cds_spreads.columns:
                mean_spread = df_cds_spreads[col_bps].mean()
                median_spread = df_cds_spreads[col_bps].median()
                std_spread = df_cds_spreads[col_bps].std()
                min_spread = df_cds_spreads[col_bps].min()
                max_spread = df_cds_spreads[col_bps].max()
                n_obs = df_cds_spreads[col_bps].notna().sum()
                pct_5 = df_cds_spreads[col_bps].quantile(0.05)
                pct_95 = df_cds_spreads[col_bps].quantile(0.95)
                
                print(f"Maturity {tau}Y:")
                print(f"  Mean:     {mean_spread:8.2f} bps")
                print(f"  Median:   {median_spread:8.2f} bps")
                print(f"  Std:      {std_spread:8.2f} bps")
                print(f"  5%-95%:   [{pct_5:7.2f}, {pct_95:7.2f}] bps")
                print(f"  Min-Max:  [{min_spread:7.2f}, {max_spread:7.2f}] bps")
                print(f"  N:        {n_obs:,}\n")
        
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

    def calculate_cds_spreads_analytical_merton(self, merton_file, output_file=None):
        """
        Calculate analytical Merton CDS spreads from Merton model results.
        
        Parameters:
        -----------
        merton_file : str
            Path to merged_data_with_merton.csv
        output_file : str, optional
            Path to save output.
        
        Returns:
        --------
        df_spreads : pd.DataFrame
            DataFrame with CDS spreads
        """
        from src.data.data_processing import load_interest_rates
        
        print(f"\n{'='*80}")
        print("ANALYTICAL MERTON CDS SPREADS")
        print(f"{'='*80}\n")
        
        if not os.path.exists(merton_file):
            print(f"Error: {merton_file} not found.")
            return None

        print(f"Loading Merton results from {merton_file}...")
        df_merton = pd.read_csv(merton_file)
        if 'date' in df_merton.columns:
            df_merton['date'] = pd.to_datetime(df_merton['date'])
        
        # Filter valid rows
        initial_rows = len(df_merton)
        # We need asset_volatility which comes from the Merton estimation (annualized)
        required_cols = ['asset_value', 'liabilities_total', 'asset_volatility']
        missing_cols = [c for c in required_cols if c not in df_merton.columns]
        if missing_cols:
             # Try fallback column names if needed
             pass

        df_merton = df_merton.dropna(subset=required_cols)
        print(f"Filtered {initial_rows - len(df_merton)} rows with missing values.")

        if df_merton.empty:
            print("No valid data points found.")
            return pd.DataFrame()

        # Add Interest Rates
        print("Loading Interest Rates...")
        try:
            df_rates = load_interest_rates()
            df_merton['month_year'] = df_merton['date'].dt.strftime('%Y-%m')
            
            # Merge
            print("Merging Interest Rates...")
            df_model = pd.merge(df_merton, df_rates, on='month_year', how='left')
            
            # Handle missing rates
            if df_model['risk_free_rate'].isna().any():
                print("Warning: Missing interest rates found. Forward filling...")
                df_model['risk_free_rate'] = df_model['risk_free_rate'].ffill().bfill().fillna(0.05)
        except Exception as e:
            print(f"Warning: Could not load interest rates ({e}). Using default 0.05.")
            df_model = df_merton.copy()
            df_model['risk_free_rate'] = 0.05

        # Calculate CDS Spreads
        print("Calculating CDS Spreads (Analytical Merton)...")
        
        vol_col = 'asset_volatility'
        
        df_spreads = self.calculate_cds_spreads_single_model(
            df_model=df_model,
            model_name='Merton',
            volatility_column=vol_col
        )
        
        # Rename columns to match the MC-based models format for consistency in correlation analysis
        # From: cds_spread_1y_bps -> To: cds_spread_merton_1y_bps
        rename_map = {}
        for tau in self.maturity_horizons:
            rename_map[f'cds_spread_{tau}y'] = f'cds_spread_merton_{tau}y'
            rename_map[f'cds_spread_{tau}y_bps'] = f'cds_spread_merton_{tau}y_bps'
            rename_map[f'pd_{tau}y'] = f'pd_merton_{tau}y'
        df_spreads = df_spreads.rename(columns=rename_map)
        
        # Save Results
        if output_file:
            print(f"Saving results to {output_file}...")
            df_spreads.to_csv(output_file, index=False)
            
        print("Done calculation.\n")
        
        return df_spreads

    def calculate_cds_spreads_from_all_mc_models(self, mc_garch_file, mc_regime_file, 
                                                   mc_msgarch_file, mc_merton_file,
                                                   daily_returns_file, merton_file, 
                                                   output_file=None):
        """
        Calculate CDS spreads from ALL Monte Carlo models including Merton MC.
        
        This provides a comprehensive comparison of:
        - GARCH MC (dynamic, single regime)
        - Regime Switching MC (two regimes, constant vol per regime)
        - MS-GARCH MC (two regimes, GARCH dynamics per regime)
        - Merton MC (constant volatility for fair comparison)
        
        Parameters:
        -----------
        mc_garch_file : str
            Path to GARCH MC results (daily_monte_carlo_garch_results.csv)
        mc_regime_file : str
            Path to Regime Switching MC results (daily_monte_carlo_regime_switching_results.csv)
        mc_msgarch_file : str
            Path to MS-GARCH MC results (daily_monte_carlo_ms_garch_results.csv)
        mc_merton_file : str
            Path to Merton MC results (daily_monte_carlo_merton_results.csv)
        daily_returns_file : str
            Path to daily returns with asset values
        merton_file : str
            Path to Merton results with liabilities
        output_file : str, optional
            Where to save combined results
            
        Returns:
        --------
        pd.DataFrame with CDS spreads for all four MC models and all maturities
        """
        
        overall_start = time.time()
        
        print(f"\n{'='*80}")
        print("CDS SPREADS FROM ALL MONTE CARLO MODELS")
        print(f"{'='*80}\n")
        
        print("This will calculate CDS spreads for:")
        print("  1. GARCH Monte Carlo (dynamic volatility)")
        print("  2. Regime Switching Monte Carlo (two regime states)")
        print("  3. MS-GARCH Monte Carlo (regime-dependent GARCH)")
        print("  4. Merton Monte Carlo (constant volatility baseline)\n")
        
        # Calculate CDS spreads for each MC model
        print("="*80)
        print("1/4: GARCH Monte Carlo")
        print("="*80)
        df_garch = self.calculate_cds_spreads_from_mc_garch(
            mc_garch_file=mc_garch_file,
            daily_returns_file=daily_returns_file,
            merton_file=merton_file,
            output_file='data/output/cds_spreads_garch_mc_all_firms.csv',
            volatility_column='mc_garch_integrated_variance',
            model_name='GARCH'
        )
        
        print("\n" + "="*80)
        print("2/4: Regime Switching Monte Carlo")
        print("="*80)
        df_regime = self.calculate_cds_spreads_from_mc_garch(
            mc_garch_file=mc_regime_file,
            daily_returns_file=daily_returns_file,
            merton_file=merton_file,
            output_file='data/output/cds_spreads_regime_switching_mc_all_firms.csv',
            volatility_column='rs_integrated_variance',
            model_name='Regime Switching'
        )
        
        print("\n" + "="*80)
        print("3/4: MS-GARCH Monte Carlo")
        print("="*80)
        df_msgarch = self.calculate_cds_spreads_from_mc_garch(
            mc_garch_file=mc_msgarch_file,
            daily_returns_file=daily_returns_file,
            merton_file=merton_file,
            output_file='data/output/cds_spreads_ms_garch_mc_all_firms.csv',
            volatility_column='mc_msgarch_integrated_variance',
            model_name='MS-GARCH'
        )
        
        print("\n" + "="*80)
        print("4/4: Merton Monte Carlo (Constant Volatility)")
        print("="*80)
        df_merton_mc = self.calculate_cds_spreads_from_mc_garch(
            mc_garch_file=mc_merton_file,
            daily_returns_file=daily_returns_file,
            merton_file=merton_file,
            output_file='data/output/cds_spreads_merton_mc_all_firms.csv',
            volatility_column='merton_mc_integrated_variance',
            model_name='Merton MC'
        )
        
        # Merge all results
        print("\n" + "="*80)
        print("MERGING RESULTS")
        print("="*80 + "\n")
        
        # Start with GARCH as base
        df_combined = df_garch[['gvkey', 'date']].copy()
        
        # Add CDS spreads from each model
        for tau in self.maturity_horizons:
            # GARCH MC
            garch_col = f'cds_spread_garch_mc_{tau}y_bps'
            if garch_col in df_garch.columns:
                df_combined[garch_col] = df_garch[garch_col]
            
            # Regime Switching MC
            regime_col = f'cds_spread_regimeswitching_mc_{tau}y_bps'
            if regime_col in df_regime.columns:
                df_combined = pd.merge(
                    df_combined, 
                    df_regime[['gvkey', 'date', regime_col]],
                    on=['gvkey', 'date'],
                    how='left'
                )
            
            # MS-GARCH MC
            msgarch_col = f'cds_spread_msgarch_mc_{tau}y_bps'
            if msgarch_col in df_msgarch.columns:
                df_combined = pd.merge(
                    df_combined,
                    df_msgarch[['gvkey', 'date', msgarch_col]],
                    on=['gvkey', 'date'],
                    how='left'
                )
            
            # Merton MC
            merton_mc_col = f'cds_spread_mertonmc_mc_{tau}y_bps'
            if merton_mc_col in df_merton_mc.columns:
                df_combined = pd.merge(
                    df_combined,
                    df_merton_mc[['gvkey', 'date', merton_mc_col]],
                    on=['gvkey', 'date'],
                    how='left'
                )
        
        # Print summary comparison
        print("="*80)
        print("CDS SPREAD COMPARISON SUMMARY (5Y Maturity, basis points)")
        print("="*80 + "\n")
        
        comparison_cols = [
            ('GARCH MC', 'cds_spread_garch_mc_5y_bps'),
            ('Regime Switching MC', 'cds_spread_regimeswitching_mc_5y_bps'),
            ('MS-GARCH MC', 'cds_spread_msgarch_mc_5y_bps'),
            ('Merton MC (Constant)', 'cds_spread_mertonmc_mc_5y_bps')
        ]
        
        print(f"{'Model':<25} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'N':>10}")
        print("-" * 95)
        
        for model_name, col in comparison_cols:
            if col in df_combined.columns:
                data = df_combined[col].dropna()
                if len(data) > 0:
                    print(f"{model_name:<25} {data.mean():>10.2f} {data.median():>10.2f} "
                          f"{data.std():>10.2f} {data.min():>10.2f} {data.max():>10.2f} {len(data):>10,}")
        
        print("\n" + "="*80)
        
        # Save combined results
        if output_file is None:
            output_file = 'data/output/cds_spreads_comparison_all_mc.csv'
        
        df_combined.to_csv(output_file, index=False)
        
        overall_time = time.time() - overall_start
        
        print(f"\n{'='*80}")
        print("ALL MONTE CARLO CDS SPREADS COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {timedelta(seconds=int(overall_time))}")
        print(f"Total observations: {len(df_combined):,}")
        print(f"Unique firms: {df_combined['gvkey'].nunique()}")
        print(f"Saved to: {output_file}")
        print(f"{'='*80}\n")
        
        return df_combined


