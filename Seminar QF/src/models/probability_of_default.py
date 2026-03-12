# Merton probability of default using GARCH, Regime Switching, and MS-GARCH volatility

import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def _add_regime_weighted_volatility(df):
    # Add regime-weighted volatility column for Regime Switching model.
    # Loads regime parameters (regime_0_vol, regime_1_vol) from saved file
    # and computes: sigma_t = P(regime=0) * sigma_0 + P(regime=1) * sigma_1
    try:
        from src.utils import config
        params_file = config.OUTPUT_DIR / 'regime_switching_parameters.csv'
    except ImportError:
        from pathlib import Path
        params_file = Path(__file__).resolve().parent.parent.parent / "data" / "output" / "regime_switching_parameters.csv"
    
    # Load regime parameters
    try:
        params_df = pd.read_csv(params_file)
    except FileNotFoundError:
        print(f"Regime parameters file not found: {params_file}")
        # Fallback: use asset_volatility if available
        if 'asset_volatility' in df.columns:
            df['regime_volatility'] = df['asset_volatility']
        return df
    
    # Create a mapping of gvkey -> (sigma_0, sigma_1)
    params_dict = {}
    for _, row in params_df.iterrows():
        gvkey = row['gvkey']
        params_dict[gvkey] = (row['regime_0_vol'], row['regime_1_vol'])
    
    # Compute regime-weighted volatility
    def compute_regime_vol(row):
        gvkey = row['gvkey']
        if gvkey not in params_dict:
            return np.nan
        sigma_0, sigma_1 = params_dict[gvkey]
        prob_0 = row.get('regime_probability_0', 0.5)
        prob_1 = row.get('regime_probability_1', 0.5)
        # Handle NaN probabilities
        if pd.isna(prob_0) or pd.isna(prob_1):
            return np.nan
        # Weighted average volatility
        return prob_0 * sigma_0 + prob_1 * sigma_1
    
    df['regime_volatility'] = df.apply(compute_regime_vol, axis=1)
    
    valid_count = df['regime_volatility'].notna().sum()
    
    return df


def load_auxiliary_data():
    # Load liabilities (from Excel) and interest rates (from ECB CSV) for daily PD calculation
    
    try:
        from src.utils import config
    except ImportError:
        from src.utils import config

    # 1. Load Liabilities from Excel
    try:
        liab_df = pd.read_excel(config.EQUITY_DATA_FILE, sheet_name=1)
        liab_df = liab_df.rename(columns={
            "(gvkey) Global Company Key - Company": "gvkey",
            "(fyear) Data Year - Fiscal": "fyear",
            "(lt) Liabilities - Total": "liabilities_total",
        })
        liab_df = liab_df[["gvkey", "fyear", "liabilities_total"]].drop_duplicates(subset=["gvkey", "fyear"])
        
        # CRITICAL: Liabilities in the data are in MILLIONS - convert to actual currency units
        # to match asset values (which are in full units from market cap * shares)
        liab_df["liabilities_total"] = liab_df["liabilities_total"] * 1_000_000
    except Exception as e:
        return None, None

    # 2. Load Interest Rates from ECB data
    try:
        rates_df = pd.read_csv(config.INTEREST_RATES_FILE)
        rate_cols = [col for col in rates_df.columns if 'EURIBOR' in col.upper()]
        rates_df['DATE'] = pd.to_datetime(rates_df['DATE'])
        rates_df['month_year'] = rates_df['DATE'].dt.strftime('%Y-%m')
        rates_df['risk_free_rate'] = pd.to_numeric(rates_df[rate_cols[0]], errors='coerce') / 100
        rates_df = rates_df[['month_year', 'risk_free_rate']].drop_duplicates()
    except Exception as e:
        return liab_df, None
    
    return liab_df, rates_df


def calculate_pd_for_model(model_name, file_path, liabilities_df, rates_df):
    # Calculate Merton PD for one volatility model (GARCH, Regime Switching, or MS-GARCH) using daily data
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract fiscal year from date
    df['fyear'] = df['date'].dt.year
    
    # Merge with liabilities (match by gvkey and fiscal year)
    df_merged = pd.merge(df, liabilities_df, on=['gvkey', 'fyear'], how='left')
    
    # Merge with interest rates (match by month)
    df_merged['month_year'] = df_merged['date'].dt.strftime('%Y-%m')
    df_merged = pd.merge(df_merged, rates_df[['month_year', 'risk_free_rate']], 
                         on='month_year', how='left')
    
    # Fill missing rates with forward/backward fill (modern pandas syntax)
    df_merged['risk_free_rate'] = df_merged.groupby('gvkey')['risk_free_rate'].transform(
        lambda x: x.ffill().bfill()
    )
    
    # Fill remaining with default rate
    df_merged['risk_free_rate'] = df_merged['risk_free_rate'].fillna(0.05)
    
    # Determine volatility column based on model
    if model_name == 'GARCH':
        vol_col = 'garch_volatility'
    elif model_name == 'Regime Switching':
        # Regime switching needs special handling - compute regime-weighted volatility
        vol_col = 'regime_volatility'
        df_merged = _add_regime_weighted_volatility(df_merged)
    elif model_name == 'MS-GARCH':
        vol_col = 'ms_garch_volatility'  # Correct column name
    else:
        vol_col = 'garch_volatility'
    
    # Check if volatility column exists
    if vol_col not in df_merged.columns:
        return None
    
    # Create PD column name (replace spaces and hyphens with underscores)
    col_name = f'pd_{model_name.lower().replace(" ", "_").replace("-", "_")}'
    df_merged[col_name] = np.nan
    
    # Validity mask
    mask_valid = (
        (df_merged[vol_col].notna()) & 
        (df_merged['asset_value'] > 0) & 
        (df_merged['liabilities_total'].notna()) &
        (df_merged['liabilities_total'] > 0)
    )
    
    if not mask_valid.any():
        print(f"    No valid data for {model_name} PD calculation")
        print(f"       vol_col notna: {df_merged[vol_col].notna().sum()}")
        print(f"       asset_value > 0: {(df_merged['asset_value'] > 0).sum()}")
        print(f"       liabilities notna: {df_merged['liabilities_total'].notna().sum()}")
        print(f"       liabilities > 0: {(df_merged['liabilities_total'] > 0).sum()}")
        return None
    
    valid_data = df_merged.loc[mask_valid]
    
    # Merton PD calculation
    V_A = valid_data['asset_value'].values
    B = valid_data['liabilities_total'].values  # Already scaled in data_processing.py
    
    # CRITICAL: Volatility in the CSVs is DAILY (sigma_daily).
    # Merton formula requires ANNUALIZED volatility for T=1.
    sigma_A_daily = valid_data[vol_col].values
    sigma_A = sigma_A_daily * np.sqrt(252) # Convert to annualized
    
    r = valid_data['risk_free_rate'].values
    T = 1.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d2 = (np.log(V_A / B) + (r - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
        pd_values = norm.cdf(-d2)
        pd_values = np.clip(pd_values, 0, 1)
    
    df_merged.loc[mask_valid, col_name] = pd_values
    
    
    return df_merged


def run_pd_pipeline(data_garch, data_regime, data_msgarch):
    # Complete pipeline: calculate PD for all three models and combine into a single daily DataFrame
    
    print("\nProbability of default")

    liabilities_df, rates_df = load_auxiliary_data()
    
    if liabilities_df is None or rates_df is None:
        return pd.DataFrame()
    
    print("\nCalculating PD for each model")
    
    # GARCH
    df_garch = calculate_pd_for_model('GARCH', data_garch, liabilities_df, rates_df)
    if df_garch is None:
        return pd.DataFrame()
    
    # Select only necessary columns from GARCH
    final_df = df_garch[['gvkey', 'date', 'asset_value', 'liabilities_total', 
                          'risk_free_rate', 'pd_garch']].copy()
    
    # Also include garch_volatility if it exists
    if 'garch_volatility' in df_garch.columns:
        final_df['garch_volatility'] = df_garch['garch_volatility']
    
    # Regime Switching
    df_regime = calculate_pd_for_model('Regime Switching', data_regime, liabilities_df, rates_df)
    if df_regime is not None:
        # Check what PD column actually exists
        pd_col_name = 'pd_regime_switching'
        if pd_col_name in df_regime.columns:
            cols_to_merge = df_regime[['gvkey', 'date', pd_col_name]].copy()
            final_df = pd.merge(final_df, cols_to_merge, on=['gvkey', 'date'], how='left')
            print(f"Merged Regime Switching results")
        else:
            print(f"PD column '{pd_col_name}' not found in Regime Switching results")
    else:
        print("Regime Switching model skipped")
    
    # MS-GARCH
    df_msgarch = calculate_pd_for_model('MS-GARCH', data_msgarch, liabilities_df, rates_df)
    if df_msgarch is not None:
        # Check what PD column actually exists
        pd_col_name = 'pd_ms_garch'
        if pd_col_name in df_msgarch.columns:
            cols_to_merge = df_msgarch[['gvkey', 'date', pd_col_name]].copy()
            final_df = pd.merge(final_df, cols_to_merge, on=['gvkey', 'date'], how='left')
            print(f"Merged MS-GARCH results")
        else:
            # Debug: print available columns
            available_cols = [col for col in df_msgarch.columns if 'pd_' in col]
    else:
        print("MS-GARCH model skipped")
    
    # Summary and Return
    print(f"\nPD pipeline complete")
    print(f"  Total records: {len(final_df):,}")
    print(f"  Firms: {final_df['gvkey'].nunique()}")
    if len(final_df) > 0:
        print(f"  Date range: {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}")
    
    return final_df


def calculate_merton_pd_normal(daily_returns_file):
    # Calculate Merton PD using rolling asset volatility (no GARCH); benchmark model
    print("\nCalculating Merton PD")
    
    # Load data
    df = pd.read_csv(daily_returns_file)
    df['date'] = pd.to_datetime(df['date'])
    df['fyear'] = df['date'].dt.year
    
    liabilities_df, rates_df = load_auxiliary_data()
    
    if liabilities_df is None or rates_df is None:
        return pd.DataFrame()
    
    # Merge with liabilities and rates
    df = pd.merge(df, liabilities_df, on=['gvkey', 'fyear'], how='left')
    df['month_year'] = df['date'].dt.strftime('%Y-%m')
    df = pd.merge(df, rates_df, on='month_year', how='left')
    
    # Fill missing rates (modern pandas syntax)
    df['risk_free_rate'] = df.groupby('gvkey')['risk_free_rate'].transform(
        lambda x: x.ffill().bfill()
    )
    df['risk_free_rate'] = df['risk_free_rate'].fillna(0.05)
    
    # Use asset_volatility from Merton estimation (stored as DAILY volatility)
    df_clean = df.dropna(subset=['asset_value', 'liabilities_total', 'asset_volatility', 'risk_free_rate'])
    df_clean = df_clean[(df_clean['asset_value'] > 0) & (df_clean['liabilities_total'] > 0)]
    
    # Merton PD calculation
    V_A = df_clean['asset_value'].values
    B = df_clean['liabilities_total'].values  # Already scaled in data_processing.py
    sigma_A_daily = df_clean['asset_volatility'].values
    sigma_A = sigma_A_daily * np.sqrt(252)
    r = df_clean['risk_free_rate'].values
    T = 1.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d2 = (np.log(V_A / B) + (r - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
        df_clean['pd_merton_normal'] = norm.cdf(-d2)
        df_clean['pd_merton_normal'] = df_clean['pd_merton_normal'].clip(0, 1)
    
    
    return df_clean[['gvkey', 'date', 'asset_value', 'liabilities_total', 
                      'asset_volatility', 'pd_merton_normal']]


# Compute regime-weighted volatility from regime state and transition probabilities
# The file has: regime_state, regime_probability_0, regime_probability_1
# but no direct volatility column; must be computed from per-regime parameters

def get_regime_volatility(df_regime):
    # Calculate regime-weighted volatility: σ = p0*σ0 + p1*σ1 using saved regime parameters;
    # falls back to asset_volatility or a 30% default if no parameter file is found
    import os
    
    # Try to load regime parameters
    params_file = 'regime_switching_parameters.csv'
    if os.path.exists(params_file):
        params_df = pd.read_csv(params_file)
        
        # Merge parameters with data
        df_merged = pd.merge(
            df_regime, 
            params_df[['gvkey', 'regime_0_vol', 'regime_1_vol']], 
            on='gvkey', 
            how='left'
        )
        
        # Calculate volatility based on regime state and probabilities
        # Weighted average: σ = p0*σ0 + p1*σ1
        df_merged['regime_volatility'] = (
            df_merged['regime_probability_0'] * df_merged['regime_0_vol'] +
            df_merged['regime_probability_1'] * df_merged['regime_1_vol']
        )
        
        return df_merged['regime_volatility']
    else:
        # Fallback: use asset_volatility if available
        if 'asset_volatility' in df_regime.columns:
            return df_regime['asset_volatility']
        else:
            return pd.Series([0.3] * len(df_regime))  # Default 30% vol
