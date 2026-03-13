import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def _add_regime_weighted_volatility(df):
    # Adds regime weighted volatility column using probability weighted average of regime volatilities
    try:
        from src.utils import config
        params_file = config.TABLES_DIR / 'regime_switching_parameters.csv'
    except ImportError:
        from pathlib import Path
        params_file = Path(__file__).resolve().parent.parent.parent / "data" / "output" / "tables" / "regime_switching_parameters.csv"
    
    # Load regime parameters
    try:
        params_df = pd.read_csv(params_file)
    except FileNotFoundError:
        print(f"Regime parameters file not found: {params_file}")
        if 'asset_volatility' in df.columns:
            df['regime_volatility'] = df['asset_volatility']
        return df
    
    # Build a mapping of gvkey to regime volatilities
    params_dict = {}
    for _, row in params_df.iterrows():
        gvkey = row['gvkey']
        params_dict[gvkey] = (row['regime_0_vol'], row['regime_1_vol'])
    
    # Compute regime weighted volatility
    def compute_regime_vol(row):
        gvkey = row['gvkey']
        if gvkey not in params_dict:
            return np.nan
        sigma_0, sigma_1 = params_dict[gvkey]
        prob_0 = row.get('regime_probability_0', 0.5)
        prob_1 = row.get('regime_probability_1', 0.5)
        if pd.isna(prob_0) or pd.isna(prob_1):
            return np.nan
        return prob_0 * sigma_0 + prob_1 * sigma_1
    
    df['regime_volatility'] = df.apply(compute_regime_vol, axis=1)
    
    valid_count = df['regime_volatility'].notna().sum()
    
    return df


def load_auxiliary_data():
    # Loads liabilities and interest rates for daily PD calculation
    
    try:
        from src.utils import config
    except ImportError:
        from src.utils import config

    # Load liabilities from Excel
    try:
        liab_df = pd.read_excel(config.EQUITY_DATA_FILE, sheet_name=1)
        liab_df = liab_df.rename(columns={
            "(gvkey) Global Company Key - Company": "gvkey",
            "(fyear) Data Year - Fiscal": "fyear",
            "(lt) Liabilities - Total": "liabilities_total",
        })
        liab_df = liab_df[["gvkey", "fyear", "liabilities_total"]].drop_duplicates(subset=["gvkey", "fyear"])
        
        # Liabilities are in millions so convert to full units to match asset values
        liab_df["liabilities_total"] = liab_df["liabilities_total"] * 1_000_000
    except Exception as e:
        return None, None

    # Load interest rates from ECB data
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
    # Calculates Merton PD for one volatility model using daily data
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None
    
    df['date'] = pd.to_datetime(df['date'])
    df['fyear'] = df['date'].dt.year
    
    # Merge liabilities by gvkey and fiscal year
    df_merged = pd.merge(df, liabilities_df, on=['gvkey', 'fyear'], how='left')
    
    # Merge interest rates by month
    df_merged['month_year'] = df_merged['date'].dt.strftime('%Y-%m')
    df_merged = pd.merge(df_merged, rates_df[['month_year', 'risk_free_rate']], 
                         on='month_year', how='left')
    
    # Fill missing rates using forward and backward fill
    df_merged['risk_free_rate'] = df_merged.groupby('gvkey')['risk_free_rate'].transform(
        lambda x: x.ffill().bfill()
    )
    df_merged['risk_free_rate'] = df_merged['risk_free_rate'].fillna(0.05)
    
    # Select the volatility column for the given model
    if model_name == 'GARCH':
        vol_col = 'garch_volatility'
    elif model_name == 'Regime Switching':
        vol_col = 'regime_volatility'
        df_merged = _add_regime_weighted_volatility(df_merged)
    elif model_name == 'MS-GARCH':
        vol_col = 'ms_garch_volatility'
    else:
        vol_col = 'garch_volatility'
    
    if vol_col not in df_merged.columns:
        return None
    
    col_name = f'pd_{model_name.lower().replace(" ", "_").replace("-", "_")}'
    df_merged[col_name] = np.nan
    
    # Filter rows with valid volatility, asset value, and liabilities
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
    # Daily volatility is annualised before applying the Merton formula with T equal to 1 year
    V_A = valid_data['asset_value'].values
    B = valid_data['liabilities_total'].values
    sigma_A_daily = valid_data[vol_col].values
    sigma_A = sigma_A_daily * np.sqrt(252)
    r = valid_data['risk_free_rate'].values
    T = 1.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        d2 = (np.log(V_A / B) + (r - 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
        pd_values = norm.cdf(-d2)
        pd_values = np.clip(pd_values, 0, 1)
    
    df_merged.loc[mask_valid, col_name] = pd_values
    
    
    return df_merged


def run_pd_pipeline(data_garch, data_regime, data_msgarch):
    # Calculates PD for all three models and merges results into a single daily dataset
    
    print("\nProbability of default")

    liabilities_df, rates_df = load_auxiliary_data()
    
    if liabilities_df is None or rates_df is None:
        return pd.DataFrame()
    
    print("\nCalculating PD for each model")
    
    # GARCH
    df_garch = calculate_pd_for_model('GARCH', data_garch, liabilities_df, rates_df)
    if df_garch is None:
        return pd.DataFrame()
    
    # Keep core columns from GARCH results
    final_df = df_garch[['gvkey', 'date', 'asset_value', 'liabilities_total', 
                          'risk_free_rate', 'pd_garch']].copy()
    
    if 'garch_volatility' in df_garch.columns:
        final_df['garch_volatility'] = df_garch['garch_volatility']
    
    # Regime Switching
    df_regime = calculate_pd_for_model('Regime Switching', data_regime, liabilities_df, rates_df)
    if df_regime is not None:
        pd_col_name = 'pd_regime_switching'
        if pd_col_name in df_regime.columns:
            cols_to_merge = df_regime[['gvkey', 'date', pd_col_name]].copy()
            final_df = pd.merge(final_df, cols_to_merge, on=['gvkey', 'date'], how='left')
            print(f"Merged Regime Switching results")
        else:
            print(f"PD column '{pd_col_name}' not found in Regime Switching results")
    else:
        print("Regime Switching model skipped")
    
    # MS GARCH
    df_msgarch = calculate_pd_for_model('MS-GARCH', data_msgarch, liabilities_df, rates_df)
    if df_msgarch is not None:
        pd_col_name = 'pd_ms_garch'
        if pd_col_name in df_msgarch.columns:
            cols_to_merge = df_msgarch[['gvkey', 'date', pd_col_name]].copy()
            final_df = pd.merge(final_df, cols_to_merge, on=['gvkey', 'date'], how='left')
            print(f"Merged MS-GARCH results")
        else:
            available_cols = [col for col in df_msgarch.columns if 'pd_' in col]
    else:
        print("MS-GARCH model skipped")
    
    # Summary
    print(f"\nPD pipeline complete")
    print(f"  Total records: {len(final_df):,}")
    print(f"  Firms: {final_df['gvkey'].nunique()}")
    if len(final_df) > 0:
        print(f"  Date range: {final_df['date'].min().strftime('%Y-%m-%d')} to {final_df['date'].max().strftime('%Y-%m-%d')}")
    
    return final_df


def calculate_merton_pd_normal(daily_returns_file):
    # Calculates Merton PD using daily asset volatility from the Merton estimation
    print("\nCalculating Merton PD")
    
    df = pd.read_csv(daily_returns_file)
    df['date'] = pd.to_datetime(df['date'])
    df['fyear'] = df['date'].dt.year
    
    liabilities_df, rates_df = load_auxiliary_data()
    
    if liabilities_df is None or rates_df is None:
        return pd.DataFrame()
    
    # Merge liabilities and interest rates
    df = pd.merge(df, liabilities_df, on=['gvkey', 'fyear'], how='left')
    df['month_year'] = df['date'].dt.strftime('%Y-%m')
    df = pd.merge(df, rates_df, on='month_year', how='left')
    
    df['risk_free_rate'] = df.groupby('gvkey')['risk_free_rate'].transform(
        lambda x: x.ffill().bfill()
    )
    df['risk_free_rate'] = df['risk_free_rate'].fillna(0.05)
    
    # Drop rows missing required fields and filter to positive values
    df_clean = df.dropna(subset=['asset_value', 'liabilities_total', 'asset_volatility', 'risk_free_rate'])
    df_clean = df_clean[(df_clean['asset_value'] > 0) & (df_clean['liabilities_total'] > 0)]
    
    # Merton PD calculation with daily volatility annualised
    V_A = df_clean['asset_value'].values
    B = df_clean['liabilities_total'].values
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


def get_regime_volatility(df_regime):
    # Returns regime weighted volatility falling back to asset volatility or a 30 percent default
    import os
    from pathlib import Path
    
    try:
        from src.utils import config
        params_file = config.TABLES_DIR / 'regime_switching_parameters.csv'
    except ImportError:
        params_file = Path(__file__).resolve().parent.parent.parent / "data" / "output" / "tables" / "regime_switching_parameters.csv"

    if os.path.exists(params_file):
        params_df = pd.read_csv(params_file)
        
        df_merged = pd.merge(
            df_regime, 
            params_df[['gvkey', 'regime_0_vol', 'regime_1_vol']], 
            on='gvkey', 
            how='left'
        )
        
        # Probability weighted average of regime volatilities
        df_merged['regime_volatility'] = (
            df_merged['regime_probability_0'] * df_merged['regime_0_vol'] +
            df_merged['regime_probability_1'] * df_merged['regime_1_vol']
        )
        
        return df_merged['regime_volatility']
    else:
        if 'asset_volatility' in df_regime.columns:
            return df_regime['asset_volatility']
        else:
            return pd.Series([0.3] * len(df_regime))
