# probability_of_default.py
# Calculate Probability of Default using Merton Model with GARCH, Regime Switching and MS-GARCH Volatility

import pandas as pd
import numpy as np
from scipy.stats import norm

def load_auxiliary_data():
    """
    Load liabilities and interest rates (common for all models).
    """
    print("Loading auxiliary data (liabilities and interest rates)...")
    
    # 1. Load merged data for Liabilities
    merged_df = pd.read_csv('merged_data_with_merton.csv')
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['month_year'] = merged_df['date'].dt.strftime('%Y-%m')
    
    # Extract monthly liabilities (use the last available value per month for each firm)
    monthly_liabilities = (merged_df.dropna(subset=['liabilities_total'])
                          .sort_values(['gvkey', 'date'])
                          .groupby(['gvkey', 'month_year'], as_index=False)
                          .apply(lambda x: x.iloc[-1])
                          [['gvkey', 'month_year', 'liabilities_total']]
                          .reset_index(drop=True))
    
    # Correct Unit Mismatch: Data is in Millions, Asset Value is in Ones
    monthly_liabilities['liabilities_total'] = monthly_liabilities['liabilities_total'] * 1_000_000

    # 2. Load Interest Rates from ECB data
    rates_df = pd.read_csv('ECB Data Portal_20260125170805.csv')
    rate_cols = [col for col in rates_df.columns if 'EURIBOR' in col.upper()]
    rates_df['DATE'] = pd.to_datetime(rates_df['DATE'])
    rates_df['month_year'] = rates_df['DATE'].dt.strftime('%Y-%m')
    rates_df['risk_free_rate'] = pd.to_numeric(rates_df[rate_cols[0]], errors='coerce') / 100
    rates_df = rates_df[['month_year', 'risk_free_rate']].drop_duplicates()
    
    return monthly_liabilities, rates_df


def calculate_pd_for_model(model_name, file_path, liabilities_df, rates_df):
    """
    Load model results, merge with aux data, and calculate PD.
    """
    print(f"Processing model: {model_name}...")
    
    # Load model results
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"  Warning: File {file_path} not found. Skipping {model_name}.")
        return None
        
    # Merge with liabilities
    df = df.merge(liabilities_df, on=['gvkey', 'month_year'], how='left')
    
    # Merge with interest rates
    df = df.merge(rates_df, on='month_year', how='left')
    
    # Determine Volatility based on model
    if model_name == 'GARCH':
        # GARCH output has 'garch_volatility'
        if 'garch_volatility' in df.columns:
            df['model_volatility'] = df['garch_volatility']
        elif 'asset_volatility' in df.columns: # fallback if needed
             df['model_volatility'] = df['asset_volatility']
        else:
            print(f"  Warning: 'garch_volatility' not found in {file_path}")
            return None
            
    elif model_name == 'Regime Switching':
        # RS output has probs and sigma2s
        if 'regime_0_prob' in df.columns and 'sigma2_0' in df.columns:
            # Expected Variance = p0*s0 + p1*s1
            # Volatility = sqrt(Expected Variance)
            # Ensure columns are numeric
            for col in ['regime_0_prob', 'sigma2_0', 'regime_1_prob', 'sigma2_1']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            expected_variance = (df['regime_0_prob'] * df['sigma2_0'] + 
                                 df['regime_1_prob'] * df['sigma2_1'])
            # Convert Monthly Variance to Annualized Volatility
            # sigma_annual = sqrt(sigma2_monthly) * sqrt(12)
            df['model_volatility'] = np.sqrt(expected_variance) * np.sqrt(12)
        else:
            print(f"  Warning: RS params not found in {file_path}")
            return None
            
    elif model_name == 'MS-GARCH':
        # MS-GARCH output has 'msgarch_cond_vol'
        if 'msgarch_cond_vol' in df.columns:
            df['model_volatility'] = df['msgarch_cond_vol']
        else:
            print(f"  Warning: 'msgarch_cond_vol' not found in {file_path}")
            return None
    
    # Calculate PD (Merton Model)
    # Time horizon (1 year = 1.0)
    time_horizon = 1.0
    
    # Filter valid rows
    df = df.dropna(subset=['liabilities_total', 'risk_free_rate', 'model_volatility'])
    
    df['log_asset_debt_ratio'] = np.log(df['asset_value'] / df['liabilities_total'])
    
    vol = df['model_volatility']
    rf = df['risk_free_rate']
    
    numerator = (df['log_asset_debt_ratio'] + (rf - 0.5 * vol**2) * time_horizon)
    denominator = vol * np.sqrt(time_horizon)
    
    df['d2'] = numerator / denominator
    df['merton_pd'] = norm.cdf(-df['d2'])
    df['merton_pd'] = df['merton_pd'].clip(0, 1)
    
    return df


def export_pd_results(pd_df, output_filename='monthly_pd_results.csv'):
    """
    Export PD calculations to CSV.
    """
    # Simply export the whole dataframe as it's already structured correctly
    pd_df.to_csv(output_filename, index=False)
    
    print(f"Saved '{output_filename}'")
    print(f"  - Total Records: {len(pd_df)}")
    print(f"  - Columns: {list(pd_df.columns)}")


def run_pd_pipeline(data_garch, data_regime, data_msgarch):
    """
    Complete pipeline: load all 3 model results, calculate PD for each, and combine.
    Results are merged horizontally to allow side-by-side comparison.
    """
    
    print("\n" + "="*60)
    print("STEP 6: PROBABILITY OF DEFAULT CALCULATION (Multi-Model)")
    print("="*60)
    
    # 1. Load Auxiliary Data
    try:
        liabilities_df, rates_df = load_auxiliary_data()
    except Exception as e:
        print(f"Error loading auxiliary data: {e}")
        return pd.DataFrame() # Return empty df on failure
    
    # 2. Process Each Model
    
    # GARCH (Base DataFrame)
    df_garch = calculate_pd_for_model('GARCH', data_garch, liabilities_df, rates_df)
    if df_garch is None:
        print("Critical Error: GARCH model processing failed. Aborting.")
        return pd.DataFrame()
        
    # Rename GARCH specific columns
    final_df = df_garch.rename(columns={
        'merton_pd': 'merton_pd_garch',
        'd2': 'd2_garch',
        'model_volatility': 'volatility_garch'
    })
    
    # Keep only necessary columns + specific ones
    base_cols = ['gvkey', 'month_year', 'asset_value', 'liabilities_total', 'risk_free_rate', 'log_asset_debt_ratio']
    specific_cols_garch = ['merton_pd_garch', 'd2_garch', 'volatility_garch']
    
    # Filter columns if they exist (safe select)
    final_df = final_df[[c for c in base_cols + specific_cols_garch if c in final_df.columns]]
    
    # Regime Switching
    df_regime = calculate_pd_for_model('Regime Switching', data_regime, liabilities_df, rates_df)
    if df_regime is not None:
        # Select relevant cols
        cols_to_merge = df_regime[['gvkey', 'month_year', 'merton_pd', 'd2', 'model_volatility']]
        cols_to_merge = cols_to_merge.rename(columns={
            'merton_pd': 'merton_pd_regime',
            'd2': 'd2_regime',
            'model_volatility': 'volatility_regime'
        })
        # Merge
        final_df = final_df.merge(cols_to_merge, on=['gvkey', 'month_year'], how='outer')
        
    # MS-GARCH
    df_msgarch = calculate_pd_for_model('MS-GARCH', data_msgarch, liabilities_df, rates_df)
    if df_msgarch is not None:
        # Select relevant cols
        cols_to_merge = df_msgarch[['gvkey', 'month_year', 'merton_pd', 'd2', 'model_volatility']]
        cols_to_merge = cols_to_merge.rename(columns={
            'merton_pd': 'merton_pd_msgarch',
            'd2': 'd2_msgarch',
            'model_volatility': 'volatility_msgarch'
        })
        # Merge
        final_df = final_df.merge(cols_to_merge, on=['gvkey', 'month_year'], how='outer')
        
    # 3. Export
    if not final_df.empty:
        export_pd_results(final_df, 'monthly_pd_results.csv')
        
        print("\n✓ PD pipeline complete!")
        return final_df
    else:
        print("No results generated.")
        return pd.DataFrame()
