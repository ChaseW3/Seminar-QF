# cds_correlation.py
"""
Compare model-implied CDS spreads with actual market CDS data.

Following Malone et al. (2009) methodology:
1. RMSE of spread prediction errors: e_{i,t}(τ) = S^CDS_{i,t}(τ) - s^{(m)}_{i,t}(τ)
2. Correlation of spread CHANGES (innovations): ρ^{(m)}(τ) = Corr(ΔS^CDS, Δs^{(m)})

References:
- Malone et al. (2009)
- Byström (2006)
- van de Ven et al. (2018)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from src.utils import config
except ImportError:
    from src.utils import config


# Company name mapping between model and CDS data
# NOTE: Financial institutions are now INCLUDED in the analysis
COMPANY_MAPPING = {
    # Model company name -> CDS data company name
    'ADIDAS AG': 'ADIDAS AG',
    'AIRBUS SE': 'AIRBUS SE',
    'ALLIANZ SE': 'ALLIANZ SE',
    'ANHEUSER-BUSCH INBEV': 'ANHEUSER-BUSCH',
    'AXA SA': 'AXA',  # Financial institution - included
    'BASF SE': 'BASF SE',
    'BAYER AG': 'BAYER AG',
    'BAYERISCHE MOTOREN WERKE AKT': 'BAYER MOTOREN WERKE',
    'BNP PARIBAS': 'BNP PARIBAS SA',  # Financial institution - included
    'DANONE SA': 'DANONE SA',
    'DEUTSCHE POST AG': 'DEUTSCHE POST AG',
    'DEUTSCHE TELEKOM': 'DEUTSCHE TELEKOM AG',
    'ENEL SPA': 'ENEL S.P.A.',
    'ENI SPA': 'ENI S.P.A.',
    'IBERDROLA SA': 'IBERDROLA, S.A.',
    'INFINEON TECHNOLOGIES AG': 'INFINEON TECS',
    'ING GROEP NV': 'ING GROEP N.V.',  # Financial institution - included
    'INTESA SANPAOLO SPA': 'INTESA SANPAOLO',  # Financial institution - included
    'KERING SA': 'KERING SA',
    'KONINKLIJKE AHOLD DELHAIZE': 'KON AHOLD DELHAIZE',
    "L'AIR LIQUIDE SA": 'AIR LIQUIDE SA',
    'LOREAL SA': "L'OREAL",
    'LVMH MOET HENNESSY LOUIS V': 'LVMH MOET HENNESSY',
    'MUNICH RE CO': 'MUNICH REINSURANCE',
    'NOKIA OYJ': 'NOKIA OYJ',
    'ORANGE SA': 'ORANGE S.A.',
    'SANOFI': 'SANOFI SA',
    'SAP SE': 'SAP SE',
    'SCHNEIDER ELECTRIC S E': 'SCHNEIDER ELECTRIC',
    'SIEMENS AG': 'SIEMENS AG',  # Parent company - included
    # 'SIEMENS ENERGY AG': excluded - spun off 2020, different company, no market CDS data
    'TOTALENERGIES SE': 'TOTALENERGIES SE',
    'UNICREDIT SPA': 'UNICREDIT SPA',  # Financial institution - included
    'VINCI SA': 'VINCI',
    'WOLTERS KLUWER NV': 'WOLTERS KLUWER NV',
}


def load_cds_market_data(filepath, maturity):
    """
    Load CDS market data from Excel file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the CDS Excel file
    maturity : int
        Maturity in years (1, 3, or 5)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, company_cds, cds_market_{maturity}y_bps
    """
    df = pd.read_excel(filepath, header=None)
    
    # Row 3 has company names, row 4 has tickers, data starts at row 5
    company_names_raw = df.iloc[3, 2:].tolist()
    
    # Extract clean company names
    company_names = []
    for name in company_names_raw:
        if pd.notna(name):
            clean = name.split(' SNR ')[0] if ' SNR ' in str(name) else str(name)
            company_names.append(clean)
        else:
            company_names.append(None)
    
    # Get dates and data
    dates = pd.to_datetime(df.iloc[5:, 0], errors='coerce')
    data = df.iloc[5:, 2:].copy()
    data.columns = company_names[:len(data.columns)]
    data['date'] = dates.values
    
    # Melt to long format
    data_long = data.melt(
        id_vars=['date'], 
        var_name='company_cds', 
        value_name=f'cds_market_{maturity}y_bps'
    )
    data_long[f'cds_market_{maturity}y_bps'] = pd.to_numeric(
        data_long[f'cds_market_{maturity}y_bps'], errors='coerce'
    )
    
    return data_long.dropna(subset=['date', 'company_cds'])


def load_all_market_cds_data(input_dir=None):
    """
    Load all CDS market data (1Y, 3Y, 5Y) and merge.
    
    Parameters
    ----------
    input_dir : Path, optional
        Directory containing CDS Excel files. Defaults to config.INPUT_DIR
    
    Returns
    -------
    pd.DataFrame
        Combined market CDS data
    """
    if input_dir is None:
        input_dir = config.INPUT_DIR
    
    input_dir = Path(input_dir)
    
    print("Loading real CDS market data...")
    cds_1y = load_cds_market_data(input_dir / 'CDS_1y_mat_data.xlsx', 1)
    cds_3y = load_cds_market_data(input_dir / 'CDS_3y_mat_data.xlsx', 3)
    cds_5y = load_cds_market_data(input_dir / 'CDS_5y_mat_data.xlsx', 5)
    
    print(f"  Loaded CDS data: 1Y={len(cds_1y)}, 3Y={len(cds_3y)}, 5Y={len(cds_5y)} rows")
    
    # Merge all maturities
    cds_market = cds_1y.merge(cds_3y, on=['date', 'company_cds'], how='outer')
    cds_market = cds_market.merge(cds_5y, on=['date', 'company_cds'], how='outer')
    
    print(f"  Combined market CDS data: {len(cds_market)} rows")
    
    return cds_market


def calculate_cds_correlations(
    model_cds_file,
    merton_file,
    cds_market_df,
    model_name,
    col_prefix='cds_spread_garch_mc',
    min_observations=30
):
    """
    Merge model CDS with market CDS and calculate metrics following Malone et al. (2009).
    
    Metrics computed:
    1. RMSE: Root Mean Squared Error of spread prediction errors
       e_{i,t}(τ) = S^CDS_{i,t}(τ) - s^{(m)}_{i,t}(τ)
    
    2. Correlation of CHANGES (innovations):
       ρ^{(m)}(τ) = Corr(ΔS^CDS_{i,t}(τ), Δs^{(m)}_{i,t}(τ))
       where ΔS = S_t - S_{t-1}
    
    Parameters
    ----------
    model_cds_file : str or Path
        Path to model CDS spreads CSV
    merton_file : str or Path
        Path to Merton results CSV (for company name mapping)
    cds_market_df : pd.DataFrame
        Market CDS data from load_all_market_cds_data()
    model_name : str
        Name of the model (for display)
    col_prefix : str
        Column prefix for CDS spread columns in model file
    min_observations : int
        Minimum observations required for per-firm metrics
    
    Returns
    -------
    tuple
        (merged_df, metrics_dict, firm_metrics_df)
    """
    # Load company mapping
    merton_df = pd.read_csv(merton_file)
    gvkey_to_company = merton_df[['gvkey', 'company']].drop_duplicates()
    gvkey_to_company = gvkey_to_company.set_index('gvkey')['company'].to_dict()
    
    # Load model CDS
    model_df = pd.read_csv(model_cds_file)
    model_df['company'] = model_df['gvkey'].map(gvkey_to_company)
    model_df['company_cds'] = model_df['company'].map(COMPANY_MAPPING)
    model_df['date'] = pd.to_datetime(model_df['date'])
    
    # Rename model columns
    cols_to_use = ['gvkey', 'date', 'company', 'company_cds']
    rename_map = {}
    for mat in [1, 3, 5]:
        old_col = f'{col_prefix}_{mat}y_bps'
        new_col = f'cds_model_{mat}y_bps'
        if old_col in model_df.columns:
            cols_to_use.append(old_col)
            rename_map[old_col] = new_col
    
    df_model = model_df[cols_to_use].copy()
    df_model = df_model.rename(columns=rename_map)
    
    # Merge with market data
    merged = df_model.merge(cds_market_df, on=['date', 'company_cds'], how='inner')
    merged = merged.sort_values(['gvkey', 'date'])
    
    print(f"\n=== {model_name} Model vs Market CDS ===")
    print(f"  Matched observations: {len(merged)}")
    print(f"  Matched companies: {merged['company'].nunique()}")
    
    # Calculate spread changes (innovations) for each firm
    for mat in [1, 3, 5]:
        model_col = f'cds_model_{mat}y_bps'
        market_col = f'cds_market_{mat}y_bps'
        if model_col in merged.columns and market_col in merged.columns:
            # Calculate first differences (changes) per firm
            merged[f'delta_model_{mat}y'] = merged.groupby('gvkey')[model_col].diff()
            merged[f'delta_market_{mat}y'] = merged.groupby('gvkey')[market_col].diff()
            # Calculate prediction error (market - model)
            merged[f'error_{mat}y'] = merged[market_col] - merged[model_col]
    
    # Calculate overall metrics
    metrics = {'rmse': {}, 'corr_levels': {}, 'corr_changes': {}}
    
    print(f"\n  RMSE (Market - Model, in bps):")
    for mat in [1, 3, 5]:
        error_col = f'error_{mat}y'
        if error_col in merged.columns:
            valid_errors = merged[error_col].dropna()
            if len(valid_errors) > 10:
                rmse = np.sqrt((valid_errors ** 2).mean())
                metrics['rmse'][f'{mat}Y'] = rmse
                print(f"    {mat}Y: {rmse:.2f} bps (n={len(valid_errors)})")
    
    print(f"\n  Correlation of LEVELS:")
    for mat in [1, 3, 5]:
        model_col = f'cds_model_{mat}y_bps'
        market_col = f'cds_market_{mat}y_bps'
        if model_col in merged.columns and market_col in merged.columns:
            valid = merged[[model_col, market_col]].dropna()
            if len(valid) > 10:
                corr = valid[model_col].corr(valid[market_col])
                metrics['corr_levels'][f'{mat}Y'] = corr
                print(f"    {mat}Y: {corr:.4f} (n={len(valid)})")
    
    print(f"\n  Correlation of CHANGES (Innovations) - Byström (2006):")
    for mat in [1, 3, 5]:
        delta_model = f'delta_model_{mat}y'
        delta_market = f'delta_market_{mat}y'
        if delta_model in merged.columns and delta_market in merged.columns:
            valid = merged[[delta_model, delta_market]].dropna()
            if len(valid) > 10:
                corr = valid[delta_model].corr(valid[delta_market])
                metrics['corr_changes'][f'{mat}Y'] = corr
                print(f"    {mat}Y: {corr:.4f} (n={len(valid)})")
    
    # Calculate per-firm metrics
    firm_metrics = []
    for company in merged['company'].unique():
        firm_data = merged[merged['company'] == company].sort_values('date')
        firm_result = {
            'company': company, 
            'gvkey': firm_data['gvkey'].iloc[0], 
            'n_obs': len(firm_data)
        }
        
        for mat in [1, 3, 5]:
            model_col = f'cds_model_{mat}y_bps'
            market_col = f'cds_market_{mat}y_bps'
            delta_model = f'delta_model_{mat}y'
            delta_market = f'delta_market_{mat}y'
            error_col = f'error_{mat}y'
            
            if model_col in firm_data.columns and market_col in firm_data.columns:
                valid_levels = firm_data[[model_col, market_col]].dropna()
                valid_changes = firm_data[[delta_model, delta_market]].dropna()
                valid_errors = firm_data[error_col].dropna()
                
                if len(valid_levels) >= min_observations:
                    # RMSE
                    firm_result[f'rmse_{mat}y'] = np.sqrt((valid_errors ** 2).mean())
                    # Correlation of levels
                    firm_result[f'corr_levels_{mat}y'] = valid_levels[model_col].corr(valid_levels[market_col])
                    # Correlation of changes
                    if len(valid_changes) >= min_observations:
                        firm_result[f'corr_changes_{mat}y'] = valid_changes[delta_model].corr(valid_changes[delta_market])
                    else:
                        firm_result[f'corr_changes_{mat}y'] = np.nan
                else:
                    firm_result[f'rmse_{mat}y'] = np.nan
                    firm_result[f'corr_levels_{mat}y'] = np.nan
                    firm_result[f'corr_changes_{mat}y'] = np.nan
        
        firm_metrics.append(firm_result)
    
    firm_metrics_df = pd.DataFrame(firm_metrics)
    
    return merged, metrics, firm_metrics_df


def run_cds_correlation_analysis(output_dir=None, input_dir=None):
    """
    Run full CDS correlation analysis for all models.
    
    Parameters
    ----------
    output_dir : Path, optional
        Directory containing model output files. Defaults to config.OUTPUT_DIR
    input_dir : Path, optional
        Directory containing CDS market data. Defaults to config.INPUT_DIR
    
    Returns
    -------
    dict
        Dictionary with results for each model
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    if input_dir is None:
        input_dir = config.INPUT_DIR
    
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    
    # Load market CDS data
    cds_market = load_all_market_cds_data(input_dir)
    
    merton_file = output_dir / 'merged_data_with_merton.csv'
    
    results = {}
    
    # Helper function to transform Monte Carlo results to CDS spread format
    def prepare_mc_file(mc_file, spread_col_prefix, output_col_prefix):
        """Read Monte Carlo file and rename spread columns."""
        try:
            df = pd.read_csv(mc_file)
            df['date'] = pd.to_datetime(df['date'])
            
            # Rename columns from Monte Carlo format to expected CDS spread format
            rename_map = {
                f'{spread_col_prefix}_1y': f'{output_col_prefix}_1y',
                f'{spread_col_prefix}_3y': f'{output_col_prefix}_3y',
                f'{spread_col_prefix}_5y': f'{output_col_prefix}_5y',
            }
            
            # Check if all required columns exist
            missing_cols = [col for col in rename_map.keys() if col not in df.columns]
            if missing_cols:
                print(f"⚠ Warning: Missing columns in {mc_file}: {missing_cols}")
                print(f"   Available columns: {df.columns.tolist()}")
                return None
            
            df = df.rename(columns=rename_map)
            
            # Save to temporary file for calculate_cds_correlations
            temp_file = output_dir / f'temp_{output_col_prefix}.csv'
            required_cols = ['date', 'gvkey'] + [f'{output_col_prefix}_1y', f'{output_col_prefix}_3y', f'{output_col_prefix}_5y']
            df[required_cols].to_csv(temp_file, index=False)
            return temp_file
        except Exception as e:
            print(f"⚠ Error preparing {mc_file}: {e}")
            return None
    
    # Merton Monte Carlo (Constant Volatility Baseline)
    merton_mc_file = output_dir / 'daily_monte_carlo_merton_results.csv'
    if merton_mc_file.exists():
        temp_file = prepare_mc_file(merton_mc_file, 'merton_mc_implied_spread', 'cds_spread_merton_mc')
        if temp_file is not None:
            results['Merton_MC'] = calculate_cds_correlations(
                model_cds_file=temp_file,
                merton_file=merton_file,
                cds_market_df=cds_market,
                model_name='Merton MC',
                col_prefix='cds_spread_merton_mc'
            )
    else:
        print(f"⚠ Warning: {merton_mc_file} not found. Run Merton MC simulation first.")
    
    # GARCH
    garch_mc_file = output_dir / 'daily_monte_carlo_garch_results.csv'
    if garch_mc_file.exists():
        temp_file = prepare_mc_file(garch_mc_file, 'mc_garch_implied_spread', 'cds_spread_garch_mc')
        if temp_file is not None:
            results['GARCH'] = calculate_cds_correlations(
                model_cds_file=temp_file,
                merton_file=merton_file,
                cds_market_df=cds_market,
                model_name='GARCH',
                col_prefix='cds_spread_garch_mc'
            )
    else:
        print(f"⚠ Warning: {garch_mc_file} not found.")
    
    # Regime Switching
    rs_mc_file = output_dir / 'daily_monte_carlo_regime_switching_results.csv'
    if rs_mc_file.exists():
        temp_file = prepare_mc_file(rs_mc_file, 'rs_implied_spread', 'cds_spread_regime_switching_mc')
        if temp_file is not None:
            results['RS'] = calculate_cds_correlations(
                model_cds_file=temp_file,
                merton_file=merton_file,
                cds_market_df=cds_market,
                model_name='Regime-Switching',
                col_prefix='cds_spread_regime_switching_mc'
            )
    else:
        print(f"⚠ Warning: {rs_mc_file} not found.")
    
    # MS-GARCH
    msgarch_mc_file = output_dir / 'daily_monte_carlo_ms_garch_results.csv'
    if msgarch_mc_file.exists():
        temp_file = prepare_mc_file(msgarch_mc_file, 'mc_ms_garch_implied_spread', 'cds_spread_msgarch_mc')
        if temp_file is not None:
            results['MSGARCH'] = calculate_cds_correlations(
                model_cds_file=temp_file,
                merton_file=merton_file,
                cds_market_df=cds_market,
                model_name='MS-GARCH',
                col_prefix='cds_spread_msgarch_mc'
            )
    else:
        print(f"⚠ Warning: {msgarch_mc_file} not found.")
    
    # Create summary table
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY (5-Year Maturity)")
    print("="*80)
    
    # Check if we have any results
    if not results:
        print("⚠ Error: No models were successfully loaded!")
        return None
    
    # Build summary DataFrame with all metrics from first available model
    first_model = list(results.keys())[0]
    summary_df = results[first_model][2][['company', 'gvkey', 'n_obs']].copy()
    
    # Include all available models
    model_list = []
    for model_key in ['Merton_MC', 'GARCH', 'RS', 'MSGARCH']:
        if model_key in results:
            model_list.append((model_key, model_key))
    
    for model_key, model_short in model_list:
        if model_key not in results:
            continue
        firm_df = results[model_key][2]
        
        # Check if required columns exist
        required_cols = ['company', 'rmse_5y', 'corr_levels_5y', 'corr_changes_5y']
        missing_cols = [col for col in required_cols if col not in firm_df.columns]
        if missing_cols:
            print(f"⚠️  Warning: Skipping {model_key} - missing columns: {missing_cols}")
            continue
        
        summary_df = summary_df.merge(
            firm_df[required_cols].rename(columns={
                'rmse_5y': f'{model_short}_rmse',
                'corr_levels_5y': f'{model_short}_corr_lvl',
                'corr_changes_5y': f'{model_short}_corr_chg'
            }),
            on='company', how='left'
        )
    
    summary_df = summary_df.sort_values('GARCH_rmse', ascending=True)
    
    print("\nFirm-Level Metrics (5Y Maturity):")
    print("  rmse = RMSE in bps, corr_lvl = correlation of levels, corr_chg = correlation of changes")
    print(summary_df.to_string(index=False))
    
    # Overall summary statistics
    print("\n" + "="*80)
    print("OVERALL SUMMARY STATISTICS (5-Year Maturity)")
    print("="*80)
    
    # Include Merton_MC if available
    model_display_list = [('GARCH', 'GARCH'), ('RS', 'Regime-Switching'), ('MSGARCH', 'MS-GARCH')]
    if 'Merton_MC' in results:
        model_display_list = [('Merton_MC', 'Merton MC')] + model_display_list
    
    for model_key, model_name in model_display_list:
        if model_key not in results:
            continue
        metrics = results[model_key][1]
        firm_df = results[model_key][2]
        
        print(f"\n{model_name}:")
        
        # RMSE
        if '5Y' in metrics.get('rmse', {}):
            print(f"  Overall RMSE:           {metrics['rmse']['5Y']:.2f} bps")
        if 'rmse_5y' in firm_df.columns:
            rmse_col = firm_df['rmse_5y'].dropna()
            if len(rmse_col) > 0:
                print(f"  Mean Firm RMSE:         {rmse_col.mean():.2f} bps")
                print(f"  Median Firm RMSE:       {rmse_col.median():.2f} bps")
        
        # Correlation of levels
        if '5Y' in metrics.get('corr_levels', {}):
            print(f"  Overall Corr (levels):  {metrics['corr_levels']['5Y']:.4f}")
        if 'corr_levels_5y' in firm_df.columns:
            corr_lvl = firm_df['corr_levels_5y'].dropna()
            if len(corr_lvl) > 0:
                print(f"  Mean Firm Corr (lvl):   {corr_lvl.mean():.4f}")
        
        # Correlation of changes (the key metric from Byström 2006)
        if '5Y' in metrics.get('corr_changes', {}):
            print(f"  Overall Corr (changes): {metrics['corr_changes']['5Y']:.4f}")
        if 'corr_changes_5y' in firm_df.columns:
            corr_chg = firm_df['corr_changes_5y'].dropna()
            if len(corr_chg) > 0:
                print(f"  Mean Firm Corr (chg):   {corr_chg.mean():.4f}")
        
        if 'rmse_5y' in firm_df.columns:
            print(f"  Firms with data:        {len(firm_df.dropna(subset=['rmse_5y']))}")
    
    # Save summary
    summary_file = output_dir / 'cds_model_vs_market_correlations.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved correlation summary to {summary_file}")
    
    results['summary'] = summary_df
    results['cds_market'] = cds_market
    
    return results


def plot_cds_correlations(results, output_dir=None, maturity=5, axis_limit=None):
    """
    Create scatter plots of model vs market CDS spreads.
    
    Parameters
    ----------
    results : dict
        Results from run_cds_correlation_analysis()
    output_dir : Path, optional
        Directory to save plot. Defaults to config.OUTPUT_DIR
    maturity : int
        Maturity to plot (1, 3, or 5)
    axis_limit : float, optional
        Maximum value for both axes (to zoom in and exclude extreme outliers).
        If None, uses data maximum. Recommended: 500-1000 for better visualization.
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    output_dir = Path(output_dir)
    
    # Determine which models are available
    available_models = []
    if 'Merton_MC' in results:
        available_models.append(('Merton_MC', 'Merton MC'))
    available_models.extend([
        ('GARCH', 'GARCH'),
        ('RS', 'Regime-Switching'),
        ('MSGARCH', 'MS-GARCH')
    ])
    
    # Filter to only models that are actually in results
    available_models = [(k, n) for k, n in available_models if k in results]
    n_models = len(available_models)
    
    # Create subplot grid (2x2 if 4 models, 1x3 if 3 models)
    if n_models == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
    
    for ax, (model_key, model_name) in zip(axes, available_models):
        merged_df = results[model_key][0]
        model_col = f'cds_model_{maturity}y_bps'
        market_col = f'cds_market_{maturity}y_bps'
        
        valid = merged_df[[model_col, market_col]].dropna()
        
        if len(valid) > 0:
            # Filter data if axis_limit is specified (for better visualization)
            if axis_limit is not None:
                valid_filtered = valid[(valid[market_col] <= axis_limit) & (valid[model_col] <= axis_limit)]
                n_excluded = len(valid) - len(valid_filtered)
                label_suffix = f' ({n_excluded} outliers excluded)' if n_excluded > 0 else ''
            else:
                valid_filtered = valid
                label_suffix = ''
            
            ax.scatter(valid_filtered[market_col], valid_filtered[model_col], alpha=0.3, s=5)
            
            # Add 45-degree line
            if axis_limit is not None:
                max_val = axis_limit
            else:
                max_val = max(valid[market_col].max(), valid[model_col].max())
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1, label='Perfect correlation')
            
            # Calculate correlation (using ALL data, not filtered)
            corr = valid[model_col].corr(valid[market_col])
            ax.set_title(f'{model_name}\nCorrelation: {corr:.4f}{label_suffix}', fontsize=10)
            ax.set_xlabel('Market CDS Spread (bps)')
            ax.set_ylabel('Model CDS Spread (bps)')
            ax.legend(fontsize=8)
            
            # Set axis limits
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Model-Implied vs Market CDS Spreads ({maturity}-Year Maturity)', fontsize=14)
    plt.tight_layout()
    
    plot_file = output_dir / f'cds_model_vs_market_scatter_{maturity}y.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Saved scatter plot to {plot_file}")
