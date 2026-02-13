# volatility_diagnostics.py
"""
Volatility Diagnostics Module
=============================
Identifies firms with extreme volatility estimates that cause unrealistic CDS spreads.

This module provides detailed firm-level analysis to understand WHY certain firms
have extreme volatility rather than just capping values.

Key diagnostics:
1. GARCH estimation quality (parameter stationarity, convergence)
2. Underlying return distributions (fat tails, outliers)
3. Volatility time series analysis
4. Firm-level summary statistics
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Thresholds for flagging problematic firms
VOLATILITY_THRESHOLDS = {
    'annualized_vol_max': 1.0,      # 100% annual volatility is extreme
    'annualized_vol_min': 0.01,     # 1% annual volatility is too low
    'daily_vol_max': 0.10,          # 10% daily volatility is extreme
    'garch_alpha_beta_sum_max': 0.999,  # Near-IGARCH is problematic
    'return_outlier_threshold': 0.20,   # 20% daily return is extreme
}


def run_volatility_diagnostics(garch_file, mc_garch_file=None, output_dir='./diagnostics/'):
    """
    Run comprehensive volatility diagnostics to identify problematic firms.
    
    Parameters:
    -----------
    garch_file : str
        Path to daily_asset_returns_with_garch.csv
    mc_garch_file : str, optional
        Path to daily_monte_carlo_garch_results.csv
    output_dir : str
        Directory to save diagnostic reports
        
    Returns:
    --------
    dict with:
        - problematic_firms: list of gvkeys with issues
        - firm_diagnostics: DataFrame with per-firm statistics
        - issue_summary: DataFrame summarizing issues by type
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("VOLATILITY DIAGNOSTICS: IDENTIFYING PROBLEMATIC FIRMS")
    print("="*80 + "\n")
    
    # Load GARCH data
    df_garch = pd.read_csv(garch_file)
    df_garch['date'] = pd.to_datetime(df_garch['date'])
    
    print(f"✓ Loaded GARCH data: {len(df_garch):,} observations")
    
    firms = df_garch['gvkey'].unique()
    print(f"✓ Number of firms: {len(firms)}")
    
    # =========================================================================
    # PART 1: GARCH Parameter Analysis
    # =========================================================================
    print("\n" + "-"*60)
    print("PART 1: GARCH PARAMETER ANALYSIS")
    print("-"*60 + "\n")
    
    firm_stats = []
    
    for gvkey in firms:
        df_firm = df_garch[df_garch['gvkey'] == gvkey].copy()
        
        # Basic statistics
        n_obs = len(df_firm)
        
        # Return statistics
        returns = df_firm['asset_return_daily'].dropna()
        if len(returns) == 0:
            continue
            
        return_mean = returns.mean()
        return_std = returns.std()
        return_min = returns.min()
        return_max = returns.max()
        return_skew = returns.skew()
        return_kurt = returns.kurtosis()
        
        # Count extreme returns (outliers)
        n_extreme_returns = (np.abs(returns) > VOLATILITY_THRESHOLDS['return_outlier_threshold']).sum()
        pct_extreme_returns = 100 * n_extreme_returns / len(returns)
        
        # GARCH parameters (take median across dates - should be constant per firm)
        garch_omega = df_firm['garch_omega'].median()
        garch_alpha = df_firm['garch_alpha'].median()
        garch_beta = df_firm['garch_beta'].median()
        garch_persistence = garch_alpha + garch_beta if pd.notna(garch_alpha) and pd.notna(garch_beta) else np.nan
        
        # GARCH volatility statistics
        garch_vol = df_firm['garch_volatility'].dropna()
        if len(garch_vol) > 0:
            garch_vol_mean = garch_vol.mean()
            garch_vol_std = garch_vol.std()
            garch_vol_min = garch_vol.min()
            garch_vol_max = garch_vol.max()
            garch_vol_median = garch_vol.median()
            
            # Annualized volatility (assuming daily vol is σ_daily)
            annualized_vol_mean = garch_vol_mean * np.sqrt(252)
            annualized_vol_max = garch_vol_max * np.sqrt(252)
        else:
            garch_vol_mean = garch_vol_std = garch_vol_min = garch_vol_max = garch_vol_median = np.nan
            annualized_vol_mean = annualized_vol_max = np.nan
        
        # Identify issues
        issues = []
        
        # Issue 1: Extreme annualized volatility
        if annualized_vol_max > VOLATILITY_THRESHOLDS['annualized_vol_max']:
            issues.append(f"HIGH_VOL (max {annualized_vol_max*100:.1f}%)")
        
        if annualized_vol_mean < VOLATILITY_THRESHOLDS['annualized_vol_min']:
            issues.append(f"LOW_VOL (mean {annualized_vol_mean*100:.1f}%)")
        
        # Issue 2: GARCH near unit root (persistence close to 1)
        if pd.notna(garch_persistence) and garch_persistence > VOLATILITY_THRESHOLDS['garch_alpha_beta_sum_max']:
            issues.append(f"NEAR_IGARCH (α+β={garch_persistence:.4f})")
        
        # Issue 3: Extreme daily returns
        if pct_extreme_returns > 1.0:  # More than 1% extreme returns
            issues.append(f"EXTREME_RETURNS ({n_extreme_returns} obs, {pct_extreme_returns:.2f}%)")
        
        # Issue 4: Very high kurtosis (fat tails)
        if return_kurt > 10:
            issues.append(f"FAT_TAILS (kurtosis={return_kurt:.1f})")
        
        # Issue 5: Missing GARCH parameters
        if pd.isna(garch_alpha) or pd.isna(garch_beta):
            issues.append("GARCH_FAILED")
        
        firm_stats.append({
            'gvkey': gvkey,
            'n_observations': n_obs,
            'return_mean': return_mean,
            'return_std': return_std,
            'return_min': return_min,
            'return_max': return_max,
            'return_skew': return_skew,
            'return_kurtosis': return_kurt,
            'n_extreme_returns': n_extreme_returns,
            'pct_extreme_returns': pct_extreme_returns,
            'garch_omega': garch_omega,
            'garch_alpha': garch_alpha,
            'garch_beta': garch_beta,
            'garch_persistence': garch_persistence,
            'garch_vol_mean_daily': garch_vol_mean,
            'garch_vol_max_daily': garch_vol_max,
            'garch_vol_median_daily': garch_vol_median,
            'annualized_vol_mean': annualized_vol_mean,
            'annualized_vol_max': annualized_vol_max,
            'issues': '; '.join(issues) if issues else 'NONE',
            'n_issues': len(issues),
            'is_problematic': len(issues) > 0
        })
    
    df_firm_stats = pd.DataFrame(firm_stats)
    
    # =========================================================================
    # PART 2: Monte Carlo Diagnostics (if available)
    # =========================================================================
    if mc_garch_file and os.path.exists(mc_garch_file):
        print("\n" + "-"*60)
        print("PART 2: MONTE CARLO VOLATILITY ANALYSIS")
        print("-"*60 + "\n")
        
        df_mc = pd.read_csv(mc_garch_file)
        df_mc['date'] = pd.to_datetime(df_mc['date'])
        
        print(f"✓ Loaded MC data: {len(df_mc):,} observations")
        
        # Calculate MC volatility statistics per firm (Using Integrated Variance)
        # Note: If reusing old files without integrated variance, fallback to cumulative
        # If no volatility columns exist, skip MC volatility diagnostics
        
        target_col = None
        is_variance = False
        is_daily_mean = False
        
        if 'mc_garch_integrated_variance' in df_mc.columns:
            target_col = 'mc_garch_integrated_variance'
            is_variance = True
            is_daily_mean = False
        elif 'mc_garch_mean_daily_volatility' in df_mc.columns:
            target_col = 'mc_garch_mean_daily_volatility'
            is_variance = False
            is_daily_mean = True
        elif 'mc_garch_cumulative_volatility' in df_mc.columns:
            target_col = 'mc_garch_cumulative_volatility'
            is_variance = False
            is_daily_mean = False
        
        if target_col is None:
            print("⚠ Warning: No volatility columns found in MC results. Skipping MC volatility diagnostics.")
            print("   (This is normal if using optimized MC that only outputs PD/spreads)")
            mc_stats = pd.DataFrame({'gvkey': df_mc['gvkey'].unique()})
        else:
            mc_stats = df_mc.groupby('gvkey').agg({
                target_col: ['mean', 'std', 'min', 'max', 'median']
            }).reset_index()
            
            mc_stats.columns = ['gvkey', 'mc_raw_mean', 'mc_raw_std', 
                               'mc_raw_min', 'mc_raw_max', 'mc_raw_median']
            
            if is_variance:
                # Annualized from Integrated Variance: σ_annual = √IV
                mc_stats['mc_annualized_vol_mean'] = np.sqrt(mc_stats['mc_raw_mean'])
                mc_stats['mc_annualized_vol_max'] = np.sqrt(mc_stats['mc_raw_max'])
            elif is_daily_mean:
                 # Annualized from Mean Daily Volatility: σ_annual = σ_daily * √252
                 mc_stats['mc_annualized_vol_mean'] = mc_stats['mc_raw_mean'] * np.sqrt(252)
                 mc_stats['mc_annualized_vol_max'] = mc_stats['mc_raw_max'] * np.sqrt(252)
            else:
                # Annualized from cumulative: σ_annual = (cumulative / 252) * √252 = cumulative / √252
                mc_stats['mc_annualized_vol_mean'] = mc_stats['mc_raw_mean'] / np.sqrt(252)
                mc_stats['mc_annualized_vol_max'] = mc_stats['mc_raw_max'] / np.sqrt(252)
        
        # Merge with firm stats
        df_firm_stats = df_firm_stats.merge(mc_stats, on='gvkey', how='left')
        
        # Update issues based on MC volatility
        for idx, row in df_firm_stats.iterrows():
            if pd.notna(row.get('mc_annualized_vol_max')) and row['mc_annualized_vol_max'] > 1.0:
                current_issues = row['issues']
                if current_issues == 'NONE':
                    df_firm_stats.at[idx, 'issues'] = f"MC_HIGH_VOL ({row['mc_annualized_vol_max']*100:.1f}%)"
                else:
                    df_firm_stats.at[idx, 'issues'] = current_issues + f"; MC_HIGH_VOL ({row['mc_annualized_vol_max']*100:.1f}%)"
                df_firm_stats.at[idx, 'n_issues'] = row['n_issues'] + 1
                df_firm_stats.at[idx, 'is_problematic'] = True
    
    # =========================================================================
    # PART 3: Summary Report
    # =========================================================================
    print("\n" + "-"*60)
    print("PART 3: DIAGNOSTIC SUMMARY")
    print("-"*60 + "\n")
    
    # Problematic firms
    problematic_firms = df_firm_stats[df_firm_stats['is_problematic']]['gvkey'].tolist()
    clean_firms = df_firm_stats[~df_firm_stats['is_problematic']]['gvkey'].tolist()
    
    print(f"Total firms analyzed: {len(df_firm_stats)}")
    print(f"Problematic firms: {len(problematic_firms)}")
    print(f"Clean firms: {len(clean_firms)}")
    print(f"Problematic rate: {100*len(problematic_firms)/len(df_firm_stats):.1f}%\n")
    
    # Sort by number of issues and severity
    df_problematic = df_firm_stats[df_firm_stats['is_problematic']].sort_values(
        ['n_issues', 'annualized_vol_max'], ascending=[False, False]
    )
    
    if len(df_problematic) > 0:
        print("PROBLEMATIC FIRMS (sorted by severity):\n")
        print("-" * 100)
        print(f"{'gvkey':>10} | {'Ann.Vol Max':>12} | {'GARCH α+β':>10} | {'Extreme Ret':>12} | Issues")
        print("-" * 100)
        
        for _, row in df_problematic.iterrows():
            ann_vol = f"{row['annualized_vol_max']*100:.1f}%" if pd.notna(row['annualized_vol_max']) else 'N/A'
            persistence = f"{row['garch_persistence']:.4f}" if pd.notna(row['garch_persistence']) else 'N/A'
            extreme = f"{row['n_extreme_returns']:.0f} ({row['pct_extreme_returns']:.1f}%)"
            print(f"{row['gvkey']:>10} | {ann_vol:>12} | {persistence:>10} | {extreme:>12} | {row['issues']}")
        
        print("-" * 100)
    
    # Issue type breakdown
    print("\n\nISSUE TYPE BREAKDOWN:")
    print("-" * 40)
    
    issue_counts = {}
    for issues_str in df_firm_stats['issues']:
        if issues_str != 'NONE':
            for issue in issues_str.split('; '):
                # Extract issue type (before the parenthesis)
                issue_type = issue.split(' (')[0]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
    
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue_type}: {count} firms")
    
    # =========================================================================
    # PART 4: Save Reports
    # =========================================================================
    print("\n" + "-"*60)
    print("PART 4: SAVING DIAGNOSTIC REPORTS")
    print("-"*60 + "\n")
    
    # Save full diagnostics
    diagnostics_file = os.path.join(output_dir, 'firm_volatility_diagnostics.csv')
    df_firm_stats.to_csv(diagnostics_file, index=False)
    print(f"✓ Saved: {diagnostics_file}")
    
    # Save problematic firms list
    problematic_file = os.path.join(output_dir, 'problematic_firms.csv')
    df_problematic.to_csv(problematic_file, index=False)
    print(f"✓ Saved: {problematic_file}")
    
    # Save clean firms list (for filtering)
    clean_firms_df = df_firm_stats[~df_firm_stats['is_problematic']][['gvkey']].copy()
    clean_file = os.path.join(output_dir, 'clean_firms.csv')
    clean_firms_df.to_csv(clean_file, index=False)
    print(f"✓ Saved: {clean_file}")
    
    # Save summary report
    summary_file = os.path.join(output_dir, 'diagnostics_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("VOLATILITY DIAGNOSTICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total firms: {len(df_firm_stats)}\n")
        f.write(f"Problematic firms: {len(problematic_firms)}\n")
        f.write(f"Clean firms: {len(clean_firms)}\n\n")
        f.write("PROBLEMATIC FIRM GVKEYS:\n")
        f.write(str(problematic_firms) + "\n\n")
        f.write("ISSUE BREAKDOWN:\n")
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            f.write(f"  {issue_type}: {count}\n")
    print(f"✓ Saved: {summary_file}")
    
    print("\n" + "="*80)
    print("DIAGNOSTICS COMPLETE")
    print("="*80 + "\n")
    
    return {
        'problematic_firms': problematic_firms,
        'clean_firms': clean_firms,
        'firm_diagnostics': df_firm_stats,
        'n_problematic': len(problematic_firms),
        'n_clean': len(clean_firms)
    }


def filter_problematic_firms(df, problematic_firms, gvkey_column='gvkey'):
    """
    Filter out problematic firms from a DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Data to filter
    problematic_firms : list
        List of gvkeys to remove
    gvkey_column : str
        Name of the gvkey column
        
    Returns:
    --------
    DataFrame with problematic firms removed
    """
    initial_rows = len(df)
    initial_firms = df[gvkey_column].nunique()
    
    df_filtered = df[~df[gvkey_column].isin(problematic_firms)].copy()
    
    final_rows = len(df_filtered)
    final_firms = df_filtered[gvkey_column].nunique()
    
    print(f"Filtered problematic firms: {initial_firms} → {final_firms} firms ({initial_rows:,} → {final_rows:,} rows)")
    
    return df_filtered


def get_problematic_firms_from_mc(mc_file, vol_threshold=1.0):
    """
    Quick function to identify firms with extreme MC volatility.
    
    Parameters:
    -----------
    mc_file : str
        Path to Monte Carlo results file
    vol_threshold : float
        Maximum annualized volatility threshold (default 100%)
        
    Returns:
    --------
    list of problematic gvkeys
    """
    df_mc = pd.read_csv(mc_file)
    
    # Annualized volatility = cumulative / sqrt(252)
    df_mc['annualized_vol'] = df_mc['mc_garch_cumulative_volatility'] / np.sqrt(252)
    
    # Find firms with ANY observation exceeding threshold
    problematic = df_mc[df_mc['annualized_vol'] > vol_threshold]['gvkey'].unique().tolist()
    
    print(f"Firms with annualized volatility > {vol_threshold*100:.0f}%: {len(problematic)}")
    
    return problematic


if __name__ == "__main__":
    # Run diagnostics standalone
    results = run_volatility_diagnostics(
        garch_file='daily_asset_returns_with_garch.csv',
        mc_garch_file='daily_monte_carlo_garch_results.csv',
        output_dir='./diagnostics/'
    )
    
    print(f"\nProblematic firms to investigate: {results['problematic_firms']}")
