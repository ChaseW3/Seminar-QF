# debug_volatility.py
"""
Debug script to check Monte Carlo volatility values and asset/liability scaling.
Run this to identify the source of large CDS spreads.
"""

import pandas as pd
import numpy as np

print("\n" + "="*80)
print("DEBUG: VOLATILITY AND ASSET/LIABILITY CHECK")
print("="*80 + "\n")

# 1. Check Monte Carlo GARCH results
print("1. GARCH Monte Carlo Results:")
print("-" * 80)

try:
    mc_garch = pd.read_csv('daily_monte_carlo_garch_results.csv')
    print(f"✓ Loaded {len(mc_garch):,} rows")
    print(f"  Columns: {list(mc_garch.columns)}\n")
    
    if 'mc_garch_cumulative_volatility' in mc_garch.columns:
        cumvol = mc_garch['mc_garch_cumulative_volatility']
        print(f"  mc_garch_cumulative_volatility:")
        print(f"    Min:    {cumvol.min():.6f}")
        print(f"    Max:    {cumvol.max():.6f}")
        print(f"    Mean:   {cumvol.mean():.6f}")
        print(f"    Median: {cumvol.median():.6f}")
        print(f"    Std:    {cumvol.std():.6f}")
        
        # What this implies for annualized vol
        mean_daily = cumvol / 252
        annualized = mean_daily * np.sqrt(252)
        print(f"\n  Implied annualized volatility (mean_daily × √252):")
        print(f"    Min:    {annualized.min():.4f} ({annualized.min()*100:.2f}%)")
        print(f"    Max:    {annualized.max():.4f} ({annualized.max()*100:.2f}%)")
        print(f"    Mean:   {annualized.mean():.4f} ({annualized.mean()*100:.2f}%)")
        print(f"    Median: {annualized.median():.4f} ({annualized.median()*100:.2f}%)")
        print(f"\n    ⚠ INTERPRETATION: Sum of daily vols over 252 days = {cumvol.mean():.2f}")
        print(f"                     Average daily vol = {mean_daily.mean():.6f}")
        print(f"                     Annualized = {annualized.mean()*100:.2f}% (reasonable!)")
    
    if 'mc_garch_mean_daily_volatility' in mc_garch.columns:
        meanvol = mc_garch['mc_garch_mean_daily_volatility']
        print(f"\n  mc_garch_mean_daily_volatility:")
        print(f"    Min:    {meanvol.min():.6f}")
        print(f"    Max:    {meanvol.max():.6f}")
        print(f"    Mean:   {meanvol.mean():.6f}")
except FileNotFoundError:
    print("✗ File not found")

# 2. Check daily returns and asset values
print("\n\n2. Daily Asset Returns:")
print("-" * 80)

try:
    daily_ret = pd.read_csv('daily_asset_returns.csv')
    print(f"✓ Loaded {len(daily_ret):,} rows")
    print(f"  Columns: {list(daily_ret.columns)}\n")
    
    print(f"  Asset Values (asset_value):")
    print(f"    Min:    {daily_ret['asset_value'].min():.2e}")
    print(f"    Max:    {daily_ret['asset_value'].max():.2e}")
    print(f"    Mean:   {daily_ret['asset_value'].mean():.2e}")
    print(f"    Median: {daily_ret['asset_value'].median():.2e}")
    
    print(f"\n  Asset Volatility (if exists):")
    if 'asset_volatility' in daily_ret.columns:
        av = daily_ret['asset_volatility'].dropna()
        print(f"    Min:    {av.min():.4f} ({av.min()*100:.2f}%)")
        print(f"    Max:    {av.max():.4f} ({av.max()*100:.2f}%)")
        print(f"    Mean:   {av.mean():.4f} ({av.mean()*100:.2f}%)")
except FileNotFoundError:
    print("✗ File not found")

# 3. Check Merton data (liabilities)
print("\n\n3. Merton Data (Liabilities):")
print("-" * 80)

try:
    merton = pd.read_csv('merged_data_with_merton.csv')
    print(f"✓ Loaded {len(merton):,} rows")
    print(f"  Columns: {list(merton.columns)}\n")
    
    liab = merton['liabilities_total'].dropna()
    print(f"  Liabilities (liabilities_total - in millions):")
    print(f"    Min:    {liab.min():.2e} million")
    print(f"    Max:    {liab.max():.2e} million")
    print(f"    Mean:   {liab.mean():.2e} million")
    print(f"    Median: {liab.median():.2e} million")
    print(f"\n  When converted to actual values (×10^6):")
    print(f"    Min:    {liab.min()*1e6:.2e}")
    print(f"    Max:    {liab.max()*1e6:.2e}")
    print(f"    Mean:   {liab.mean()*1e6:.2e}")
except FileNotFoundError:
    print("✗ File not found")

# 4. Check leverage ratios
print("\n\n4. Leverage Ratios:")
print("-" * 80)

try:
    # Merge asset values and liabilities
    merged_check = pd.merge(
        daily_ret[['gvkey', 'date', 'asset_value']],
        merton[['gvkey', 'date', 'liabilities_total']],
        on=['gvkey', 'date'],
        how='inner'
    )
    
    merged_check['leverage'] = (merged_check['liabilities_total'] * 1e6) / merged_check['asset_value']
    
    print(f"✓ Merged {len(merged_check):,} rows\n")
    
    print(f"  Leverage Ratio (Liabilities/Assets):")
    print(f"    Min:      {merged_check['leverage'].min():.4f}")
    print(f"    Max:      {merged_check['leverage'].max():.4f}")
    print(f"    Mean:     {merged_check['leverage'].mean():.4f}")
    print(f"    Median:   {merged_check['leverage'].median():.4f}")
    
    pct_insolvent = (merged_check['leverage'] > 1).mean() * 100
    print(f"    % > 1 (insolvent): {pct_insolvent:.2f}%")
    
    if pct_insolvent > 5:
        print(f"\n    ⚠ WARNING: {pct_insolvent:.1f}% of firms have leverage > 1!")
        print(f"              This would cause extreme CDS spreads or errors.")
    
    print(f"\n  Leverage Distribution:")
    for pct in [5, 25, 50, 75, 95]:
        print(f"    {pct}th percentile: {merged_check['leverage'].quantile(pct/100):.4f}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# 5. Quick CDS spread check
print("\n\n5. Expected CDS Spreads (Sanity Check):")
print("-" * 80)

try:
    # Load one sample row and calculate
    from scipy.special import ndtr
    
    sample = merged_check.iloc[0]
    V = sample['asset_value']
    K = sample['liabilities_total'] * 1e6
    sigma = 0.3  # 30% equity volatility
    r = 0.05
    tau = 5
    
    sig_sqrt_tau = sigma * np.sqrt(tau)
    d2 = (np.log(V / K) + (r - 0.5 * sigma**2) * tau) / sig_sqrt_tau
    
    P = K * np.exp(-r * tau) * (1 - ndtr(d2)) - V * (1 - ndtr(d2 + sigma*np.sqrt(tau)))
    P_ratio = P / (K * np.exp(-r * tau))
    P_ratio = np.clip(P_ratio, 0, 0.9999)
    
    spread = -(1/tau) * np.log(1 - P_ratio)
    spread_bps = spread * 10000
    
    print(f"  Sample calculation:")
    print(f"    Asset value (V): {V:.2e}")
    print(f"    Liabilities (K): {K:.2e}")
    print(f"    Leverage (K/V): {K/V:.4f}")
    print(f"    Volatility (σ): {sigma:.1%}")
    print(f"    Rate (r): {r:.1%}")
    print(f"    Maturity (τ): {tau}Y")
    print(f"\n    d2: {d2:.4f}")
    print(f"    Put value (P): {P:.2e}")
    print(f"    5Y CDS spread: {spread_bps:.2f} bps ({spread:.2%})")
    
except Exception as e:
    print(f"✗ Error in calculation: {e}")

print("\n" + "="*80)
print("END DEBUG OUTPUT")
print("="*80 + "\n")
