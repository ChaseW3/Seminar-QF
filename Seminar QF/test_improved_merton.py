"""
Test script for improved Merton solver on highly leveraged firms.
"""

import pandas as pd
import numpy as np
from src.data.data_processing import load_and_preprocess_data, load_interest_rates, run_merton_estimation

print("="*80)
print("TESTING IMPROVED MERTON SOLVER")
print("="*80)
print("\nThis will re-run Merton estimation with:")
print("  1. Better initialization for high-leverage firms")
print("  2. Constraints to prevent unrealistic jumps")
print("  3. More iterations (200 vs 100)")
print("  4. Damping and oscillation detection")
print("  5. Tighter convergence tolerance (1e-5 vs 1e-4)")
print()

# Load data
print("Loading and preprocessing data...")
df = load_and_preprocess_data()
interest_rates_df = load_interest_rates()

print(f"Loaded {len(df):,} observations for {df['gvkey'].nunique()} firms")
print()

# Re-run Merton with improved solver (force recomputation by disabling cache)
print("Running improved Merton estimation...")
print("(This will take a few minutes...)")
print()

df_merged, daily_returns = run_merton_estimation(
    df, 
    interest_rates_df, 
    n_jobs=-1, 
    use_cache=False  # Force recomputation
)

print()
print("="*80)
print("VERIFICATION: Checking previously problematic firms")
print("="*80)
print()

# Check the 5 problematic firms around their issue dates
problem_cases = [
    (15549, '2012-01-09'),  # UniCredit - the worst case
    (16348, '2011-09-15'),  # Intesa
    (15532, '2013-04-03'),  # BNP
    (15617, '2012-11-30'),  # ING
    (63120, '2011-10-21'),  # AXA
]

for gvkey, problem_date in problem_cases:
    firm_data = daily_returns[daily_returns['gvkey'] == gvkey].copy()
    firm_data = firm_data.sort_values('date')
    
    # Get data around the problem date
    problem_dt = pd.to_datetime(problem_date)
    window = firm_data[(firm_data['date'] >= problem_dt - pd.Timedelta(days=3)) & 
                       (firm_data['date'] <= problem_dt + pd.Timedelta(days=3))]
    
    if len(window) > 0:
        print(f"Firm {gvkey} around {problem_date}:")
        print(f"  Asset values: {window['asset_value'].min():.2e} to {window['asset_value'].max():.2e}")
        print(f"  Volatilities: {window['asset_volatility'].min():.4f} to {window['asset_volatility'].max():.4f}")
        
        # Calculate returns
        returns = np.diff(np.log(window['asset_value'].values))
        if len(returns) > 0:
            max_return = np.max(np.abs(returns))
            print(f"  Max |return|: {max_return:.2%}")
            if max_return > 1.0:
                print(f"    ❌ STILL HAS EXTREME RETURN!")
            else:
                print(f"    ✅ Returns look reasonable")
        print()

print("="*80)
print("Saving results...")
daily_returns.to_csv('data/output/daily_asset_returns.csv', index=False)
df_merged.to_csv('data/output/merged_data_with_merton.csv', index=False)
print("✓ Saved to:")
print("  - data/output/daily_asset_returns.csv")
print("  - data/output/merged_data_with_merton.csv")
print()
print("Next step: Re-run regime switching estimation and Monte Carlo simulations")
