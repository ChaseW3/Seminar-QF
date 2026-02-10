"""
Test improved Merton solver on ultra-high leverage firms (15532 BNP, 16348 Intesa).

This script focuses specifically on the two remaining problematic firms.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_processing import run_merton_estimation
from src.utils import config

print("="*80)
print("TESTING ULTRA-HIGH LEVERAGE MERTON IMPROVEMENTS")
print("="*80)
print("\nFocus: Firms 15532 (BNP) and 16348 (Intesa)")
print("Expected: Max daily returns < 50%, reasonable asset values")
print()

# Load original data
print("Loading data...")
df = pd.read_excel(config.INPUT_DIR / 'Jan2025_Accenture_Dataset_ErasmusCase.xlsx', 
                   sheet_name='Sheet1', engine='openpyxl')
df2 = pd.read_excel(config.INPUT_DIR / 'Jan2025_Accenture_Dataset_ErasmusCase.xlsx', 
                    sheet_name='Sheet2', engine='openpyxl')

# Filter to just the two problematic firms
problem_firms = [15532, 16348]
df = df[df['gvkey'].isin(problem_firms)].copy()
df2 = df2[df2['gvkey'].isin(problem_firms)].copy()

print(f"✓ Loaded {len(df):,} observations for {df['gvkey'].nunique()} firms")
print()

# Run Merton estimation (will use parallel processing with 2 workers)
print("Running improved Merton estimation...")
print("(This will take ~30 seconds)")
print()

results_df = run_merton_estimation(df, df2)

print()
print("="*80)
print("VERIFICATION RESULTS")
print("="*80)

# Check each firm
for gvkey in problem_firms:
    firm_data = results_df[results_df['gvkey'] == gvkey].copy()
    firm_data = firm_data.sort_values('date')
    firm_name = firm_data['company'].iloc[0]
    
    # Calculate returns
    firm_data['asset_return'] = firm_data['asset_value'].pct_change()
    
    # Statistics
    leverage_median = (firm_data['liabilities_total'] / firm_data['mkt_cap']).median()
    max_return = firm_data['asset_return'].abs().max()
    extreme_returns = (firm_data['asset_return'].abs() > 0.5).sum()
    max_vol = firm_data['asset_volatility'].max()
    at_max_vol = (firm_data['asset_volatility'] >= 0.49).sum()  # Check if hitting 50% cap
    
    print(f"\nFirm {gvkey}: {firm_name}")
    print(f"  Median leverage: {leverage_median:.1f}×")
    print(f"  Max |return|: {max_return*100:.1f}%")
    print(f"  Extreme returns (>50%): {extreme_returns}")
    print(f"  Max volatility: {max_vol:.4f}")
    print(f"  Days at volatility cap: {at_max_vol}")
    
    if max_return < 0.5 and extreme_returns == 0:
        print(f"    ✅ FIXED! Returns now reasonable")
    elif max_return < 1.0 and extreme_returns < 10:
        print(f"    ⚠️  IMPROVED but still some extreme returns")
    else:
        print(f"    ❌ Still has extreme returns")
    
    # Find worst dates
    if extreme_returns > 0:
        worst_5 = firm_data.nlargest(5, lambda x: x['asset_return'].abs())
        print(f"\n  Worst 5 dates:")
        for idx, row in worst_5.iterrows():
            print(f"    {row['date']}: {row['asset_return']*100:+.1f}% return, "
                  f"Vol={row['asset_volatility']:.4f}, "
                  f"Lev={row['liabilities_total']/row['mkt_cap']:.1f}×")

print()
print("="*80)
print("SUMMARY")
print("="*80)

total_extreme = (results_df['asset_return'].abs() > 0.5).sum() if 'asset_return' in results_df.columns else 0
print(f"Total extreme returns (>50%): {total_extreme}")

if total_extreme == 0:
    print("\n✅ SUCCESS! Both firms now have reasonable returns.")
    print("Next step: Re-run full pipeline (Regime Switching → Monte Carlo → CDS)")
elif total_extreme < 50:
    print(f"\n⚠️  PARTIAL SUCCESS: Reduced from ~240 to {total_extreme} extreme returns")
    print("May need further tuning or consider excluding these firms")
else:
    print(f"\n❌ STILL PROBLEMATIC: {total_extreme} extreme returns remain")
    print("Consider alternative approaches (e.g., Merton with debt structure)")

print()
