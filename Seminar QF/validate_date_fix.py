"""
Quick validation script for date-handling bug fix.
Run this after executing the Merton estimation to verify the fix.
"""

import pandas as pd
from pathlib import Path
from src.utils import config

print("="*80)
print("VALIDATING DATE-HANDLING BUG FIX")
print("="*80)

# Load the merged data
merged_file = config.OUTPUT_DIR / 'merged_data_with_merton.csv'
if not merged_file.exists():
    print(f"❌ File not found: {merged_file}")
    print("   Please run the Merton estimation first.")
    exit(1)

df = pd.read_csv(merged_file, parse_dates=['date'])

print(f"\n1. Checking merged_data_with_merton.csv:")
print(f"   - Total rows: {len(df):,}")
print(f"   - Total firms: {df['gvkey'].nunique()}")
print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")

# Check when data starts
equity_start = df['date'].min()
print(f"\n2. Dataset starts at: {equity_start}")
if equity_start.year == 2010:
    print(f"   ✅ CORRECT: Data starts in 2010 (equity data start)")
elif equity_start.year == 2011 and equity_start.month >= 9:
    print(f"   ❌ BUG PRESENT: Data starts in Sep 2011 (should be 2010)")
elif equity_start.year == 2011 and equity_start.month < 9:
    print(f"   ⚠️  Data starts in early 2011 (check if this is expected)")

# Check when Merton columns start
df_merton = df.dropna(subset=['asset_value'])
if not df_merton.empty:
    merton_start = df_merton['date'].min()
    print(f"\n3. First Merton result: {merton_start}")
    
    expected_start = equity_start + pd.DateOffset(days=252)
    print(f"   Expected around: {expected_start.strftime('%Y-%m-%d')} (252 days after equity start)")
    
    if merton_start.year == 2011 and merton_start.month == 1:
        print(f"   ✅ CORRECT: Merton starts ~Jan 2011 (after 252-day window)")
    else:
        days_diff = (merton_start - equity_start).days
        print(f"   ℹ️  Actual delay: {days_diff} days from equity start")
else:
    print(f"\n3. ❌ No Merton results found!")

# Check when liabilities start
df_liab = df.dropna(subset=['liabilities_total'])
if not df_liab.empty:
    liab_start = df_liab['date'].min()
    print(f"\n4. First liabilities: {liab_start}")
    if liab_start.year == 2011 and liab_start.month in [1, 2, 3]:
        print(f"   ✅ CORRECT: Liabilities start in early 2011 (as expected)")
    else:
        print(f"   ℹ️  Liabilities start date: {liab_start}")
else:
    print(f"\n4. ❌ No liabilities found!")

# Check complete data (both Merton and liabilities)
df_complete = df.dropna(subset=['asset_value', 'liabilities_total'])
if not df_complete.empty:
    complete_start = df_complete['date'].min()
    print(f"\n5. Complete data (Merton + Liabilities): {complete_start}")
    print(f"   - Total rows: {len(df_complete):,}")
    print(f"   - Total firms: {df_complete['gvkey'].nunique()}")
    
    if complete_start.year == 2011 and complete_start.month in [2, 3]:
        print(f"   ✅ CORRECT: Complete data starts Feb-Mar 2011")
    elif complete_start.year == 2011 and complete_start.month >= 9:
        print(f"   ❌ BUG PRESENT: Complete data starts Sep 2011 (should be Feb-Mar 2011)")
else:
    print(f"\n5. ❌ No complete data found!")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
