
import pandas as pd
import numpy as np

# Load the data
print("Loading GARCH data...")
df_garch = pd.read_csv(r'Seminar QF/cds_spreads_garch_mc.csv')

# Check column names
print("Columns:", df_garch.columns.tolist())

# Rename for consistency if needed (checking if they are already named bps)
cols_to_check = [c for c in df_garch.columns if 'bps' in c]
print(f"Checking statistics for: {cols_to_check}")

# Describe statistics
print(df_garch[cols_to_check].describe())

# Check for extreme values
for col in cols_to_check:
    print(f"\nTop 10 values for {col}:")
    print(df_garch[col].nlargest(10))
    
    print(f"\nCount > 1000 bps for {col}:")
    print((df_garch[col] > 1000).sum())

# Check if there are NaNs
print("\nNaN counts:")
print(df_garch[cols_to_check].isna().sum())

print("\nSpecific Firm 14447 stats:")
print(df_garch[df_garch['gvkey'] == 14447][cols_to_check].describe())
