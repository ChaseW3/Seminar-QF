# main.py
#%% 
import pandas as pd
import os
import shutil
from data_processing import load_and_preprocess_data, run_merton_estimation, load_interest_rates
from garch_model import run_garch_estimation
from regime_switching import run_regime_switching_estimation
from ms_garch import run_ms_garch_estimation
from probability_of_default import run_pd_pipeline
from result_summary import generate_results_summary

#%%
# CACHE CLEANUP: Delete cached Merton results to force reprocessing with updated firm list
print("Cleaning up cache files...\n")
cache_dir = './intermediates/'
cache_files = [
    'merton_results_cache.pkl',
    'mc_garch_cache.csv'
]

for cache_file in cache_files:
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            print(f"✓ Deleted: {cache_path}")
        except Exception as e:
            print(f"⚠ Could not delete {cache_path}: {e}")
    else:
        print(f"  (No cached file: {cache_path})")

print("Cache cleanup complete.\n")

#%% 
# 0. Load Interest Rates (needed for all steps)
interest_rates_df = load_interest_rates()
print(f"Loaded {len(interest_rates_df)} months of interest rate data")

#%% 
# 1. Load and Process Data
df = load_and_preprocess_data()

#%%
# 2. Run Merton Model (Rolling Window)
df_merged, daily_returns_df = run_merton_estimation(df, interest_rates_df)

# Save Step 1 & 2 Results
df_merged.to_csv("merged_data_with_merton.csv", index=False)
daily_returns_df.to_csv("daily_asset_returns.csv", index=False)

print("Saved 'merged_data_with_merton.csv'")
print("Saved 'daily_asset_returns.csv'")

#%%
# 3. Run GARCH Model
final_daily_returns = run_garch_estimation(daily_returns_df)

# Save Final Results
final_daily_returns.to_csv("daily_asset_returns_with_garch.csv", index=False)
print("Saved 'daily_asset_returns_with_garch.csv'")

#%%
# 4. Run Regime Switching Model (Hamilton Filter)
final_daily_returns_rs = run_regime_switching_estimation(daily_returns_df)

# Save Regime Switching Results
final_daily_returns_rs.to_csv("daily_asset_returns_with_regime.csv", index=False)
print("Saved 'daily_asset_returns_with_regime.csv'")

#%%
# 5. Run PROPER MS-GARCH (Markov Switching GARCH with GARCH dynamics per regime)
from ms_garch_proper import run_ms_garch_estimation

print("\n" + "="*80)
print("STEP 5: PROPER MS-GARCH ESTIMATION")
print("="*80)
print("Note: This is a TRUE MS-GARCH with GARCH(1,1) dynamics in each regime,")
print("      estimated via MLE with Hamilton filter.\n")

final_daily_returns_msgarch = run_ms_garch_estimation(daily_returns_df)

# Save MS-GARCH Results
final_daily_returns_msgarch.to_csv("daily_asset_returns_with_msgarch.csv", index=False)
print("Saved 'daily_asset_returns_with_msgarch.csv'")

#%%
# 6. Calculate Probability of Default (using MS-GARCH volatility)
pd_results = run_pd_pipeline('daily_asset_returns_with_garch.csv', 
                             'daily_asset_returns_with_regime.csv', 
                             'daily_asset_returns_with_msgarch.csv')

# Save PD Results
pd_results.to_csv("daily_pd_results.csv", index=False)
print("Saved 'daily_pd_results.csv'")

#%%
# 6b. Calculate Probability of Default (Merton Model with Normal Returns - Benchmark)
from probability_of_default import calculate_merton_pd_normal

merton_normal_pd = calculate_merton_pd_normal('daily_asset_returns.csv')
merton_normal_pd.to_csv("daily_pd_results_merton_normal.csv", index=False)
print("Saved 'daily_pd_results_merton_normal.csv'")

#%%
# 6c. Monte Carlo GARCH Volatility Forecast (1-year, ALL FIRMS)
from monte_carlo_garch import monte_carlo_garch_1year

print("\n" + "="*80)
print("STEP 6c: MONTE CARLO GARCH VOLATILITY FORECASTS (ALL FIRMS)")
print("="*80)

mc_results = monte_carlo_garch_1year('daily_asset_returns_with_garch.csv', 
                                      gvkey_selected=None,  # ALL firms
                                      num_simulations=1000,
                                      num_days=252)

mc_results.to_csv("daily_monte_carlo_garch_results.csv", index=False)
print("Saved 'daily_monte_carlo_garch_results.csv'")

#%%
# 6c.1 VOLATILITY DIAGNOSTICS - Identify problematic firms BEFORE CDS calculation
from volatility_diagnostics import run_volatility_diagnostics

print("\n" + "="*80)
print("STEP 6c.1: VOLATILITY DIAGNOSTICS - IDENTIFYING PROBLEMATIC FIRMS")
print("="*80)

diagnostics_results = run_volatility_diagnostics(
    garch_file='daily_asset_returns_with_garch.csv',
    mc_garch_file='daily_monte_carlo_garch_results.csv',
    output_dir='./diagnostics/'
)

# Store problematic firms for later use
PROBLEMATIC_FIRMS = diagnostics_results['problematic_firms']
CLEAN_FIRMS = diagnostics_results['clean_firms']

print(f"\n⚠️  PROBLEMATIC FIRMS IDENTIFIED: {len(PROBLEMATIC_FIRMS)}")
print(f"    These firms have extreme volatility and will distort CDS spread calculations.")
print(f"    Review ./diagnostics/problematic_firms.csv for details.")
print(f"✓  CLEAN FIRMS: {len(CLEAN_FIRMS)}")

#%%
# 6d. Monte Carlo Regime-Switching Volatility Forecast (1-year, ALL FIRMS)
from monte_carlo_regime_switching import monte_carlo_regime_switching_1year

print("\n" + "="*80)
print("STEP 6d: MONTE CARLO REGIME-SWITCHING VOLATILITY FORECASTS (ALL FIRMS)")
print("="*80)

mc_rs_results = monte_carlo_regime_switching_1year(
    garch_file='daily_asset_returns_with_garch.csv',
    regime_params_file='regime_switching_parameters.csv',
    gvkey_selected=None,  # ALL firms
    num_simulations=1000,
    num_days=252
)

mc_rs_results.to_csv("daily_monte_carlo_regime_switching_results.csv", index=False)
print("Saved 'daily_monte_carlo_regime_switching_results.csv'")

#%%
# 6e. Monte Carlo MS-GARCH Volatility Forecast (1-year, ALL FIRMS) - PROPER MS-GARCH
from monte_carlo_ms_garch import monte_carlo_ms_garch_1year

print("\n" + "="*80)
print("STEP 6e: MONTE CARLO MS-GARCH VOLATILITY FORECASTS (ALL FIRMS)")
print("="*80)
print("Note: This uses PROPER MS-GARCH with GARCH dynamics per regime\n")

mc_msgarch_results = monte_carlo_ms_garch_1year(
    daily_returns_file='daily_asset_returns_with_msgarch.csv',
    ms_garch_params_file='ms_garch_parameters.csv',
    gvkey_selected=None,  # ALL firms
    num_simulations=1000,
    num_days=252
)

mc_msgarch_results.to_csv("daily_monte_carlo_ms_garch_results.csv", index=False)
print("Saved 'daily_monte_carlo_ms_garch_results.csv'")

#%%
# 8. Calculate Model-Implied CDS Spreads (Section 2.4.2)
from cds_spread_calculator import CDSSpreadCalculator
from volatility_diagnostics import filter_problematic_firms

cds_calc = CDSSpreadCalculator(maturity_horizons=[1, 3, 5])

# 8a. CDS Spreads from GARCH Monte Carlo (ALL FIRMS - for comparison)
print("\n" + "="*80)
print("STEP 8a: CDS SPREADS FROM GARCH MONTE CARLO (ALL FIRMS)")
print("="*80)

df_cds_spreads_garch_all = cds_calc.calculate_cds_spreads_from_mc_garch(
    mc_garch_file='daily_monte_carlo_garch_results.csv',
    daily_returns_file='daily_asset_returns.csv',
    merton_file='merged_data_with_merton.csv',
    output_file='cds_spreads_garch_mc_all_firms.csv'
)
print("Saved CDS spreads to 'cds_spreads_garch_mc_all_firms.csv'")

# 8a.1 CDS Spreads from GARCH Monte Carlo (CLEAN FIRMS ONLY)
print("\n" + "="*80)
print("STEP 8a.1: CDS SPREADS FROM GARCH MONTE CARLO (CLEAN FIRMS ONLY)")
print("="*80)

# Filter to clean firms only
df_cds_spreads_garch = filter_problematic_firms(df_cds_spreads_garch_all, PROBLEMATIC_FIRMS)
df_cds_spreads_garch.to_csv('cds_spreads_garch_mc.csv', index=False)
print("Saved CDS spreads to 'cds_spreads_garch_mc.csv' (clean firms only)")

# 8b. CDS Spreads from Regime-Switching Monte Carlo (ALL FIRMS)
print("\n" + "="*80)
print("STEP 8b: CDS SPREADS FROM REGIME-SWITCHING MONTE CARLO (ALL FIRMS)")
print("="*80)

df_cds_spreads_rs_all = cds_calc.calculate_cds_spreads_from_mc_garch(
    mc_garch_file='daily_monte_carlo_regime_switching_results.csv',
    daily_returns_file='daily_asset_returns.csv',
    merton_file='merged_data_with_merton.csv',
    output_file='cds_spreads_regime_switching_mc_all_firms.csv'
)
print("Saved CDS spreads to 'cds_spreads_regime_switching_mc_all_firms.csv'")

# 8b.1 CDS Spreads from Regime-Switching Monte Carlo (CLEAN FIRMS ONLY)
print("\n" + "="*80)
print("STEP 8b.1: CDS SPREADS FROM REGIME-SWITCHING MC (CLEAN FIRMS ONLY)")
print("="*80)

df_cds_spreads_rs = filter_problematic_firms(df_cds_spreads_rs_all, PROBLEMATIC_FIRMS)
df_cds_spreads_rs.to_csv('cds_spreads_regime_switching_mc.csv', index=False)
print("Saved CDS spreads to 'cds_spreads_regime_switching_mc.csv' (clean firms only)")

#%%
# 8c. CDS Spreads from MS-GARCH Monte Carlo (ALL FIRMS)
print("\n" + "="*80)
print("STEP 8c: CDS SPREADS FROM MS-GARCH MONTE CARLO (ALL FIRMS)")
print("="*80)
print("Note: Using proper MS-GARCH with GARCH dynamics per regime\n")

df_cds_spreads_msgarch_all = cds_calc.calculate_cds_spreads_from_mc_garch(
    mc_garch_file='daily_monte_carlo_ms_garch_results.csv',
    daily_returns_file='daily_asset_returns.csv',
    merton_file='merged_data_with_merton.csv',
    output_file='cds_spreads_ms_garch_mc_all_firms.csv',
    volatility_column='mc_msgarch_integrated_variance'
)
print("Saved CDS spreads to 'cds_spreads_ms_garch_mc_all_firms.csv'")

# 8c.1 CDS Spreads from MS-GARCH Monte Carlo (CLEAN FIRMS ONLY)
print("\n" + "="*80)
print("STEP 8c.1: CDS SPREADS FROM MS-GARCH MC (CLEAN FIRMS ONLY)")
print("="*80)

df_cds_spreads_msgarch = filter_problematic_firms(df_cds_spreads_msgarch_all, PROBLEMATIC_FIRMS)
df_cds_spreads_msgarch.to_csv('cds_spreads_ms_garch_mc.csv', index=False)
print("Saved CDS spreads to 'cds_spreads_ms_garch_mc.csv' (clean firms only)")

#%%
# 9. Compare CDS Spreads: GARCH vs Regime-Switching vs MS-GARCH (CLEAN FIRMS ONLY)
print("\n" + "="*80)
print("STEP 9: CDS SPREAD COMPARISON (GARCH vs RS vs MS-GARCH) - CLEAN FIRMS ONLY")
print("="*80 + "\n")

print(f"Note: Comparison excludes {len(PROBLEMATIC_FIRMS)} problematic firms with extreme volatility.\n")

# Merge the three CDS spread datasets (clean firms only)
df_cds_garch = pd.read_csv('cds_spreads_garch_mc.csv')
df_cds_rs = pd.read_csv('cds_spreads_regime_switching_mc.csv')
df_cds_msgarch = pd.read_csv('cds_spreads_ms_garch_mc.csv')

df_cds_garch['date'] = pd.to_datetime(df_cds_garch['date'])
df_cds_rs['date'] = pd.to_datetime(df_cds_rs['date'])
df_cds_msgarch['date'] = pd.to_datetime(df_cds_msgarch['date'])

# Rename columns for clarity
df_cds_garch = df_cds_garch.rename(columns={
    'cds_spread_garch_mc_1y_bps': 'cds_garch_1y_bps',
    'cds_spread_garch_mc_3y_bps': 'cds_garch_3y_bps',
    'cds_spread_garch_mc_5y_bps': 'cds_garch_5y_bps'
})

df_cds_rs = df_cds_rs.rename(columns={
    'cds_spread_garch_mc_1y_bps': 'cds_rs_1y_bps',
    'cds_spread_garch_mc_3y_bps': 'cds_rs_3y_bps',
    'cds_spread_garch_mc_5y_bps': 'cds_rs_5y_bps'
})

df_cds_msgarch = df_cds_msgarch.rename(columns={
    'cds_spread_garch_mc_1y_bps': 'cds_msgarch_1y_bps',
    'cds_spread_garch_mc_3y_bps': 'cds_msgarch_3y_bps',
    'cds_spread_garch_mc_5y_bps': 'cds_msgarch_5y_bps'
})

# Merge on gvkey and date (all three models)
df_comparison = pd.merge(
    df_cds_garch[['gvkey', 'date', 'cds_garch_1y_bps', 'cds_garch_3y_bps', 'cds_garch_5y_bps']],
    df_cds_rs[['gvkey', 'date', 'cds_rs_1y_bps', 'cds_rs_3y_bps', 'cds_rs_5y_bps']],
    on=['gvkey', 'date'],
    how='inner'
)

df_comparison = pd.merge(
    df_comparison,
    df_cds_msgarch[['gvkey', 'date', 'cds_msgarch_1y_bps', 'cds_msgarch_3y_bps', 'cds_msgarch_5y_bps']],
    on=['gvkey', 'date'],
    how='inner'
)

# Calculate differences (MS-GARCH as reference)
df_comparison['diff_garch_msgarch_1y'] = df_comparison['cds_garch_1y_bps'] - df_comparison['cds_msgarch_1y_bps']
df_comparison['diff_garch_msgarch_3y'] = df_comparison['cds_garch_3y_bps'] - df_comparison['cds_msgarch_3y_bps']
df_comparison['diff_garch_msgarch_5y'] = df_comparison['cds_garch_5y_bps'] - df_comparison['cds_msgarch_5y_bps']
df_comparison['diff_rs_msgarch_1y'] = df_comparison['cds_rs_1y_bps'] - df_comparison['cds_msgarch_1y_bps']
df_comparison['diff_rs_msgarch_3y'] = df_comparison['cds_rs_3y_bps'] - df_comparison['cds_msgarch_3y_bps']
df_comparison['diff_rs_msgarch_5y'] = df_comparison['cds_rs_5y_bps'] - df_comparison['cds_msgarch_5y_bps']

# Print summary statistics
print("CDS SPREAD COMPARISON STATISTICS (basis points):\n")

for maturity in [1, 3, 5]:
    garch_col = f'cds_garch_{maturity}y_bps'
    rs_col = f'cds_rs_{maturity}y_bps'
    msgarch_col = f'cds_msgarch_{maturity}y_bps'
    
    print(f"Maturity {maturity}Y:")
    print(f"  GARCH MC:           Mean={df_comparison[garch_col].mean():8.2f}, Median={df_comparison[garch_col].median():8.2f}")
    print(f"  Regime-Switching:   Mean={df_comparison[rs_col].mean():8.2f}, Median={df_comparison[rs_col].median():8.2f}")
    print(f"  MS-GARCH:           Mean={df_comparison[msgarch_col].mean():8.2f}, Median={df_comparison[msgarch_col].median():8.2f}")
    print(f"  Correlations:")
    print(f"    GARCH vs RS:      {df_comparison[garch_col].corr(df_comparison[rs_col]):.4f}")
    print(f"    GARCH vs MSGARCH: {df_comparison[garch_col].corr(df_comparison[msgarch_col]):.4f}")
    print(f"    RS vs MSGARCH:    {df_comparison[rs_col].corr(df_comparison[msgarch_col]):.4f}")
    print()

# Save comparison
df_comparison.to_csv('cds_spreads_comparison.csv', index=False)
print("Saved comparison to 'cds_spreads_comparison.csv'")

# Summary by firm
print("\nPer-firm average CDS spreads (5Y):")
firm_summary = df_comparison.groupby('gvkey').agg({
    'cds_garch_5y_bps': 'mean',
    'cds_rs_5y_bps': 'mean',
    'cds_msgarch_5y_bps': 'mean'
}).round(2)
print(firm_summary.head(10))

#%%
# 10. FINAL SUMMARY - Problematic Firms Report
print("\n" + "="*80)
print("STEP 10: FINAL SUMMARY - PROBLEMATIC FIRMS REPORT")
print("="*80 + "\n")

print(f"ANALYSIS COMPLETE")
print(f"-" * 40)
print(f"Total firms in dataset: {len(PROBLEMATIC_FIRMS) + len(CLEAN_FIRMS)}")
print(f"Problematic firms (excluded from final comparison): {len(PROBLEMATIC_FIRMS)}")
print(f"Clean firms (used in final comparison): {len(CLEAN_FIRMS)}")

if len(PROBLEMATIC_FIRMS) > 0:
    print(f"\n⚠️  PROBLEMATIC FIRM GVKEYs:")
    print(f"    {PROBLEMATIC_FIRMS}")
    print(f"\n    These firms have extreme volatility estimates that cause unrealistic CDS spreads.")
    print(f"    To investigate root causes, see:")
    print(f"      - ./diagnostics/firm_volatility_diagnostics.csv (full firm-level analysis)")
    print(f"      - ./diagnostics/problematic_firms.csv (detailed issue breakdown)")
    print(f"      - ./diagnostics/diagnostics_summary.txt (summary report)")
    
    print(f"\n    Potential causes:")
    print(f"      1. GARCH parameter instability (α + β ≈ 1, near-IGARCH)")
    print(f"      2. Extreme return outliers in underlying data")
    print(f"      3. Fat-tailed return distributions (high kurtosis)")
    print(f"      4. Missing or insufficient data for GARCH estimation")
else:
    print(f"\n✓ No problematic firms detected - all firms have reasonable volatility estimates.")

print(f"\nOUTPUT FILES:")
print(f"  - cds_spreads_garch_mc.csv (clean firms only)")
print(f"  - cds_spreads_regime_switching_mc.csv (clean firms only)")
print(f"  - cds_spreads_garch_mc_all_firms.csv (includes problematic)")
print(f"  - cds_spreads_regime_switching_mc_all_firms.csv (includes problematic)")
print(f"  - cds_spreads_comparison.csv (clean firms comparison)")
print(f"  - ./diagnostics/ (volatility diagnostics reports)")

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("="*80)

# %
