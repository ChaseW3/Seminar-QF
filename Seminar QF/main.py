# main.py
#%% 
import pandas as pd
from data_processing import load_and_preprocess_data, run_merton_estimation, load_interest_rates
from garch_model import run_garch_estimation
from regime_switching import run_regime_switching_estimation
from ms_garch import run_ms_garch_estimation
from probability_of_default import run_pd_pipeline
from result_summary import generate_results_summary

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
# 5. Run MS-GARCH (Markov Switching GARCH)
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
# 8. Calculate Model-Implied CDS Spreads (Section 2.4.2)
from cds_spread_calculator import CDSSpreadCalculator

cds_calc = CDSSpreadCalculator(maturity_horizons=[1, 3, 5])

# 8a. CDS Spreads from GARCH Monte Carlo
print("\n" + "="*80)
print("STEP 8a: CDS SPREADS FROM GARCH MONTE CARLO")
print("="*80)

df_cds_spreads_garch = cds_calc.calculate_cds_spreads_from_mc_garch(
    mc_garch_file='daily_monte_carlo_garch_results.csv',
    daily_returns_file='daily_asset_returns.csv',
    merton_file='merged_data_with_merton.csv',
    output_file='cds_spreads_garch_mc.csv'
)
print("Saved CDS spreads to 'cds_spreads_garch_mc.csv'")

# 8b. CDS Spreads from Regime-Switching Monte Carlo
print("\n" + "="*80)
print("STEP 8b: CDS SPREADS FROM REGIME-SWITCHING MONTE CARLO")
print("="*80)

df_cds_spreads_rs = cds_calc.calculate_cds_spreads_from_mc_garch(
    mc_garch_file='daily_monte_carlo_regime_switching_results.csv',
    daily_returns_file='daily_asset_returns.csv',
    merton_file='merged_data_with_merton.csv',
    output_file='cds_spreads_regime_switching_mc.csv'
)
print("Saved CDS spreads to 'cds_spreads_regime_switching_mc.csv'")

#%%
# 9. Compare CDS Spreads: GARCH vs Regime-Switching
print("\n" + "="*80)
print("STEP 9: CDS SPREAD COMPARISON (GARCH vs REGIME-SWITCHING)")
print("="*80 + "\n")

# Merge the two CDS spread datasets
df_cds_garch = pd.read_csv('cds_spreads_garch_mc.csv')
df_cds_rs = pd.read_csv('cds_spreads_regime_switching_mc.csv')

df_cds_garch['date'] = pd.to_datetime(df_cds_garch['date'])
df_cds_rs['date'] = pd.to_datetime(df_cds_rs['date'])

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

# Merge on gvkey and date
df_comparison = pd.merge(
    df_cds_garch[['gvkey', 'date', 'cds_garch_1y_bps', 'cds_garch_3y_bps', 'cds_garch_5y_bps']],
    df_cds_rs[['gvkey', 'date', 'cds_rs_1y_bps', 'cds_rs_3y_bps', 'cds_rs_5y_bps']],
    on=['gvkey', 'date'],
    how='inner'
)

# Calculate differences
df_comparison['diff_1y_bps'] = df_comparison['cds_rs_1y_bps'] - df_comparison['cds_garch_1y_bps']
df_comparison['diff_3y_bps'] = df_comparison['cds_rs_3y_bps'] - df_comparison['cds_garch_3y_bps']
df_comparison['diff_5y_bps'] = df_comparison['cds_rs_5y_bps'] - df_comparison['cds_garch_5y_bps']

# Print summary statistics
print("CDS SPREAD COMPARISON STATISTICS (basis points):\n")

for maturity in [1, 3, 5]:
    garch_col = f'cds_garch_{maturity}y_bps'
    rs_col = f'cds_rs_{maturity}y_bps'
    diff_col = f'diff_{maturity}y_bps'
    
    print(f"Maturity {maturity}Y:")
    print(f"  GARCH MC:           Mean={df_comparison[garch_col].mean():8.2f}, Median={df_comparison[garch_col].median():8.2f}")
    print(f"  Regime-Switching:   Mean={df_comparison[rs_col].mean():8.2f}, Median={df_comparison[rs_col].median():8.2f}")
    print(f"  Difference (RS-G):  Mean={df_comparison[diff_col].mean():8.2f}, Median={df_comparison[diff_col].median():8.2f}")
    print(f"  Correlation:        {df_comparison[garch_col].corr(df_comparison[rs_col]):.4f}")
    print()

# Save comparison
df_comparison.to_csv('cds_spreads_comparison.csv', index=False)
print("Saved comparison to 'cds_spreads_comparison.csv'")

# Summary by firm
print("\nPer-firm average CDS spreads (5Y):")
firm_summary = df_comparison.groupby('gvkey').agg({
    'cds_garch_5y_bps': 'mean',
    'cds_rs_5y_bps': 'mean',
    'diff_5y_bps': 'mean'
}).round(2)
print(firm_summary.head(10))

print("\n" + "="*80)
print("PIPELINE COMPLETE")
print("="*80)

# %
