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
# This generates continuous daily estimates and daily/monthly returns
# Note: run_merton_estimation now returns (merged_daily_df, monthly_returns_df)
df_merged, monthly_returns_df = run_merton_estimation(df, interest_rates_df)

# Save Step 1 & 2 Results
df_merged.to_csv("merged_data_with_merton.csv", index=False)
monthly_returns_df.to_csv("monthly_asset_returns.csv", index=False)

print("Saved 'merged_data_with_merton.csv'")
print("Saved 'monthly_asset_returns.csv'")

#%% 
# 3. Run GARCH Model
final_monthly_returns = run_garch_estimation(monthly_returns_df)

# Save Final Results
final_monthly_returns.to_csv("monthly_asset_returns_with_garch.csv", index=False)
print("Saved 'monthly_asset_returns_with_garch.csv'")

#%%
# 4. Run Regime Switching Model (Hamilton Filter)
final_monthly_returns_rs = run_regime_switching_estimation(monthly_returns_df)

# Save Regime Switching Results
final_monthly_returns_rs.to_csv("monthly_asset_returns_with_regime.csv", index=False)
print("Saved 'monthly_asset_returns_with_regime.csv'")

#%%
# 5. Run MS-GARCH (Markov Switching GARCH)
final_monthly_returns_msgarch = run_ms_garch_estimation(monthly_returns_df)

# Save MS-GARCH Results
final_monthly_returns_msgarch.to_csv("monthly_asset_returns_with_msgarch.csv", index=False)
print("Saved 'monthly_asset_returns_with_msgarch.csv'")

#%%
# 6. Calculate Probability of Default (using MS-GARCH volatility)
pd_results = run_pd_pipeline('monthly_asset_returns_with_garch.csv', 'monthly_asset_returns_with_regime.csv', 'monthly_asset_returns_with_msgarch.csv')

# Save PD Results
pd_results.to_csv("monthly_pd_results.csv", index=False)
print("Saved 'monthly_pd_results.csv'")

#%%
# 6b. Calculate Probability of Default (Merton Model with Normal Returns - Benchmark)
from probability_of_default import calculate_merton_pd_normal

merton_normal_pd = calculate_merton_pd_normal('monthly_asset_returns.csv')
merton_normal_pd.to_csv("monthly_pd_results_merton_normal.csv", index=False)
print("Saved 'monthly_pd_results_merton_normal.csv'")

#%%
# 6c. Monte Carlo GARCH Volatility Forecast (1-year, single firm FIRST)
from monte_carlo_garch import monte_carlo_garch_1year

# Get first firm in dataset for testing
garch_data = pd.read_csv('monthly_asset_returns_with_garch.csv')
first_gvkey = garch_data[garch_data['garch_volatility'].notna()]['gvkey'].iloc[0]

print(f"Testing with firm: {first_gvkey}")

mc_results = monte_carlo_garch_1year('monthly_asset_returns_with_garch.csv', 
                                      gvkey_selected=first_gvkey,
                                      num_simulations=1000,
                                      num_months=12)

mc_results.to_csv("monte_carlo_garch_results.csv", index=False)
print("Saved 'monte_carlo_garch_results.csv'")

# To run for ALL firms later, just change to:
# mc_results = monte_carlo_garch_1year('monthly_asset_returns_with_garch.csv', 
#                                       gvkey_selected=None,
#                                       num_simulations=1000,
#                                       num_months=12)

#%%
# 7. Generate Summary and Plot
import importlib
import data_processing
import regime_switching
import probability_of_default
import result_summary

# Force reload of modified modules to ensure fixes are applied
importlib.reload(data_processing)
importlib.reload(regime_switching)
importlib.reload(probability_of_default)
importlib.reload(result_summary)

from data_processing import run_merton_estimation, load_interest_rates
from regime_switching import run_regime_switching_estimation
from probability_of_default import run_pd_pipeline
from result_summary import generate_results_summary

print("Re-running pipeline with corrected units...")

# RE-RUN STEPS 2, 4, 6, 7

# --- Step 2: Merton (Corrected Units) ---
# We need 'df' and 'interest_rates_df' from Step 1 loaded in memory
# If they aren't, uncomment the lines below:
# interest_rates_df = load_interest_rates()
# df = load_and_preprocess_data() 

if 'df' in locals() and 'interest_rates_df' in locals():
    df_merged, monthly_returns_df = run_merton_estimation(df, interest_rates_df)
    df_merged.to_csv("merged_data_with_merton.csv", index=False)
    monthly_returns_df.to_csv("monthly_asset_returns.csv", index=False)
    print("Step 2 (Merton) Complete")

    # --- Step 4: Regime Switching ---
    final_monthly_returns_rs = run_regime_switching_estimation(monthly_returns_df)
    final_monthly_returns_rs.to_csv("monthly_asset_returns_with_regime.csv", index=False)
    print("Step 4 (Regime Switching) Complete")

    # --- Step 6: Probability of Default ---
    # Note: We need GARCH and MS-GARCH files to exist. If you haven't re-run them, 
    # they will use old data. Ideally re-run ALL estimations.
    pd_results = run_pd_pipeline('monthly_asset_returns_with_garch.csv', 
                                 'monthly_asset_returns_with_regime.csv', 
                                 'monthly_asset_returns_with_msgarch.csv')

    # --- Step 7: Summary ---
    generate_results_summary("monthly_pd_results.csv")
else:
    print("Error: 'df' or 'interest_rates_df' not found. Please run Step 1 (Load Data) first.")


# %%
print("hi")
