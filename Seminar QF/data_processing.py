# data_processing.py

import pandas as pd
import numpy as np
from scipy.stats import norm

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
FILENAME_EQUITY_DATA = "Jan2025_Accenture_Dataset_ErasmusCase.xlsx"
FILENAME_INTEREST_RATES = "ECB Data Portal_20260125170805.csv"
SHEET_EQUITY = 0
SHEET_LIABILITY = 1
MIN_OBSERVATIONS = 150 # Minimum daily observations for Merton estimation
# RISK_FREE_RATE = 0.05  # Now loaded from ECB data
# Time horizon T is 1 year (risk horizon)
T_HORIZON = 1.0


def load_interest_rates():
    """
    Load monthly interest rates from ECB data.
    Returns a dataframe with month_year and risk_free_rate columns.
    """
    rates_df = pd.read_csv(FILENAME_INTEREST_RATES)
    
    # Extract the Euribor rate column (the long column name with EURIBOR)
    rate_cols = [col for col in rates_df.columns if 'EURIBOR' in col.upper()]
    
    rates_df['DATE'] = pd.to_datetime(rates_df['DATE'])
    rates_df['month_year'] = rates_df['DATE'].dt.strftime('%Y-%m')
    rates_df['risk_free_rate'] = pd.to_numeric(rates_df[rate_cols[0]], errors='coerce') / 100
    
    return rates_df[['month_year', 'risk_free_rate']].drop_duplicates()


def load_and_preprocess_data():
    """Reads Excel data, cleanses it, merges equity and liabilities."""
    print("Loading equity data...")
    df = pd.read_excel(FILENAME_EQUITY_DATA, sheet_name=SHEET_EQUITY)
    
    df = df.rename(columns={
        "(fic) Current ISO Country Code - Incorporation": "country",
        "(isin) International Security Identification Number": "isin",
        "(datadate) Data Date - Daily Prices": "date",
        "(conm) Company Name": "company",
        "(gvkey) Global Company Key - Company": "gvkey",
        "(cshoc) Shares Outstanding": "shares_out",
        "(prccd) Price - Close - Daily": "close",
        "Market Capitalization (# Shares * Close Price)": "mkt_cap",
    })
    
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.sort_values(["isin", "date"])
    
    # Fill missing prices/shares
    df[["shares_out", "close"]] = (
        df.groupby("isin")[["shares_out", "close"]]
          .ffill()
          .bfill()
    )
    df["mkt_cap"] = df["shares_out"] * df["close"]
    
    print("Loading liability data...")
    df2 = pd.read_excel(FILENAME_EQUITY_DATA, sheet_name=SHEET_LIABILITY)
    
    df2 = df2.rename(columns={
        "(gvkey) Global Company Key - Company": "gvkey",
        "(fyear) Data Year - Fiscal": "fyear",
        "(lt) Liabilities - Total": "liabilities_total",
        "(datadate) Data Date": "date", 
    })
    
    # Merge strategy: Match by Fiscal Year
    df["fyear"] = df["date"].dt.year
    df2_subset = df2[["gvkey", "fyear", "liabilities_total"]].drop_duplicates(subset=["gvkey", "fyear"])
    
    df = pd.merge(df, df2_subset, on=["gvkey", "fyear"], how="left")

    print("Loaded liability data")
    
    return df

def run_merton_estimation(df, interest_rates_df=None):
    """
    Runs the rolling window iterative Merton model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input data with equity and liability information
    interest_rates_df : pd.DataFrame, optional
        DataFrame with columns 'month_year' and 'risk_free_rate'
        If None, uses a default rate of 0.05
    
    Returns:
    --------
    full_df (pd.DataFrame): The original dataframe with added Merton columns (filled monthly).
    monthly_returns_df (pd.DataFrame): Dataframe containing monthly asset returns and volatilities.
    """
    
    # Filter valid data
    solver_df = df.dropna(subset=["mkt_cap", "liabilities_total"]).copy()
    solver_df = solver_df.sort_values(["gvkey", "date"])
    
    if solver_df.empty:
        raise ValueError("No valid data found for Merton model estimation.")

    print(f"Starting Rolling Merton model estimation for {len(solver_df)} rows...")
    
    results_list = []
    firms = solver_df["gvkey"].unique()
    
    print(f"Processing {len(firms)} firms...")

    for i, gvkey in enumerate(firms):
        firm_data = solver_df[solver_df["gvkey"] == gvkey].sort_values("date")
        if firm_data.empty:
            continue
            
        # Resample to find month ends present in the data.
        month_ends = firm_data.groupby(firm_data["date"].dt.to_period("M"))["date"].max()
        
        for date_t in month_ends:
            # Define 12-month estimation window
            start_date = date_t - pd.DateOffset(months=12)
            
            # Slice data
            window_mask = (firm_data["date"] > start_date) & (firm_data["date"] <= date_t)
            window_df = firm_data.loc[window_mask]
            
            if len(window_df) < MIN_OBSERVATIONS:
                continue
                
            # Get interest rate for this month
            month_str = date_t.strftime('%Y-%m')
            if interest_rates_df is not None and month_str in interest_rates_df['month_year'].values:
                r = interest_rates_df[interest_rates_df['month_year'] == month_str]['risk_free_rate'].values[0]
            else:
                r = 0.05  # Default fallback rate
                
            # Inputs
            E_vec = window_df["mkt_cap"].values
            B_vec = window_df["liabilities_total"].values * 1_000_000
            
            # --- Merton Core Logic ---
            E_safe = np.maximum(E_vec, 1e-4) 
            ret_E = np.diff(np.log(E_safe))
            ret_E = ret_E[np.isfinite(ret_E)]
            
            if len(ret_E) < 10:
                continue
                
            sigma_E_init = np.std(ret_E) * np.sqrt(252)
            if sigma_E_init < 1e-6: 
                sigma_E_init = 0.4
            
            sigma_A = sigma_E_init
            
            # Params (r was already determined from interest_rates_df above)
            T_val = T_HORIZON
            tol = 1e-4
            max_iter_algo = 100
            
            V_A_vec = E_vec + B_vec # Initial guess
            
            for k in range(max_iter_algo):
                sigma_A_old = sigma_A
                
                # Inner Loop: Newton-Raphson for V_A
                for _ in range(5):
                    V_A_vec = np.maximum(V_A_vec, 1e-4)
                    sig_sqrt_T = sigma_A * np.sqrt(T_val)
                    sig2_half = 0.5 * sigma_A**2
                    
                    d1 = (np.log(V_A_vec / B_vec) + (r + sig2_half) * T_val) / sig_sqrt_T
                    d2 = d1 - sig_sqrt_T
                    Nd1 = norm.cdf(d1)
                    Nd2 = norm.cdf(d2)
                    
                    f_val = (V_A_vec * Nd1 - B_vec * np.exp(-r * T_val) * Nd2) - E_vec
                    f_prime = Nd1
                    
                    mask_valid = f_prime > 1e-5
                    if not np.any(mask_valid):
                        break
                        
                    step = np.zeros_like(V_A_vec)
                    step[mask_valid] = f_val[mask_valid] / f_prime[mask_valid]
                    V_A_vec = V_A_vec - step

                V_A_vec = np.maximum(V_A_vec, 1e-4)
                
                # Outer Update: sigma_A
                with np.errstate(divide='ignore', invalid='ignore'):
                     ret_A = np.diff(np.log(V_A_vec))
                ret_A = ret_A[np.isfinite(ret_A)]
                
                if len(ret_A) < 10: break

                sigma_A_new = np.std(ret_A) * np.sqrt(252)
                
                if abs(sigma_A_new - sigma_A_old) < tol:
                    sigma_A = sigma_A_new
                    break
                sigma_A = sigma_A_new
            
            # Save Month-End Result
            V_A_final = V_A_vec[-1]
            
            results_list.append({
                "gvkey": gvkey,
                "date": date_t,
                "asset_value": V_A_final,
                "asset_volatility": sigma_A,
            })

        if (i + 1) % 10 == 0:
            print(f"Processed Merton for {i+1} firms...")

    merton_results = pd.DataFrame(results_list)
    print("Merton Estimation complete.")
    
    # 1. Merge back to main DF
    df_merged = pd.merge(df, merton_results, on=["gvkey", "date"], how="left", suffixes=("", "_merton"))
    
    # 2. Compute Monthly Returns DataFrame
    monthly_returns_df = pd.DataFrame()
    if not merton_results.empty:
        monthly_app = merton_results.copy().sort_values(["gvkey", "date"])
        monthly_app["asset_return_monthly"] = monthly_app.groupby("gvkey")["asset_value"].transform(
            lambda x: np.log(x / x.shift(1))
        )
        monthly_app["month_year"] = monthly_app["date"].dt.to_period("M")
        
        monthly_returns_df = monthly_app[["gvkey", "month_year", "asset_return_monthly", "asset_value", "asset_volatility"]].dropna()
        
    return df_merged, monthly_returns_df
