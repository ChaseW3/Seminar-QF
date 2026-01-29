# monte_carlo_garch.py

import pandas as pd
import numpy as np

def monte_carlo_garch_1year(garch_file, gvkey_selected=None, num_simulations=1000, num_months=12):
    """
    Monte Carlo simulation for 1-year GARCH(1,1) volatility paths.
    
    Parameters:
    - garch_file: CSV with GARCH results
    - gvkey_selected: Single firm to process. If None, process all firms.
    - num_simulations: Number of paths per starting date (default 1000)
    - num_months: Forecast horizon in months (default 12)
    
    Returns:
    - DataFrame with simulation results for each (gvkey, start_date) combination
    """
    
    print(f"\nMonte Carlo GARCH(1,1) Simulation")
    print(f"Simulations per date: {num_simulations}, Horizon: {num_months} months\n")
    
    # Load GARCH results
    df = pd.read_csv(garch_file)
    df['month_year'] = pd.to_datetime(df['month_year'])
    
    # Filter to rows with complete GARCH parameters
    df_complete = df.dropna(subset=['garch_volatility', 'garch_omega', 'garch_alpha', 'garch_beta']).copy()
    
    if df_complete.empty:
        print("No complete GARCH data found.")
        return pd.DataFrame()
    
    # Filter to selected firm if specified
    if gvkey_selected is not None:
        df_complete = df_complete[df_complete['gvkey'] == gvkey_selected]
        print(f"Processing firm: {gvkey_selected}")
    else:
        print(f"Processing all firms")
    
    firms = df_complete['gvkey'].unique()
    print(f"Number of firms: {len(firms)}\n")
    
    all_results = []
    
    np.random.seed(42)
    
    # Loop through each firm
    for firm_idx, gvkey in enumerate(firms):
        firm_data = df_complete[df_complete['gvkey'] == gvkey].copy()
        firm_data = firm_data.sort_values('month_year')
        
        print(f"Firm {firm_idx + 1}/{len(firms)}: gvkey={gvkey}, months={len(firm_data)}")
        
        # Loop through each monthly date for this firm
        for idx, row in firm_data.iterrows():
            start_date = row['month_year']
            start_vol = row['garch_volatility']  # Monthly volatility
            omega = row['garch_omega']
            alpha = row['garch_alpha']
            beta = row['garch_beta']
            
            # Initialize simulation matrix for this starting date
            volatility_paths = np.zeros((num_simulations, num_months + 1))
            volatility_paths[:, 0] = start_vol  # Start with current monthly volatility
            
            # Monte Carlo simulation: 1000 paths, each 12 months forward
            for sim in range(num_simulations):
                for month in range(1, num_months + 1):
                    prev_vol = volatility_paths[sim, month - 1]
                    
                    # Random shock: return from N(0, prev_vol²)
                    random_return = np.random.normal(0, prev_vol)
                    shock_squared = random_return ** 2
                    
                    # GARCH(1,1) variance recursion
                    new_variance = omega + alpha * shock_squared + beta * (prev_vol ** 2)
                    new_vol = np.sqrt(max(new_variance, 0.0001))  # Ensure non-negative
                    
                    volatility_paths[sim, month] = new_vol
            
            # Extract results at month 12 and annualize
            final_vols_monthly = volatility_paths[:, num_months]
            final_vols_annualized = final_vols_monthly * np.sqrt(12)
            
            mean_final_vol = final_vols_annualized.mean()
            std_final_vol = final_vols_annualized.std()
            percentile_5 = np.percentile(final_vols_annualized, 5)
            percentile_95 = np.percentile(final_vols_annualized, 95)
            
            # Append result for this starting date
            all_results.append({
                'gvkey': gvkey,
                'start_date': start_date,
                'end_date': start_date + pd.DateOffset(months=num_months),
                'initial_volatility_monthly': start_vol,
                'initial_volatility_annualized': start_vol * np.sqrt(12),
                'garch_omega': omega,
                'garch_alpha': alpha,
                'garch_beta': beta,
                'mean_final_volatility_annualized': mean_final_vol,
                'std_final_volatility_annualized': std_final_vol,
                'percentile_5_annualized': percentile_5,
                'percentile_95_annualized': percentile_95
            })
    
    results_df = pd.DataFrame(all_results)
    
    print("\nResults Summary:")
    print(f"  Total rows created: {len(results_df)}")
    print(f"  Firms: {results_df['gvkey'].nunique()}")
    if len(results_df) > 0:
        print(f"  Date range: {results_df['start_date'].min().strftime('%Y-%m')} to {results_df['start_date'].max().strftime('%Y-%m')}")
        print(f"  Mean 12M volatility range: {results_df['mean_final_volatility_annualized'].min():.4f} to {results_df['mean_final_volatility_annualized'].max():.4f}\n")
    
    return results_df