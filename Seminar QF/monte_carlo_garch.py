# monte_carlo_garch.py

import pandas as pd
import numpy as np
import time
from datetime import timedelta

def monte_carlo_garch_1year(garch_file, gvkey_selected=None, num_simulations=1000, num_days=252):
    """
    Monte Carlo simulation for 1-year GARCH(1,1) volatility paths.
    Runs for EACH daily date for EACH firm in the dataset.
    
    Parameters:
    - garch_file: CSV with GARCH results (daily)
    - gvkey_selected: Single firm to process. If None, process all firms.
    - num_simulations: Number of paths per starting date (default 1000)
    - num_days: Forecast horizon in days (default 252 = 1 year)
    
    Returns:
    - DataFrame with simulation results for each (gvkey, start_date) combination
    """
    
    overall_start = time.time()
    
    print(f"\n{'='*80}")
    print(f"Monte Carlo GARCH(1,1) Simulation (Daily)")
    print(f"{'='*80}")
    print(f"Simulations per date: {num_simulations}")
    print(f"Forecast horizon: {num_days} days")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load GARCH results
    print("Loading GARCH data...")
    load_start = time.time()
    df = pd.read_csv(garch_file)
    df['date'] = pd.to_datetime(df['date'])
    load_time = time.time() - load_start
    print(f"✓ Loaded {len(df)} rows in {load_time:.2f}s")
    
    # Filter to rows with complete GARCH parameters
    print("Filtering complete GARCH parameters...")
    df_complete = df.dropna(subset=['garch_volatility', 'garch_omega', 'garch_alpha', 'garch_beta']).copy()
    
    if df_complete.empty:
        print("✗ No complete GARCH data found.")
        return pd.DataFrame()
    
    print(f"✓ {len(df_complete)} rows with complete parameters\n")
    
    # Filter to selected firm if specified
    if gvkey_selected is not None:
        df_complete = df_complete[df_complete['gvkey'] == gvkey_selected]
        print(f"Processing selected firm: {gvkey_selected}")
        print(f"  - {len(df_complete)} daily observations\n")
    else:
        print(f"Processing all firms")
    
    firms = df_complete['gvkey'].unique()
    print(f"Number of firms to process: {len(firms)}\n")
    
    all_results = []
    total_sims_run = 0
    
    np.random.seed(42)
    
    # Loop through each firm
    for firm_idx, gvkey in enumerate(firms):
        firm_start = time.time()
        firm_data = df_complete[df_complete['gvkey'] == gvkey].copy()
        firm_data = firm_data.sort_values('date')
        
        num_dates = len(firm_data)
        print(f"\n{'─'*80}")
        print(f"Firm {firm_idx + 1}/{len(firms)}: gvkey={gvkey}")
        print(f"  Daily observations: {num_dates}")
        print(f"  Expected simulations: {num_dates * num_simulations:,}")
        print(f"{'─'*80}")
        
        firm_sims = 0
        
        # Loop through each daily date for this firm
        for date_idx, (idx, row) in enumerate(firm_data.iterrows()):
            start_date = row['date']
            start_vol = row['garch_volatility']
            omega = row['garch_omega']
            alpha = row['garch_alpha']
            beta = row['garch_beta']
            
            # Initialize simulation matrix for this starting date
            volatility_paths = np.zeros((num_simulations, num_days + 1))
            volatility_paths[:, 0] = start_vol
            
            # Monte Carlo simulation: 1000 paths, each 252 days forward
            for sim in range(num_simulations):
                for day in range(1, num_days + 1):
                    prev_vol = volatility_paths[sim, day - 1]
                    
                    # Random shock: return from N(0, prev_vol²)
                    random_return = np.random.normal(0, prev_vol)
                    shock_squared = random_return ** 2
                    
                    # GARCH(1,1) variance recursion
                    new_variance = omega + alpha * shock_squared + beta * (prev_vol ** 2)
                    new_vol = np.sqrt(max(new_variance, 0.0001))
                    
                    volatility_paths[sim, day] = new_vol
            
            firm_sims += num_simulations
            total_sims_run += num_simulations
            
            # Extract results at day 252 and annualize
            final_vols_daily = volatility_paths[:, num_days]
            final_vols_annualized = final_vols_daily * np.sqrt(252)
            
            mean_final_vol = final_vols_annualized.mean()
            std_final_vol = final_vols_annualized.std()
            percentile_5 = np.percentile(final_vols_annualized, 5)
            percentile_95 = np.percentile(final_vols_annualized, 95)
            
            # Append result for this starting date
            all_results.append({
                'gvkey': gvkey,
                'start_date': start_date,
                'end_date': start_date + pd.DateOffset(days=num_days),
                'initial_volatility_daily': start_vol,
                'initial_volatility_annualized': start_vol * np.sqrt(252),
                'garch_omega': omega,
                'garch_alpha': alpha,
                'garch_beta': beta,
                'mean_final_volatility_annualized': mean_final_vol,
                'std_final_volatility_annualized': std_final_vol,
                'percentile_5_annualized': percentile_5,
                'percentile_95_annualized': percentile_95
            })
            
            # Progress update every 50 dates
            if (date_idx + 1) % 50 == 0 or date_idx == num_dates - 1:
                pct_complete = ((date_idx + 1) / num_dates) * 100
                print(f"  Progress: {date_idx + 1}/{num_dates} dates ({pct_complete:.1f}%) | "
                      f"Sims: {firm_sims:,}")
        
        firm_time = time.time() - firm_start
        sims_per_second = firm_sims / firm_time if firm_time > 0 else 0
        print(f"\n✓ Firm {firm_idx + 1} complete:")
        print(f"    Total simulations: {firm_sims:,}")
        print(f"    Time elapsed: {timedelta(seconds=int(firm_time))}")
        print(f"    Speed: {sims_per_second:,.0f} sims/second")
    
    results_df = pd.DataFrame(all_results)
    
    # Final summary
    overall_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print(f"Monte Carlo Simulation Complete")
    print(f"{'='*80}")
    print(f"Total simulations run: {total_sims_run:,}")
    print(f"Total time: {timedelta(seconds=int(overall_time))}")
    print(f"Average speed: {total_sims_run / overall_time:,.0f} sims/second")
    print(f"\nResults Summary:")
    print(f"  Total result rows: {len(results_df)}")
    print(f"  Firms processed: {results_df['gvkey'].nunique()}")
    
    if len(results_df) > 0:
        print(f"  Date range: {results_df['start_date'].min().strftime('%Y-%m-%d')} to {results_df['start_date'].max().strftime('%Y-%m-%d')}")
        print(f"  Mean 252D volatility: {results_df['mean_final_volatility_annualized'].min():.4f} - {results_df['mean_final_volatility_annualized'].max():.4f}")
    
    print(f"{'='*80}\n")
    
    return results_df