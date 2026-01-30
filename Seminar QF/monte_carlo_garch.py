# monte_carlo_garch.py

import pandas as pd
import numpy as np
import time
from datetime import timedelta
import numba
import os

# Create cache directory if it doesn't exist
os.makedirs('./intermediates/', exist_ok=True)

@numba.jit(nopython=True, parallel=True)
def mc_garch_vectorized(omega, alpha, beta, sigma_0, num_simulations, num_days, random_seed):
    """
    Vectorized GARCH(1,1) Monte Carlo using Numba JIT compilation.
    
    This is ~100x faster than nested loops.
    Uses numba's parallel=True for additional speedup on multi-core.
    """
    np.random.seed(random_seed)
    
    # Generate ALL random shocks at once (not in loop)
    shocks = np.random.normal(0, 1, (num_simulations, num_days))
    
    # Initialize volatility paths
    paths = np.zeros((num_simulations, num_days + 1))
    paths[:, 0] = sigma_0
    
    # Vectorized GARCH recursion
    for day in range(1, num_days + 1):
        prev_vols = paths[:, day - 1]
        prev_vols_sq = prev_vols ** 2
        
        # Random return: shock * previous volatility
        returns = shocks[:, day - 1] * prev_vols
        returns_sq = returns ** 2
        
        # GARCH(1,1): h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
        variances = omega + alpha * returns_sq + beta * prev_vols_sq
        variances = np.maximum(variances, 1e-6)  # Ensure non-negative
        
        paths[:, day] = np.sqrt(variances)
    
    return paths


def monte_carlo_garch_1year(garch_file, gvkey_selected=None, num_simulations=1000, num_days=252, random_seed=42):
    """
    Monte Carlo simulation for 1-year GARCH(1,1) volatility paths (OPTIMIZED).
    
    Parameters:
    - garch_file: CSV with GARCH results (daily)
    - gvkey_selected: Single firm or list of firms. If None, process all firms.
    - num_simulations: Number of paths per starting date (default 1000)
    - num_days: Forecast horizon in days (default 252 = 1 year)
    - random_seed: Seed for reproducibility
    
    Returns:
    - DataFrame with simulation results for each (gvkey, start_date) combination
    """
    
    overall_start = time.time()
    
    print(f"\n{'='*80}")
    print(f"Monte Carlo GARCH(1,1) Simulation (Vectorized, Daily Data)")
    print(f"{'='*80}")
    print(f"Simulations per date: {num_simulations:,}")
    print(f"Forecast horizon: {num_days} days")
    print(f"Random seed: {random_seed}")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load GARCH data
    print("Loading GARCH data...")
    load_start = time.time()
    df = pd.read_csv(garch_file)
    df['date'] = pd.to_datetime(df['date'])
    load_time = time.time() - load_start
    print(f"✓ Loaded {len(df):,} rows in {load_time:.2f}s")
    
    # Filter to rows with complete GARCH parameters
    print("Filtering complete GARCH parameters...")
    df_complete = df.dropna(subset=['garch_volatility', 'garch_omega', 'garch_alpha', 'garch_beta']).copy()
    
    if df_complete.empty:
        print("✗ No complete GARCH data found.")
        return pd.DataFrame()
    
    print(f"✓ {len(df_complete):,} rows with complete parameters\n")
    
    # Handle firm selection (can be single gvkey or list)
    if gvkey_selected is not None:
        if isinstance(gvkey_selected, list):
            df_complete = df_complete[df_complete['gvkey'].isin(gvkey_selected)]
            print(f"Processing selected firms: {gvkey_selected}")
        else:
            df_complete = df_complete[df_complete['gvkey'] == gvkey_selected]
            print(f"Processing selected firm: {gvkey_selected}")
    
    firms = sorted(df_complete['gvkey'].unique())
    print(f"Number of firms to process: {len(firms)}\n")
    
    all_results = []
    total_sims_run = 0
    
    # Loop through each firm
    for firm_idx, gvkey in enumerate(firms):
        firm_start = time.time()
        firm_data = df_complete[df_complete['gvkey'] == gvkey].copy()
        firm_data = firm_data.sort_values('date')
        
        num_dates = len(firm_data)
        print(f"\n{'─'*80}")
        print(f"Firm {firm_idx + 1}/{len(firms)}: gvkey={gvkey}")
        print(f"  Daily observations: {num_dates:,}")
        print(f"  Expected simulations: {num_dates * num_simulations:,}")
        print(f"{'─'*80}")
        
        firm_sims = 0
        firm_results_batch = []
        
        # Process each daily date for this firm
        for date_idx, (idx, row) in enumerate(firm_data.iterrows()):
            start_date = row['date']
            start_vol = row['garch_volatility']
            omega = row['garch_omega']
            alpha = row['garch_alpha']
            beta = row['garch_beta']
            
            # CALL VECTORIZED NUMBA FUNCTION
            volatility_paths = mc_garch_vectorized(
                omega=omega,
                alpha=alpha,
                beta=beta,
                sigma_0=start_vol,
                num_simulations=num_simulations,
                num_days=num_days,
                random_seed=random_seed + date_idx  # Different seed per date
            )
            
            firm_sims += num_simulations
            total_sims_run += num_simulations
            
            # Extract final volatilities and annualize
            final_vols_daily = volatility_paths[:, num_days]
            final_vols_annualized = final_vols_daily * np.sqrt(252)
            
            # Compute statistics
            mean_final_vol = final_vols_annualized.mean()
            std_final_vol = final_vols_annualized.std()
            percentile_5 = np.percentile(final_vols_annualized, 5)
            percentile_95 = np.percentile(final_vols_annualized, 95)
            
            # Batch append (faster than appending to all_results each time)
            firm_results_batch.append({
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
                'percentile_95_annualized': percentile_95,
                'num_simulations': num_simulations
            })
            
            # Progress update every 50 dates
            if (date_idx + 1) % 50 == 0 or date_idx == num_dates - 1:
                pct_complete = ((date_idx + 1) / num_dates) * 100
                print(f"  Progress: {date_idx + 1:,}/{num_dates:,} dates ({pct_complete:.1f}%) | "
                      f"Sims: {firm_sims:,}")
        
        # Add all firm results at once (batch append is faster)
        all_results.extend(firm_results_batch)
        
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
    print(f"  Total result rows: {len(results_df):,}")
    print(f"  Firms processed: {results_df['gvkey'].nunique()}")
    
    if len(results_df) > 0:
        print(f"  Date range: {results_df['start_date'].min().strftime('%Y-%m-%d')} to {results_df['start_date'].max().strftime('%Y-%m-%d')}")
        print(f"  Mean 252D volatility: {results_df['mean_final_volatility_annualized'].min():.4f} - {results_df['mean_final_volatility_annualized'].max():.4f}")
    
    print(f"{'='*80}\n")
    
    # Cache results
    cache_file = './intermediates/mc_garch_cache.csv'
    results_df.to_csv(cache_file, index=False)
    print(f"✓ Cached results to: {cache_file}\n")
    
    return results_df