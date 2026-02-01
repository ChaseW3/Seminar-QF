import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_results_summary(results_file="monthly_pd_results.csv"):
    """
    Reads the PD results file, prints summary statistics, and generates a plot
    of the average Probability of Default (PD) over time.
    """
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found.")
        return

    print(f"Generating summary from {results_file}...")
    
    df = pd.read_csv(results_file)
    
    # Ensure month_year is datetime
    df['month_year'] = pd.to_datetime(df['month_year'])
    
    # Calculate Leverage (Liabilities / Assets)
    if 'liabilities_total' in df.columns and 'asset_value' in df.columns:
        df['leverage_ratio'] = df['liabilities_total'] / df['asset_value']

    # --- 1. Extended Summary Statistics ---
    cols_to_summarize = ['merton_pd', 'asset_volatility', 'asset_return_monthly', 'risk_free_rate', 'leverage_ratio']
    # Filter for columns that actually exist
    cols_to_summarize = [c for c in cols_to_summarize if c in df.columns]
    
    print("\n--- Descriptive Statistics ---")
    desc_stats = df[cols_to_summarize].describe()
    print(desc_stats)

    # --- 2. Correlation Matrix ---
    print("\n--- Correlation Matrix ---")
    corr_matrix = df[cols_to_summarize].corr()
    print(corr_matrix)
    
    # --- 3. Plotting ---
    
    # Plot 1: Average Merton PD Over Time (Separate Plots per Model)
    
    # Check for wide-format result columns (e.g. merton_pd_garch)
    pd_cols = [c for c in df.columns if 'merton_pd_' in c]
    
    if pd_cols:
        model_colors = {
            'GARCH': '#1f77b4',      # Blue
            'REGIME': '#ff7f0e',     # Orange
            'MSGARCH': '#2ca02c',    # Green
        }

        for col in pd_cols:
            model_suffix = col.replace('merton_pd_', '').upper()
            
            # Create a new figure for EACH model
            plt.figure(figsize=(10, 6))
            
            # Group by date and calculate mean
            avg_pd_over_time = df.groupby('month_year')[col].mean()
            
            color = model_colors.get(model_suffix, 'blue')
            
            plt.plot(avg_pd_over_time.index, avg_pd_over_time.values, 
                     label=f'Avg PD ({model_suffix})', color=color, linewidth=2.5)
            
            plt.title(f'Average Merton Probability of Default ({model_suffix} Model)')
            plt.xlabel('Date')
            plt.ylabel('Probability of Default')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            
            # Save separate files
            filename = f"average_pd_over_time_{model_suffix.lower()}.png"
            plt.savefig(filename)
            print(f"Saved plot: {filename}")
            plt.show()

    elif 'model' in df.columns:
        # Plot each model separately (long format)
        for model_name in df['model'].unique():
            plt.figure(figsize=(10, 6))
            subset = df[df['model'] == model_name]
            avg_pd_over_time = subset.groupby('month_year')['merton_pd'].mean()
            plt.plot(avg_pd_over_time.index, avg_pd_over_time.values, label=f'Avg PD ({model_name})')
            
            plt.title(f'Average Merton PD ({model_name})')
            plt.xlabel('Date')
            plt.ylabel('Probability of Default')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"average_pd_over_time_{model_name}.png")
            plt.show()
            
    elif 'merton_pd' in df.columns:
        # Fallback for single model result without 'model' column
        avg_pd_over_time = df.groupby('month_year')['merton_pd'].mean()
        plt.figure(figsize=(10, 6))
        plt.plot(avg_pd_over_time.index, avg_pd_over_time.values, label='Average Merton PD', color='blue')
        plt.title('Average Merton Probability of Default Over Time')
        plt.xlabel('Date')
        plt.ylabel('Probability of Default')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("average_pd_over_time.png")
        plt.show() 

    
    # Plot 2: MS-GARCH Regime Probability Over Time (if available)
    if 'msgarch_prob_0' in df.columns:
        avg_prob0_over_time = df.groupby('month_year')['msgarch_prob_0'].mean()
        
        plt.figure(figsize=(10, 6))
        plt.plot(avg_prob0_over_time.index, avg_prob0_over_time.values, label='Avg Prob State 0 (Stable)', color='green')
        plt.title('Average MS-GARCH Regime Probability (State 0) Over Time')
        plt.xlabel('Date')
        plt.ylabel('Probability of State 0')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("regime_probability_over_time.png")
        print(f"\nPlot saved to regime_probability_over_time.png")
        plt.show() # Show second plot

    # Plot 3: Asset Volatility vs Risk Free Rate (scatter)
    # Check for any volatility column
    vol_cols = [c for c in df.columns if 'volatility' in c and 'asset' not in c]
    if 'asset_volatility' in df.columns:
         y_col = 'asset_volatility'
         y_label = 'Asset Volatility'
    elif vol_cols:
         y_col = vol_cols[0] # Take first available model volatility
         y_label = f'{y_col} (Model)'
    else:
         y_col = None

    if y_col and 'risk_free_rate' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['risk_free_rate'], df[y_col], alpha=0.5, s=10)
        plt.title(f'{y_label} vs Risk Free Rate')
        plt.xlabel('Risk Free Rate')
        plt.ylabel(y_label)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("volatility_vs_rate.png")
        print(f"\nPlot saved to volatility_vs_rate.png")
        plt.show()


    # --- 4. Save Text Report ---
    with open("summary_report.txt", "w") as f:
        f.write("Results Summary Report\n")
        f.write("======================\n\n")
        
        f.write("1. Descriptive Statistics:\n")
        f.write(desc_stats.to_string())
        f.write("\n\n")
        
        f.write("2. Correlation Matrix:\n")
        f.write(corr_matrix.to_string())
        f.write("\n\n")
        
        if 'msgarch_prob_0' in df.columns:
             f.write("3. MS-GARCH Statistics:\n")
             f.write(df['msgarch_prob_0'].describe().to_string())
             f.write("\n")
    
    print("Full summary report saved to summary_report.txt")
