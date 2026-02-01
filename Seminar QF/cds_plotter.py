import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class CDSPlotter:
    """
    A class to visualize CDS spreads over time for different maturities and models.
    Designed to be extensible for other models.
    """
    
    def __init__(self):
        """
        Initialize the CDSPlotter.
        """
        self.data = {} # Stores DataFrames keyed by model_name
        self.model_colors = {
            'GARCH': 'blue',
            'Regime Switching': 'green',
            'Merton': 'red',
            'Msgarch': 'purple'
        }
        # Standard matplotlib settings
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12

    def load_data(self, model_name, file_path):
        """
        Load CDS spread data for a specific model from a CSV file.
        
        Parameters:
        -----------
        model_name : str
            Name of the model (e.g., 'GARCH', 'Regime Switching')
        file_path : str
            Path to the CSV file containing the results
        """
        print(f"Loading data for {model_name} from {file_path}...")
        try:
            df = pd.read_csv(file_path)
            
            # Standardize date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'])
            
            # Identify columns for maturities
            # We expect columns like 'cds_spread_..._1y_bps' or similar
            # Logic to rename/map columns to a standard internal format if needed
            # For now, we'll try to identify the maturity columns dynamically or rename them
            
            # Fix for potential mislabeled columns (e.g. regime switching file having 'garch' in headers)
            # We will rename columns to a standard format: 'spread_{maturity}y_bps'
            
            rename_map = {}
            for col in df.columns:
                if 'bps' in col:
                    if '1y' in col:
                        rename_map[col] = 'spread_1y_bps'
                    elif '3y' in col:
                        rename_map[col] = 'spread_3y_bps'
                    elif '5y' in col:
                        rename_map[col] = 'spread_5y_bps'
            
            if rename_map:
                df = df.rename(columns=rename_map)
                print(f"  Mapped columns: {list(rename_map.values())}")
            
            self.data[model_name] = df
            print(f"  Successfully loaded {len(df)} rows for {model_name}.")
            
        except Exception as e:
            print(f"  Error loading data: {e}")

    def plot_spreads_over_time(self, model_name, gvkey, maturities=[1, 3, 5], title=None):
        """
        Plot CDS spreads over time for a single firm and model, showing multiple maturities.
        
        Parameters:
        -----------
        model_name : str
            The model to plot (must be loaded first)
        gvkey : int
            The firm identifier to filter by
        maturities : list
            List of maturities to plot (e.g. [1, 3, 5])
        """
        if model_name not in self.data:
            print(f"Error: Model '{model_name}' not loaded.")
            return

        df = self.data[model_name]
        df_firm = df[df['gvkey'] == gvkey].sort_values('date')
        
        if df_firm.empty:
            print(f"No data found for firm {gvkey} in model {model_name}")
            return

        plt.figure(figsize=(12, 6))
        
        # Color map for maturities
        # lighter versions of the model color or distinct colors? 
        # distinct colors for maturities is better for a single model plot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green standard
        
        for i, mat in enumerate(maturities):
            col_name = f'spread_{mat}y_bps'
            if col_name in df_firm.columns:
                plt.plot(df_firm['date'], df_firm[col_name], 
                         label=f'{mat}-Year Maturity', linewidth=1.5,
                         alpha=0.8)
            else:
                print(f"Warning: Column {col_name} not found in data.")

        term_title = title if title else f"{model_name} CDS Spreads Over Time (Firm {gvkey})"
        plt.title(term_title)
        plt.xlabel("Date")
        plt.ylabel("CDS Spread (bps)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, gvkey, maturity=5, models=None, title=None):
        """
        Compare different models for a specific maturity and firm.
        
        Parameters:
        -----------
        gvkey : int
            The firm identifier
        maturity : int
            The maturity to compare (default 5 years)
        models : list
            List of model names to compare. If None, uses all loaded models.
        """
        if models is None:
            models = list(self.data.keys())
            
        plt.figure(figsize=(12, 6))
        
        col_name = f'spread_{maturity}y_bps'
        
        has_data = False
        for model in models:
            if model not in self.data:
                continue
                
            df = self.data[model]
            df_firm = df[df['gvkey'] == gvkey].sort_values('date')
            
            if col_name in df_firm.columns and not df_firm.empty:
                plt.plot(df_firm['date'], df_firm[col_name], 
                         label=f'{model}', linewidth=1.5, alpha=0.8)
                has_data = True
        
        if not has_data:
            print(f"No data found for firm {gvkey} and maturity {maturity}y")
            plt.close()
            return

        plot_title = title if title else f"{maturity}-Year CDS Spread Model Comparison (Firm {gvkey})"
        plt.title(plot_title)
        plt.xlabel("Date")
        plt.ylabel("CDS Spread (bps)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_average_spreads_over_time(self, model_name, maturities=[1, 3, 5], title=None, aggregation='mean'):
        """
        Plot the aggregated (mean or median) CDS spreads across all firms over time for a model.
        
        Parameters:
        -----------
        model_name : str
            The model to plot (must be loaded first)
        maturities : list
            List of maturities to plot (e.g. [1, 3, 5])
        title : str, optional
            Custom title for the plot
        aggregation : str, optional
            Method to aggregate across firms: 'mean' or 'median'. Default is 'mean'.
            'median' is recommended if there are outliers.
        """
        if model_name not in self.data:
            print(f"Error: Model '{model_name}' not loaded.")
            return

        df = self.data[model_name]
        
        # Group by date and calculate aggregate
        if aggregation == 'median':
            df_daily_agg = df.groupby('date').median(numeric_only=True)
            agg_label = 'Median'
        else:
            df_daily_agg = df.groupby('date').mean(numeric_only=True)
            agg_label = 'Mean'
        
        plt.figure(figsize=(12, 6))
        
        for i, mat in enumerate(maturities):
            col_name = f'spread_{mat}y_bps'
            if col_name in df_daily_agg.columns:
                plt.plot(df_daily_agg.index, df_daily_agg[col_name], 
                         label=f'{mat}-Year Maturity ({agg_label})', linewidth=1.5,
                         alpha=0.8)
            else:
                print(f"Warning: Column {col_name} not found in data.")

        term_title = title if title else f"{model_name} {agg_label} CDS Spreads Over Time (All Firms)"
        plt.title(term_title)
        plt.xlabel("Date")
        plt.ylabel(f"{agg_label} CDS Spread (bps)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Example usage helper
if __name__ == "__main__":
    # Small test if run directly
    plotter = CDSPlotter()
    # Paths assumed relative to where script is run
    # plotter.load_data('GARCH', 'cds_spreads_garch_mc.csv')
    # plotter.load_data('Regime Switching', 'cds_spreads_regime_switching_mc.csv')
