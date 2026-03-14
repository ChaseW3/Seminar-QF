import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class CDSPlotter:
    
    def __init__(self):
        self.data = {} 
        self.model_colors = {
            'GARCH': 'blue',
            'Regime Switching': 'green',
            'Merton': 'red',
            'Msgarch': 'purple'
        }
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12

    def load_data(self, model_name, file_path):
        try:
            df = pd.read_csv(file_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'])
            
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
            
            self.data[model_name] = df
            
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    def plot_spreads_over_time(self, model_name, gvkey, maturities=[1, 3, 5], title=None):
        if model_name not in self.data:
            return

        df = self.data[model_name]
        df_firm = df[df['gvkey'] == gvkey].sort_values('date')
        
        if df_firm.empty:
            return

        plt.figure(figsize=(12, 6))
        
        for i, mat in enumerate(maturities):
            col_name = f'spread_{mat}y_bps'
            if col_name in df_firm.columns:
                plt.plot(df_firm['date'], df_firm[col_name], 
                         label=f'{mat}-Year Maturity', linewidth=1.5,
                         alpha=0.8)

        term_title = title if title else f"{model_name} CDS Spreads Over Time (Firm {gvkey})"
        plt.title(term_title)
        plt.xlabel("Date")
        plt.ylabel("CDS Spread (bps)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_model_comparison(self, gvkey, maturity=5, models=None, title=None):
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

    def plot_average_spreads_over_time(self, model_name, maturities=[1, 3, 5], title=None, aggregation='median'):
        if model_name not in self.data:
            return

        df = self.data[model_name]
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

        term_title = title if title else f"{model_name} {agg_label} CDS Spreads Over Time (All Firms)"
        plt.title(term_title)
        plt.xlabel("Date")
        plt.ylabel(f"{agg_label} CDS Spread (bps)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    plotter = CDSPlotter()