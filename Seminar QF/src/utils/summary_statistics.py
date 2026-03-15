import pandas as pd
import numpy as np

def calculate_summary_statistics(df, return_col='asset_return_daily'):
    stats = {}

    if 'gvkey' in df.columns:
        stats['Number of Firms'] = df['gvkey'].nunique()
    
    if 'date' in df.columns:
        stats['Sample Period Start'] = df['date'].min()
        stats['Sample Period End'] = df['date'].max()

    stats['Firm-Day Observations'] = len(df)

    if return_col in df.columns:
        stats['Mean Return'] = df[return_col].mean()
        stats['Median Return'] = df[return_col].median()
        stats['Std Dev Return'] = df[return_col].std()
        stats['Skewness Return'] = df[return_col].skew()
        stats['Kurtosis Return'] = df[return_col].kurtosis()
        stats['Min Return'] = df[return_col].min()
        stats['Max Return'] = df[return_col].max()

    stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    
    return stats_df
