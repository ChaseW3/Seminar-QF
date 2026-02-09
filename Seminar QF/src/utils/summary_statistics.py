import pandas as pd
import numpy as np

def calculate_summary_statistics(df, return_col='asset_return_daily'):
    """
    Calculate summary statistics for the dataset.

    Args:
        df (pd.DataFrame): Input dataframe containing firm data and returns.
        return_col (str): Column name for asset returns.

    Returns:
        pd.DataFrame: A dataframe containing the calculated statistics.
    """
    stats = {}

    # 1. Number of firms
    if 'gvkey' in df.columns:
        stats['Number of Firms'] = df['gvkey'].nunique()
    
    # 2. Sample period
    if 'date' in df.columns:
        stats['Sample Period Start'] = df['date'].min()
        stats['Sample Period End'] = df['date'].max()

    # 3. Firm-day observations
    stats['Firm-Day Observations'] = len(df)

    # 4. Statistics for asset returns
    if return_col in df.columns:
        stats['Mean Return'] = df[return_col].mean()
        stats['Median Return'] = df[return_col].median()
        stats['Std Dev Return'] = df[return_col].std()
        stats['Skewness Return'] = df[return_col].skew()
        stats['Kurtosis Return'] = df[return_col].kurtosis()
        stats['Min Return'] = df[return_col].min()
        stats['Max Return'] = df[return_col].max()
    else:
        print(f"Warning: Column '{return_col}' not found in dataframe. Statistics for returns will be missing.")

    # Convert to DataFrame for nicer display
    stats_df = pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])
    
    return stats_df
