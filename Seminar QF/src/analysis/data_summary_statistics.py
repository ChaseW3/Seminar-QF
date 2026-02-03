"""
Data Summary Statistics Generator
==================================
Generates comprehensive summary statistics for the raw input data:
- Equity data (market cap, shares, prices)
- Liability data (balance sheet)
- CDS spreads (market data)

Produces both firm-level and cross-sectional statistics suitable for academic reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import config


def load_equity_data():
    """Load and process equity market data."""
    print("Loading equity data...")
    df = pd.read_excel(config.EQUITY_DATA_FILE, sheet_name=0)
    
    df = df.rename(columns={
        "(gvkey) Global Company Key - Company": "gvkey",
        "(datadate) Data Date - Daily Prices": "date",
        "(conm) Company Name": "company_name",
        "(fic) Current ISO Country Code - Incorporation": "country",
        "(cshoc) Shares Outstanding": "shares_outstanding",
        "(prccd) Price - Close - Daily": "price_close",
        "Market Capitalization (# Shares * Close Price)": "market_cap"
    })
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    
    return df


def load_liability_data():
    """Load and process liability data."""
    print("Loading liability data...")
    df = pd.read_excel(config.EQUITY_DATA_FILE, sheet_name=1)
    
    df = df.rename(columns={
        "(gvkey) Global Company Key - Company": "gvkey",
        "(fyear) Data Year - Fiscal": "fyear",
        "(conm) Company Name": "company_name",
        "(fic) Current ISO Country Code - Incorporation": "country",
        "(lt) Liabilities - Total": "liabilities_total",
        "(datadate) Data Date": "datadate"
    })
    
    # Convert liabilities from millions to billions for reporting
    df['liabilities_billions'] = df['liabilities_total'] / 1000
    
    return df


def load_cds_data():
    """Load and process CDS spread data."""
    print("Loading CDS spread data...")
    
    cds_files = {
        '1Y': config.DATA_DIR / 'input' / 'CDS_1y_mat_data.xlsx',
        '3Y': config.DATA_DIR / 'input' / 'CDS_3y_mat_data.xlsx',
        '5Y': config.DATA_DIR / 'input' / 'CDS_5y_mat_data.xlsx'
    }
    
    cds_data = {}
    
    for maturity, file_path in cds_files.items():
        try:
            df = pd.read_excel(file_path, skiprows=1)
            # First column is date
            df = df.rename(columns={df.columns[0]: 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Melt to long format
            df_long = df.reset_index().melt(id_vars=['date'], var_name='company', value_name='cds_spread')
            df_long['maturity'] = maturity
            df_long['cds_spread'] = pd.to_numeric(df_long['cds_spread'], errors='coerce')
            
            cds_data[maturity] = df_long
        except Exception as e:
            print(f"  Warning: Could not load {maturity} CDS data: {e}")
    
    if cds_data:
        return pd.concat(cds_data.values(), ignore_index=True)
    else:
        return None


def generate_equity_summary(df_equity):
    """Generate summary statistics for equity data."""
    print("\n" + "="*80)
    print("EQUITY DATA SUMMARY STATISTICS")
    print("="*80)
    
    # Panel structure
    print("\n1. PANEL STRUCTURE")
    print("-" * 80)
    n_firms = df_equity['gvkey'].nunique()
    n_obs = len(df_equity)
    date_range = f"{df_equity['date'].min().date()} to {df_equity['date'].max().date()}"
    n_countries = df_equity['country'].nunique()
    
    print(f"Number of firms: {n_firms}")
    print(f"Number of observations: {n_obs:,}")
    print(f"Date range: {date_range}")
    print(f"Countries: {n_countries}")
    print(f"Avg observations per firm: {n_obs/n_firms:.0f}")
    
    # Country breakdown
    print("\nFirms by country:")
    country_counts = df_equity.groupby('country')['gvkey'].nunique().sort_values(ascending=False)
    for country, count in country_counts.items():
        print(f"  {country}: {count} firms")
    
    # Time series structure
    print("\n2. TIME SERIES COVERAGE")
    print("-" * 80)
    yearly_stats = df_equity.groupby('year').agg({
        'gvkey': 'nunique',
        'market_cap': ['count', 'mean', 'median']
    }).round(2)
    print(yearly_stats.to_string())
    
    # Cross-sectional statistics
    print("\n3. CROSS-SECTIONAL STATISTICS (Market Capitalization)")
    print("-" * 80)
    
    # Overall statistics (in billions)
    market_cap_billions = df_equity['market_cap'] / 1e9
    
    stats_df = pd.DataFrame({
        'Mean': [market_cap_billions.mean()],
        'Median': [market_cap_billions.median()],
        'Std': [market_cap_billions.std()],
        'Min': [market_cap_billions.min()],
        'p25': [market_cap_billions.quantile(0.25)],
        'p75': [market_cap_billions.quantile(0.75)],
        'Max': [market_cap_billions.max()],
        'N': [len(market_cap_billions)]
    })
    
    print("\nAll observations (EUR billions):")
    print(stats_df.round(2).to_string(index=False))
    
    # Firm-level averages (time-series average for each firm)
    print("\n4. FIRM-LEVEL AVERAGES (Time-series mean per firm, EUR billions)")
    print("-" * 80)
    firm_avg = df_equity.groupby('gvkey')['market_cap'].mean() / 1e9
    
    firm_stats = pd.DataFrame({
        'Mean': [firm_avg.mean()],
        'Median': [firm_avg.median()],
        'Std': [firm_avg.std()],
        'Min': [firm_avg.min()],
        'p25': [firm_avg.quantile(0.25)],
        'p75': [firm_avg.quantile(0.75)],
        'Max': [firm_avg.max()],
        'N': [len(firm_avg)]
    })
    
    print(firm_stats.round(2).to_string(index=False))
    
    # Top 10 firms by average market cap
    print("\n5. TOP 10 FIRMS BY AVERAGE MARKET CAPITALIZATION")
    print("-" * 80)
    top10 = df_equity.groupby(['gvkey', 'company_name']).agg({
        'market_cap': 'mean',
        'date': 'count'
    }).reset_index()
    top10.columns = ['gvkey', 'company_name', 'avg_market_cap', 'n_obs']
    top10['avg_market_cap_billions'] = top10['avg_market_cap'] / 1e9
    top10 = top10.sort_values('avg_market_cap', ascending=False).head(10)
    
    print(top10[['company_name', 'avg_market_cap_billions', 'n_obs']].to_string(index=False))
    
    return {
        'n_firms': n_firms,
        'n_obs': n_obs,
        'date_range': date_range,
        'cross_sectional': stats_df,
        'firm_level': firm_stats
    }


def generate_liability_summary(df_liability):
    """Generate summary statistics for liability data."""
    print("\n" + "="*80)
    print("LIABILITY DATA SUMMARY STATISTICS")
    print("="*80)
    
    # Panel structure
    print("\n1. PANEL STRUCTURE")
    print("-" * 80)
    n_firms = df_liability['gvkey'].nunique()
    n_obs = len(df_liability)
    year_range = f"{df_liability['fyear'].min()} to {df_liability['fyear'].max()}"
    n_countries = df_liability['country'].nunique()
    
    print(f"Number of firms: {n_firms}")
    print(f"Number of firm-year observations: {n_obs}")
    print(f"Fiscal year range: {year_range}")
    print(f"Countries: {n_countries}")
    print(f"Avg years per firm: {n_obs/n_firms:.1f}")
    
    # Missing data analysis
    print("\n2. DATA COMPLETENESS")
    print("-" * 80)
    missing = df_liability['liabilities_total'].isna().sum()
    print(f"Missing liability values: {missing} ({100*missing/n_obs:.1f}%)")
    
    # Check for zero or negative values
    zero_or_neg = (df_liability['liabilities_total'] <= 0).sum()
    print(f"Zero or negative liabilities: {zero_or_neg} ({100*zero_or_neg/n_obs:.1f}%)")
    
    # Cross-sectional statistics
    print("\n3. CROSS-SECTIONAL STATISTICS (Total Liabilities)")
    print("-" * 80)
    
    # Filter valid observations
    valid = df_liability[df_liability['liabilities_total'] > 0].copy()
    liab_billions = valid['liabilities_billions']
    
    stats_df = pd.DataFrame({
        'Mean': [liab_billions.mean()],
        'Median': [liab_billions.median()],
        'Std': [liab_billions.std()],
        'Min': [liab_billions.min()],
        'p25': [liab_billions.quantile(0.25)],
        'p75': [liab_billions.quantile(0.75)],
        'Max': [liab_billions.max()],
        'N': [len(liab_billions)]
    })
    
    print("\nAll observations (EUR billions):")
    print(stats_df.round(2).to_string(index=False))
    
    # Firm-level averages
    print("\n4. FIRM-LEVEL AVERAGES (Time-series mean per firm, EUR billions)")
    print("-" * 80)
    firm_avg = valid.groupby('gvkey')['liabilities_billions'].mean()
    
    firm_stats = pd.DataFrame({
        'Mean': [firm_avg.mean()],
        'Median': [firm_avg.median()],
        'Std': [firm_avg.std()],
        'Min': [firm_avg.min()],
        'p25': [firm_avg.quantile(0.25)],
        'p75': [firm_avg.quantile(0.75)],
        'Max': [firm_avg.max()],
        'N': [len(firm_avg)]
    })
    
    print(firm_stats.round(2).to_string(index=False))
    
    # Top 10 firms by average liabilities
    print("\n5. TOP 10 FIRMS BY AVERAGE TOTAL LIABILITIES")
    print("-" * 80)
    top10 = valid.groupby(['gvkey', 'company_name']).agg({
        'liabilities_billions': 'mean',
        'fyear': 'count'
    }).reset_index()
    top10.columns = ['gvkey', 'company_name', 'avg_liabilities_billions', 'n_years']
    top10 = top10.sort_values('avg_liabilities_billions', ascending=False).head(10)
    
    print(top10[['company_name', 'avg_liabilities_billions', 'n_years']].to_string(index=False))
    
    # Time trend
    print("\n6. TIME TREND (Average by fiscal year)")
    print("-" * 80)
    time_trend = valid.groupby('fyear')['liabilities_billions'].agg(['count', 'mean', 'median']).round(2)
    print(time_trend.to_string())
    
    return {
        'n_firms': n_firms,
        'n_obs': n_obs,
        'year_range': year_range,
        'cross_sectional': stats_df,
        'firm_level': firm_stats
    }


def generate_cds_summary(df_cds):
    """Generate summary statistics for CDS spread data."""
    print("\n" + "="*80)
    print("CDS SPREAD DATA SUMMARY STATISTICS")
    print("="*80)
    
    if df_cds is None or len(df_cds) == 0:
        print("No CDS data available")
        return None
    
    # Panel structure
    print("\n1. PANEL STRUCTURE")
    print("-" * 80)
    n_companies = df_cds['company'].nunique()
    n_obs_total = len(df_cds)
    n_obs_valid = df_cds['cds_spread'].notna().sum()
    date_range = f"{df_cds['date'].min().date()} to {df_cds['date'].max().date()}"
    
    print(f"Number of companies: {n_companies}")
    print(f"Total observations: {n_obs_total:,}")
    print(f"Valid observations (non-missing): {n_obs_valid:,} ({100*n_obs_valid/n_obs_total:.1f}%)")
    print(f"Date range: {date_range}")
    
    # By maturity
    print("\n2. COVERAGE BY MATURITY")
    print("-" * 80)
    by_maturity = df_cds.groupby('maturity').agg({
        'cds_spread': lambda x: x.notna().sum(),
        'company': 'nunique'
    })
    by_maturity.columns = ['Valid observations', 'Companies']
    print(by_maturity.to_string())
    
    # Cross-sectional statistics by maturity
    print("\n3. CROSS-SECTIONAL STATISTICS BY MATURITY (Basis Points)")
    print("-" * 80)
    
    for maturity in sorted(df_cds['maturity'].unique()):
        print(f"\n{maturity} Maturity:")
        print("-" * 40)
        
        mat_data = df_cds[df_cds['maturity'] == maturity]['cds_spread'].dropna()
        
        if len(mat_data) == 0:
            print("No valid data")
            continue
        
        stats_df = pd.DataFrame({
            'Mean': [mat_data.mean()],
            'Median': [mat_data.median()],
            'Std': [mat_data.std()],
            'Min': [mat_data.min()],
            'p25': [mat_data.quantile(0.25)],
            'p75': [mat_data.quantile(0.75)],
            'Max': [mat_data.max()],
            'N': [len(mat_data)]
        })
        
        print(stats_df.round(2).to_string(index=False))
    
    # Time series characteristics
    print("\n4. TIME SERIES CHARACTERISTICS")
    print("-" * 80)
    
    # Company-level statistics (time-series average per company)
    for maturity in sorted(df_cds['maturity'].unique()):
        mat_data = df_cds[df_cds['maturity'] == maturity]
        company_avg = mat_data.groupby('company')['cds_spread'].mean().dropna()
        
        if len(company_avg) == 0:
            continue
            
        print(f"\n{maturity} - Company-level averages (basis points):")
        firm_stats = pd.DataFrame({
            'Mean': [company_avg.mean()],
            'Median': [company_avg.median()],
            'Std': [company_avg.std()],
            'Min': [company_avg.min()],
            'p25': [company_avg.quantile(0.25)],
            'p75': [company_avg.quantile(0.75)],
            'Max': [company_avg.max()],
            'N': [len(company_avg)]
        })
        print(firm_stats.round(2).to_string(index=False))
    
    # Extreme values
    print("\n5. EXTREME VALUES (>1000 bps)")
    print("-" * 80)
    extreme = df_cds[df_cds['cds_spread'] > 1000]
    if len(extreme) > 0:
        extreme_summary = extreme.groupby('maturity')['cds_spread'].agg(['count', 'mean', 'min', 'max'])
        print(extreme_summary.round(2).to_string())
    else:
        print("No CDS spreads above 1000 bps")
    
    return {
        'n_companies': n_companies,
        'n_obs': n_obs_total,
        'date_range': date_range
    }


def save_latex_tables(equity_stats, liability_stats, cds_stats=None):
    """Save summary statistics as LaTeX tables for the report."""
    print("\n" + "="*80)
    print("GENERATING LATEX TABLES")
    print("="*80)
    
    output_dir = config.OUTPUT_DIR / 'summary_statistics'
    output_dir.mkdir(exist_ok=True)
    
    # Table 1: Equity market data
    if equity_stats:
        latex_equity = equity_stats['cross_sectional'].T.to_latex(
            header=['All Observations'],
            float_format='%.2f',
            caption='Summary Statistics: Market Capitalization (EUR billions)',
            label='tab:equity_summary'
        )
        
        with open(output_dir / 'table_equity_summary.tex', 'w') as f:
            f.write(latex_equity)
        print(f"✓ Saved: table_equity_summary.tex")
    
    # Table 2: Liability data
    if liability_stats:
        latex_liability = liability_stats['cross_sectional'].T.to_latex(
            header=['All Observations'],
            float_format='%.2f',
            caption='Summary Statistics: Total Liabilities (EUR billions)',
            label='tab:liability_summary'
        )
        
        with open(output_dir / 'table_liability_summary.tex', 'w') as f:
            f.write(latex_liability)
        print(f"✓ Saved: table_liability_summary.tex")
    
    print(f"\nLaTeX tables saved to: {output_dir}")


def main():
    """Main execution function."""
    print("="*80)
    print("DATA SUMMARY STATISTICS GENERATOR")
    print("="*80)
    
    # Load data
    df_equity = load_equity_data()
    df_liability = load_liability_data()
    df_cds = load_cds_data()
    
    # Generate summaries
    equity_stats = generate_equity_summary(df_equity)
    liability_stats = generate_liability_summary(df_liability)
    cds_stats = generate_cds_summary(df_cds)
    
    # Save LaTeX tables
    save_latex_tables(equity_stats, liability_stats, cds_stats)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS GENERATION COMPLETE")
    print("="*80)
    print("\nUse these statistics in your report to describe:")
    print("  1. Sample composition (firms, countries, time period)")
    print("  2. Cross-sectional variation in firm size")
    print("  3. Time-series coverage and trends")
    print("  4. Data quality (completeness, outliers)")


if __name__ == "__main__":
    main()
