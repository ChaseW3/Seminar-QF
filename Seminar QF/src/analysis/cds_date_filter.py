import pandas as pd


def load_allowed_cds_dates(cds_filter_file: str, use_column: str = 'Use_For_Model') -> pd.DatetimeIndex:
    """Load allowed simulation dates from CDS quality-screen output.

    Expected file from notebook: cds_date_level_model_use_flag.csv
    Required columns:
      - Date (or date)
      - Use_For_Model (default), optionally bool-like values
    """
    df = pd.read_csv(cds_filter_file)

    date_col = None
    for candidate in ['Date', 'date']:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError(f"{cds_filter_file} must contain 'Date' or 'date' column")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df[df[date_col].notna()].copy()

    if use_column in df.columns:
        use_vals = df[use_column]
        if use_vals.dtype != bool:
            use_vals = (
                use_vals.astype(str)
                .str.strip()
                .str.lower()
                .isin(['true', '1', 'yes', 'y'])
            )
        df = df[use_vals]

    if df.empty:
        return pd.DatetimeIndex([])

    allowed_dates = pd.DatetimeIndex(df[date_col].dt.normalize().drop_duplicates().sort_values())
    return allowed_dates


def filter_df_to_allowed_dates(df: pd.DataFrame, allowed_dates: pd.DatetimeIndex, date_col: str = 'date') -> pd.DataFrame:
    """Filter a dataframe to allowed normalized dates."""
    if date_col not in df.columns:
        return df
    if len(allowed_dates) == 0:
        return df.iloc[0:0].copy()

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
    work = work[work[date_col].notna()]
    return work[work[date_col].dt.normalize().isin(allowed_dates)].copy()
