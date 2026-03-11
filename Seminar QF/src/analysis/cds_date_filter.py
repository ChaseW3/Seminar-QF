import pandas as pd
from typing import Union

# cds_date_filter.py
# Utility functions for filtering simulation dates to match CDS data availability.
# Supports several input formats: date-only, firm-date pairs, and firm-window files.


def _normalize_gvkey_key(value) -> str:
    # Normalize gvkey values to consistent string keys across int/float/string inputs
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if text == '':
        return ''
    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass
    return text


def _maturity_to_horizon_days(value):
    # Map maturity labels (1Y/3Y/5Y) to trading-day horizons (252/756/1260)
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {'1', '1Y', '12M'}:
        return 252
    if text in {'3', '3Y', '36M'}:
        return 756
    if text in {'5', '5Y', '60M'}:
        return 1260
    return None


def load_allowed_cds_dates(cds_filter_file: str, use_column: str = 'Use_For_Model') -> Union[pd.DatetimeIndex, pd.DataFrame]:
    # Load allowed simulation dates from a CDS quality-screen file.
    # Supports multiple formats: date-level, firm-date, firm-window, and firm-window-by-maturity.
    df = pd.read_csv(cds_filter_file)

    if {'gvkey', 'maturity', 'start_date', 'end_date'}.issubset(df.columns):
        out = df[['gvkey', 'maturity', 'start_date', 'end_date']].copy()
        out['gvkey'] = out['gvkey'].map(_normalize_gvkey_key)
        out['maturity'] = out['maturity'].astype(str).str.upper().str.strip()
        out['horizon_days'] = out['maturity'].map(_maturity_to_horizon_days)
        out['start_date'] = pd.to_datetime(out['start_date'], errors='coerce').dt.normalize()
        out['end_date'] = pd.to_datetime(out['end_date'], errors='coerce').dt.normalize()
        out = out[
            (out['gvkey'] != '')
            & out['start_date'].notna()
            & out['end_date'].notna()
            & out['horizon_days'].notna()
        ].copy()
        out = out[out['end_date'] >= out['start_date']].drop_duplicates().reset_index(drop=True)
        out['horizon_days'] = out['horizon_days'].astype(int)
        return out

    if {'gvkey', 'start_date', 'end_date'}.issubset(df.columns):
        out = df[['gvkey', 'start_date', 'end_date']].copy()
        out['gvkey'] = out['gvkey'].map(_normalize_gvkey_key)
        out['start_date'] = pd.to_datetime(out['start_date'], errors='coerce').dt.normalize()
        out['end_date'] = pd.to_datetime(out['end_date'], errors='coerce').dt.normalize()
        out = out[(out['gvkey'] != '') & out['start_date'].notna() & out['end_date'].notna()].copy()
        out = out[out['end_date'] >= out['start_date']].drop_duplicates().reset_index(drop=True)
        return out

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

    if 'gvkey' in df.columns:
        allowed_pairs = df[[date_col, 'gvkey']].copy()
        allowed_pairs[date_col] = pd.to_datetime(allowed_pairs[date_col], errors='coerce').dt.normalize()
        allowed_pairs = allowed_pairs[allowed_pairs[date_col].notna() & allowed_pairs['gvkey'].notna()].copy()
        allowed_pairs['gvkey'] = allowed_pairs['gvkey'].map(_normalize_gvkey_key)
        allowed_pairs = allowed_pairs[allowed_pairs['gvkey'] != '']
        allowed_pairs = allowed_pairs.drop_duplicates().reset_index(drop=True)
        return allowed_pairs

    allowed_dates = pd.DatetimeIndex(df[date_col].dt.normalize().drop_duplicates().sort_values())
    return allowed_dates


def filter_df_to_allowed_dates(df: pd.DataFrame, allowed_dates: Union[pd.DatetimeIndex, pd.DataFrame], date_col: str = 'date', firm_col: str = 'gvkey') -> pd.DataFrame:
    # Filter a DataFrame to rows that fall within allowed dates or firm-date windows
    if date_col not in df.columns:
        return df

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors='coerce')
    work = work[work[date_col].notna()]

    if isinstance(allowed_dates, pd.DataFrame):
        if len(allowed_dates) == 0:
            return work.iloc[0:0].copy()

        if {'gvkey', 'maturity', 'start_date', 'end_date'}.issubset(allowed_dates.columns) and firm_col in work.columns:
            lhs = work.copy()
            lhs['__row_id'] = lhs.index
            lhs['__date_norm'] = lhs[date_col].dt.normalize()
            lhs['__firm_key'] = lhs[firm_col].map(_normalize_gvkey_key)
            lhs = lhs[lhs['__firm_key'] != '']

            windows = allowed_dates[['gvkey', 'maturity', 'start_date', 'end_date']].copy()
            windows['__firm_key'] = windows['gvkey'].map(_normalize_gvkey_key)
            windows['start_date'] = pd.to_datetime(windows['start_date'], errors='coerce').dt.normalize()
            windows['end_date'] = pd.to_datetime(windows['end_date'], errors='coerce').dt.normalize()
            windows['horizon_days'] = windows['maturity'].map(_maturity_to_horizon_days)
            windows = windows[
                (windows['__firm_key'] != '')
                & windows['start_date'].notna()
                & windows['end_date'].notna()
                & windows['horizon_days'].notna()
                & (windows['end_date'] >= windows['start_date'])
            ][['__firm_key', 'start_date', 'end_date', 'horizon_days']].drop_duplicates()

            merged = lhs.merge(windows, on='__firm_key', how='inner')
            merged = merged[
                (merged['__date_norm'] >= merged['start_date'])
                & (merged['__date_norm'] <= merged['end_date'])
            ]
            if merged.empty:
                return work.iloc[0:0].copy()

            per_row_horizon = merged.groupby('__row_id', as_index=True)['horizon_days'].max()
            out = lhs[lhs['__row_id'].isin(per_row_horizon.index)].copy()
            out['cds_max_horizon_days'] = out['__row_id'].map(per_row_horizon).astype(int)
            return out.drop(columns=['__row_id', '__date_norm', '__firm_key'])

        if {'gvkey', 'start_date', 'end_date'}.issubset(allowed_dates.columns) and firm_col in work.columns:
            lhs = work.copy()
            lhs['__date_norm'] = lhs[date_col].dt.normalize()
            lhs['__firm_key'] = lhs[firm_col].map(_normalize_gvkey_key)
            lhs = lhs[lhs['__firm_key'] != '']

            windows = allowed_dates[['gvkey', 'start_date', 'end_date']].copy()
            windows['__firm_key'] = windows['gvkey'].map(_normalize_gvkey_key)
            windows['start_date'] = pd.to_datetime(windows['start_date'], errors='coerce').dt.normalize()
            windows['end_date'] = pd.to_datetime(windows['end_date'], errors='coerce').dt.normalize()
            windows = windows[
                (windows['__firm_key'] != '')
                & windows['start_date'].notna()
                & windows['end_date'].notna()
                & (windows['end_date'] >= windows['start_date'])
            ][['__firm_key', 'start_date', 'end_date']].drop_duplicates()

            out = lhs.merge(windows, on='__firm_key', how='inner')
            out = out[(out['__date_norm'] >= out['start_date']) & (out['__date_norm'] <= out['end_date'])]
            return out.drop(columns=['__date_norm', '__firm_key', 'start_date', 'end_date'])

        allowed_date_col = 'Date' if 'Date' in allowed_dates.columns else ('date' if 'date' in allowed_dates.columns else None)
        if allowed_date_col is None:
            return work.iloc[0:0].copy()

        if firm_col in work.columns and 'gvkey' in allowed_dates.columns:
            lhs = work.copy()
            lhs['__date_norm'] = lhs[date_col].dt.normalize()
            lhs['__firm_key'] = lhs[firm_col].map(_normalize_gvkey_key)
            lhs = lhs[lhs['__firm_key'] != '']

            rhs = allowed_dates[[allowed_date_col, 'gvkey']].copy()
            rhs['__date_norm'] = pd.to_datetime(rhs[allowed_date_col], errors='coerce').dt.normalize()
            rhs['__firm_key'] = rhs['gvkey'].map(_normalize_gvkey_key)
            rhs = rhs[rhs['__date_norm'].notna() & (rhs['__firm_key'] != '')][['__date_norm', '__firm_key']].drop_duplicates()

            out = lhs.merge(rhs, on=['__date_norm', '__firm_key'], how='inner')
            return out.drop(columns=['__date_norm', '__firm_key'])

        # Fallback to date-only filtering if firm key is unavailable
        allowed_index = pd.DatetimeIndex(pd.to_datetime(allowed_dates[allowed_date_col], errors='coerce').dropna().dt.normalize().drop_duplicates())
        return work[work[date_col].dt.normalize().isin(allowed_index)].copy()

    if len(allowed_dates) == 0:
        return work.iloc[0:0].copy()
    return work[work[date_col].dt.normalize().isin(allowed_dates)].copy()
