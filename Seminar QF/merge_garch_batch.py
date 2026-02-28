#!/usr/bin/env python3
"""
Fast merge of GARCH batch results with progress tracking and time estimates.
Optimized for quick concatenation of batch CSV files.
"""
import pandas as pd
import glob
import os
import time
from pathlib import Path

def merge_garch_batch_results():
    """
    Merge all GARCH batch result CSV files into one consolidated file.
    Optimized for speed with progress tracking.
    """
    print("\n" + "="*70)
    print("GARCH BATCH RESULTS MERGER")
    print("="*70)
    
    start_time = time.time()
    
    # Set up paths
    base_dir = Path(__file__).parent
    batch_dir = base_dir / "data" / "output" / "garch_batch"
    output_file = base_dir / "data" / "output" / "daily_asset_returns_with_garch_batch.csv"
    
    # Get all batch result files
    pattern = str(batch_dir / "batch_results_*.csv")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].replace('.csv', '')))
    
    if not files:
        print(f"⚠ ERROR: No batch result files found in {batch_dir}")
        return
    
    print(f"\n📁 Found {len(files)} batch result files")
    print(f"📂 Reading from: {batch_dir}")
    print(f"💾 Writing to: {output_file}")
    print(f"\n⏳ Starting merge process...")
    
    # Read all files with progress
    dfs = []
    skipped = []
    
    for i, file in enumerate(files, 1):
        try:
            df = pd.read_csv(file)
            
            if df.empty:
                print(f"  ⚠ Skipping empty file: {os.path.basename(file)}")
                skipped.append(file)
                continue
                
            dfs.append(df)
            
            # Progress indicator every 10 files
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining_files = len(files) - i
                eta = remaining_files / rate if rate > 0 else 0
                print(f"  ✓ Loaded {i}/{len(files)} files ({i/len(files)*100:.1f}%) | "
                      f"Rate: {rate:.1f} files/sec | ETA: {eta:.1f}s")
                
        except Exception as e:
            print(f"  ⚠ Error reading {os.path.basename(file)}: {e}")
            skipped.append(file)
            continue
    
    load_time = time.time() - start_time
    print(f"\n✓ Loaded {len(dfs)}/{len(files)} files in {load_time:.2f} seconds")
    
    if skipped:
        print(f"  ⚠ Skipped {len(skipped)} files due to errors")
    
    if not dfs:
        print(f"⚠ ERROR: No valid dataframes to merge")
        return
    
    # Concatenate all dataframes
    print(f"\n⏳ Concatenating {len(dfs)} dataframes...")
    concat_start = time.time()
    merged_df = pd.concat(dfs, ignore_index=True)
    concat_time = time.time() - concat_start
    print(f"✓ Concatenated in {concat_time:.2f} seconds")
    
    # Sort by gvkey and date for consistency
    print(f"\n⏳ Sorting by gvkey and date...")
    sort_start = time.time()
    merged_df = merged_df.sort_values(['gvkey', 'date']).reset_index(drop=True)
    sort_time = time.time() - sort_start
    print(f"✓ Sorted in {sort_time:.2f} seconds")
    
    # Save to output file
    print(f"\n⏳ Saving to CSV...")
    save_start = time.time()
    merged_df.to_csv(output_file, index=False)
    save_time = time.time() - save_start
    print(f"✓ Saved in {save_time:.2f} seconds")
    
    # Print summary statistics
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✅ MERGE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"📊 Total rows: {len(merged_df):,}")
    print(f"🏢 Unique companies (gvkeys): {merged_df['gvkey'].nunique():,}")
    print(f"📅 Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"📋 Columns: {len(merged_df.columns)}")
    print(f"💾 Output file size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"\n📁 Output file: {output_file}")
    print("="*70)
    
    # Show column names
    print(f"\n📋 Available columns:")
    for i, col in enumerate(merged_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # Show sample of non-null data
    print(f"\n📈 Sample of results (first non-null row):")
    non_null_mask = merged_df['mc_garch_pd_1y'].notna()
    if non_null_mask.any():
        sample_row = merged_df[non_null_mask].iloc[0]
        print(f"  gvkey: {sample_row['gvkey']}")
        print(f"  date: {sample_row['date']}")
        print(f"  PD 1Y: {sample_row['mc_garch_pd_1y']:.4%}")
        print(f"  PD 3Y: {sample_row['mc_garch_pd_3y']:.4%}")
        print(f"  PD 5Y: {sample_row['mc_garch_pd_5y']:.4%}")
    else:
        print("  ⚠ No non-null PD values found in results")
    
    print("\n✅ Ready for use in notebooks like visualize_results.ipynb")
    print("="*70 + "\n")

if __name__ == "__main__":
    merge_garch_batch_results()
