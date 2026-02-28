#!/usr/bin/env python3
"""
Merge batch result files from GCS downloads into consolidated CSV files.
Combines all batch_results_*.csv files for each model into single output files.
"""
import pandas as pd
import glob
import os
from pathlib import Path

def merge_batch_results(batch_dir, output_file, model_name):
    """
    Merge all batch result CSV files in a directory into one consolidated file.
    
    Args:
        batch_dir: Directory containing batch_results_*.csv files
        output_file: Path to output consolidated CSV file
        model_name: Name of the model for logging
    """
    print(f"\n{'='*60}")
    print(f"Merging {model_name} batch results...")
    print(f"{'='*60}")
    
    # Get all batch result files
    pattern = os.path.join(batch_dir, "batch_results_*.csv")
    files = sorted(glob.glob(pattern), key=lambda x: int(x.split('_')[-1].replace('.csv', '')))
    
    if not files:
        print(f"⚠ WARNING: No batch result files found in {batch_dir}")
        return
    
    print(f"Found {len(files)} batch result files")
    
    # Read and concatenate all files
    dfs = []
    skipped = []
    for i, file in enumerate(files):
        try:
            df = pd.read_csv(file)
            if df.empty:
                print(f"  ⚠ Skipping empty file: {os.path.basename(file)}")
                skipped.append(file)
                continue
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Error reading {os.path.basename(file)}: {e}")
            skipped.append(file)
            continue
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i + 1}/{len(files)} files...")
    
    print(f"  Loaded {len(dfs)}/{len(files)} files")
    if skipped:
        print(f"  ⚠ Skipped {len(skipped)} files due to errors")
    
    if not dfs:
        print(f"⚠ ERROR: No valid dataframes to merge for {model_name}")
        return
    
    # Concatenate all dataframes
    print("  Concatenating dataframes...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by gvkey and date for consistency
    print("  Sorting by gvkey and date...")
    merged_df = merged_df.sort_values(['gvkey', 'date']).reset_index(drop=True)
    
    # Save to output file
    print(f"  Saving to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print(f"\n✓ {model_name} Results Summary:")
    print(f"  - Total rows: {len(merged_df):,}")
    print(f"  - Unique gvkeys: {merged_df['gvkey'].nunique():,}")
    print(f"  - Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"  - Columns: {len(merged_df.columns)}")
    print(f"  - Output file size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    print(f"  - Saved to: {output_file}")

def main():
    # Set up paths
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data" / "output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("BATCH RESULTS CONSOLIDATION")
    print("="*60)
    
    # Merge GARCH results from the downloaded location
    garch_batch_dir = output_dir / "garch_batch"
    garch_output = output_dir / "batch_garch_results.csv"
    if garch_batch_dir.exists():
        merge_batch_results(garch_batch_dir, garch_output, "GARCH")
    else:
        print(f"⚠ WARNING: GARCH batch directory not found: {garch_batch_dir}")
    
    # Merge Regime Switching results (when available)
    rs_batch_dir = output_dir / "regime_switching_batch"
    rs_output = output_dir / "batch_regime_switching_results.csv"
    if rs_batch_dir.exists():
        merge_batch_results(rs_batch_dir, rs_output, "Regime Switching")
    else:
        print(f"⚠ INFO: Regime Switching batch directory not found: {rs_batch_dir}")
    
    print("\n" + "="*60)
    print("✓ CONSOLIDATION COMPLETE")
    print("="*60)
    print(f"\nOutput files created in: {output_dir}")
    if garch_batch_dir.exists():
        print(f"  - {garch_output.name}")
    if rs_batch_dir.exists():
        print(f"  - {rs_output.name}")
    print("\n")

if __name__ == "__main__":
    main()
