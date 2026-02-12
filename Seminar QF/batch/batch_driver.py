import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from google.cloud import storage

# Add project root to path
# Assuming this script runs from /app in the container
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.analysis.monte_carlo_garch import monte_carlo_garch_1year_parallel
except ImportError:
    # Try adding one level up if run from a subdirectory
    sys.path.append(str(project_root.parent))
    from src.analysis.monte_carlo_garch import monte_carlo_garch_1year_parallel

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Monte Carlo GARCH Simulation")
    parser.add_argument('--job-index', type=int, default=os.environ.get('BATCH_TASK_INDEX', 0), help='Index of current batch task')
    parser.add_argument('--task-count', type=int, default=os.environ.get('BATCH_TASK_COUNT', 1), help='Total number of batch tasks')
    parser.add_argument('--input-bucket', required=True, help='GCS bucket containing input data')
    parser.add_argument('--input-file', default='data/output/daily_asset_returns_with_garch.csv', help='Path to input CSV in bucket')
    parser.add_argument('--merton-file', default='data/output/merged_data_with_merton.csv', help='Path to Merton CSV in bucket')
    parser.add_argument('--output-bucket', required=True, help='GCS bucket for output results')
    parser.add_argument('--output-prefix', default='output/results', help='Prefix for output files')
    parser.add_argument('--num-simulations', type=int, default=1000, help='Number of simulations per firm')
    return parser.parse_args()

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded gs://{bucket_name}/{source_blob_name} to {destination_file_name}")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}")

def main():
    args = parse_args()
    
    # Define local paths
    local_input = 'input_data.csv'
    local_merton = 'merton_data.csv'
    
    # 1. Download Input Data
    # For large scale, downloading repeatedly is okay if data is small (<100MB).
    # If data is huge, consider baking into image or using a persistent volume.
    print(f"Task {args.job_index}/{args.task_count}: Downloading inputs...")
    try:
        download_blob(args.input_bucket, args.input_file, local_input)
        # Try to download merton file, but it's optional in the logic (though recommended)
        try:
            download_blob(args.input_bucket, args.merton_file, local_merton)
        except Exception as e:
            print(f"Warning: Could not download Merton file: {e}")
            local_merton = None
            
    except Exception as e:
        print(f"Critical Error downloading input: {e}")
        sys.exit(1)

    # 2. Load and Partition Data by DATE
    df = pd.read_csv(local_input)
    if 'date' not in df.columns:
        # If no date column, we can't split by date. Fail or run all.
        print("Error: Input data has no 'date' column.")
        sys.exit(1)
        
    unique_dates = sorted(df['date'].unique())
    total_dates = len(unique_dates)
    print(f"Total unique dates in dataset: {total_dates}")
    
    # Calculate shard
    dates_per_task = int(np.ceil(total_dates / args.task_count))
    start_idx = args.job_index * dates_per_task
    end_idx = min(start_idx + dates_per_task, total_dates)
    
    if start_idx >= total_dates:
        print(f"Task index {args.job_index} is out of range for {total_dates} dates. Exiting.")
        sys.exit(0)
        
    task_dates = unique_dates[start_idx:end_idx]
    print(f"Processing {len(task_dates)} dates: from {task_dates[0]} to {task_dates[-1]}")
    
    # Filter DataFrame
    df_subset = df[df['date'].isin(task_dates)].copy()
    
    # Load Merton DF if available (needed for efficient PD calc)
    merton_df_arg = None
    if local_merton and os.path.exists(local_merton):
        merton_full = pd.read_csv(local_merton)
        # Filter Merton data to relevant dates/firms to save memory
        merton_df_arg = merton_full[merton_full['date'].isin(task_dates)].copy()
    
    # 3. Run Simulation
    # n_jobs=1 because we rely on Batch parallelism. 
    # If the VM has multiple cores, we could increase this, but for 10000 splits, 1 core is best.
    results = monte_carlo_garch_1year_parallel(
        garch_file=df_subset,
        num_simulations=args.num_simulations,
        n_jobs=1,
        merton_df=merton_df_arg
    )
    
    # 4. Upload Results
    local_output = f"results_{args.job_index}.csv"
    results.to_csv(local_output, index=False)
    
    target_blob = f"{args.output_prefix}/batch_results_{args.job_index}.csv"
    upload_blob(args.output_bucket, local_output, target_blob)
    
    print("Batch Task Complete.")

if __name__ == "__main__":
    main()
