import os
import argparse
import pandas as pd
from google.cloud import storage
import glob
from pathlib import Path

# Configuration
BUCKET_NAME = "seminar-qf-batch-data-001"
PREFIX = "output/results/"
LOCAL_DIR = "./batch_results"
FINAL_FILE = "final_monte_carlo_results.csv"

def download_results():
    """Downloads all CSV files from the GCS bucket prefix."""
    print(f"Connecting to bucket: {BUCKET_NAME}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Create local directory
    Path(LOCAL_DIR).mkdir(parents=True, exist_ok=True)
    
    blobs = bucket.list_blobs(prefix=PREFIX)
    count = 0
    
    print(f"Downloading files to {LOCAL_DIR}...")
    for blob in blobs:
        if blob.name.endswith(".csv"):
            filename = os.path.basename(blob.name)
            local_path = os.path.join(LOCAL_DIR, filename)
            blob.download_to_filename(local_path)
            count += 1
            if count % 100 == 0:
                print(f"Downloaded {count} files...")
    
    print(f"Download complete. Total files: {count}")
    return count

def merge_results():
    """Merges all downloaded CSV files into one."""
    csv_files = glob.glob(os.path.join(LOCAL_DIR, "*.csv"))
    
    if not csv_files:
        print("No CSV files found to merge.")
        return

    print(f"Merging {len(csv_files)} files...")
    
    # Use a generator to read files to save memory if needed, 
    # but for result summaries pandas concat is usually fine.
    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(FINAL_FILE, index=False)
        print(f"Successfully merged results into: {FINAL_FILE}")
        print(f"Total rows: {len(final_df)}")
    else:
        print("Nothing to merge.")

if __name__ == "__main__":
    if download_results() > 0:
        merge_results()
