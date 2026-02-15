import os
import argparse
import glob
from pathlib import Path

import pandas as pd
from google.cloud import storage


DEFAULT_BUCKET = "seminar-qf-batch-data-001"
DEFAULT_PREFIX = "output/results"

MODEL_TO_OUTPUT_FILE = {
    "garch": "daily_monte_carlo_garch_results.csv",
    "regime-switching": "daily_monte_carlo_regime_switching_results.csv",
    "ms-garch": "daily_monte_carlo_ms_garch_results.csv",
}


def normalize_model(model_raw: str) -> str:
    model = model_raw.strip().lower()
    aliases = {
        "msgarch": "ms-garch",
        "msgrach": "ms-garch",
    }
    model = aliases.get(model, model)

    if model not in MODEL_TO_OUTPUT_FILE:
        valid = ", ".join(sorted(list(MODEL_TO_OUTPUT_FILE.keys()) + list(aliases.keys())))
        raise ValueError(f"Unsupported model '{model_raw}'. Use one of: {valid}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and merge Google Batch Monte Carlo shards by model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to download: garch, regime-switching, ms-garch (aliases: msgarch, msgrach)",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="GCS bucket name")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="Base GCS prefix for batch outputs")
    parser.add_argument(
        "--local-dir",
        default="./batch_results",
        help="Local temp root where shard CSVs are downloaded",
    )
    return parser.parse_args()


def download_results(bucket_name: str, model: str, model_prefix: str, model_local_dir: Path) -> int:
    """Downloads model-specific CSV shards from GCS to local temp folder."""
    print(f"Connecting to bucket: {bucket_name}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    model_local_dir.mkdir(parents=True, exist_ok=True)

    blobs = bucket.list_blobs(prefix=model_prefix)
    count = 0

    print(f"Downloading model='{model}' files from gs://{bucket_name}/{model_prefix} ...")
    for blob in blobs:
        if blob.name.endswith(".csv"):
            filename = os.path.basename(blob.name)
            local_path = model_local_dir / filename
            blob.download_to_filename(str(local_path))
            count += 1
            if count % 100 == 0:
                print(f"Downloaded {count} files...")

    print(f"Download complete. Total files: {count}")
    return count


def merge_results(model_local_dir: Path, final_output_file: Path):
    """Merges downloaded shard CSVs and writes the final model-specific output file."""
    csv_files = glob.glob(str(model_local_dir / "*.csv"))

    if not csv_files:
        print("No CSV files found to merge.")
        return False

    print(f"Merging {len(csv_files)} files...")

    df_list = []
    for file_path in csv_files:
        try:
            df_list.append(pd.read_csv(file_path))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not df_list:
        print("Nothing to merge.")
        return False

    final_df = pd.concat(df_list, ignore_index=True)
    final_output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(final_output_file, index=False)

    print(f"Successfully merged results into: {final_output_file}")
    print(f"Total rows: {len(final_df)}")
    return True


def cleanup_source_shards(model_local_dir: Path):
    """Deletes downloaded shard CSVs after merge, keeping only merged result."""
    csv_files = glob.glob(str(model_local_dir / "*.csv"))
    deleted = 0
    for file_path in csv_files:
        try:
            os.remove(file_path)
            deleted += 1
        except Exception as e:
            print(f"Could not delete {file_path}: {e}")

    try:
        model_local_dir.rmdir()
    except OSError:
        pass

    print(f"Cleaned up {deleted} source shard files from {model_local_dir}")


if __name__ == "__main__":
    args = parse_args()
    model = normalize_model(args.model)

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "data" / "output"
    final_filename = MODEL_TO_OUTPUT_FILE[model]
    final_output_file = output_dir / final_filename

    local_root = Path(args.local_dir)
    model_local_dir = local_root / model
    model_prefix = f"{args.prefix.rstrip('/')}/{model}/"

    downloaded = download_results(args.bucket, model, model_prefix, model_local_dir)
    if downloaded > 0:
        merged_ok = merge_results(model_local_dir, final_output_file)
        if merged_ok:
            cleanup_source_shards(model_local_dir)
