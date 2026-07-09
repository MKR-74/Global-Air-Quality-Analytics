import pandas as pd
import os
from datetime import datetime

SOURCE_FILE = "waqi-covid19-airqualitydata-2023Q4-1.csv"
OUTPUT_FILE = "bronze_output.csv"
SOURCE_FILE_NAME = "waqi-covid19-airqualitydata-2023Q4-1.csv"

def load_raw_data(filepath: str) -> pd.DataFrame:
    print(f"[BRONZE] Loading raw data from: {filepath}")
    try:
        df = pd.read_csv(
            filepath,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
        print(f"[BRONZE] ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"[BRONZE] ✗ ERROR: File not found at path: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        print("[BRONZE] ✗ ERROR: The source file is empty.")
        raise
    except Exception as e:
        print(f"[BRONZE] ✗ UNEXPECTED ERROR while loading data: {e}")
        raise

def add_metadata_columns(df: pd.DataFrame, source_file_name: str) -> pd.DataFrame:
    print("[BRONZE] Adding metadata columns: ingestion_date, source_file ...")
    try:
        ingestion_timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        df = df.copy()
        df["ingestion_date"] = ingestion_timestamp
        df["source_file"] = source_file_name
        print(f"[BRONZE] ✓ ingestion_date set to: {ingestion_timestamp}")
        print(f"[BRONZE] ✓ source_file set to   : {source_file_name}")
        return df
    except Exception as e:
        print(f"[BRONZE] ✗ ERROR while adding metadata: {e}")
        raise

def save_bronze_output(df: pd.DataFrame, output_path: str) -> None:
    print(f"[BRONZE] Saving bronze output to: {output_path}")
    try:
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[BRONZE] ✓ File saved successfully.")
        print(f"[BRONZE]   → Rows    : {len(df):,}")
        print(f"[BRONZE]   → Columns : {len(df.columns)}")
        print(f"[BRONZE]   → Size    : {file_size_mb:.2f} MB")
    except PermissionError:
        print(f"[BRONZE] ✗ ERROR: No write permission at: {output_path}")
        raise
    except Exception as e:
        print(f"[BRONZE] ✗ ERROR while saving bronze output: {e}")
        raise

def print_bronze_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  BRONZE LAYER — EXECUTION SUMMARY")
    print("=" * 60)
    print(f"  Total Records   : {len(df):,}")
    print(f"  Total Columns   : {len(df.columns)}")
    print(f"  Column Names    : {df.columns.tolist()}")
    print(f"  Duplicate Rows  : {df.duplicated().sum():,}")
    empty_counts = (df == "").sum()
    total_empty = empty_counts.sum()
    print(f"  Empty Values    : {total_empty:,}")
    print(f"  Ingestion Date  : {df['ingestion_date'].iloc[0]}")
    print(f"  Source File     : {df['source_file'].iloc[0]}")
    print("=" * 60 + "\n")

def run_bronze_pipeline():
    print("\n" + "=" * 60)
    print("  BRONZE LAYER PIPELINE — START")
    print("=" * 60 + "\n")
    try:
        df_raw = load_raw_data(SOURCE_FILE)
        df_bronze = add_metadata_columns(df_raw, SOURCE_FILE_NAME)
        save_bronze_output(df_bronze, OUTPUT_FILE)
        print_bronze_summary(df_bronze)
        print("[BRONZE] ✓ Pipeline completed successfully.\n")
        return df_bronze
    except Exception as e:
        print(f"\n[BRONZE] ✗ PIPELINE FAILED: {e}")
        raise

if __name__ == "__main__":
    run_bronze_pipeline()