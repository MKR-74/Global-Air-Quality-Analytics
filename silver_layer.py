import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone

BRONZE_FILE = "bronze_output.csv"
OUTPUT_FILE = "silver_layer.csv"

NON_NEGATIVE_SPECIES = {
    "pm25", "pm10", "pm1",
    "no2", "o3", "so2", "co",
    "humidity", "pressure",
    "wind-speed", "wind-gust",
    "precipitation", "uvi",
    "aqi", "mepaqi",
}

def load_bronze_data(filepath: str) -> pd.DataFrame:
    print(f"[SILVER] Loading bronze data from: {filepath}")
    try:
        df = pd.read_csv(filepath, dtype=str, keep_default_na=False, encoding="utf-8")
        print(f"[SILVER] ✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"[SILVER] ✗ ERROR: Bronze file not found: {filepath}")
        raise
    except Exception as e:
        print(f"[SILVER] ✗ ERROR loading bronze data: {e}")
        raise

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    print("[SILVER] Standardizing column names to snake_case ...")
    try:
        df = df.copy()
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[\s\-]+", "_", regex=True)
            .str.replace(r"[^\w]", "", regex=True)
        )
        return df
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in standardize_column_names: {e}")
        raise

def strip_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    print("[SILVER] Stripping leading/trailing whitespace ...")
    try:
        df = df.copy()
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in strip_string_whitespace: {e}")
        raise

def enforce_data_types(df: pd.DataFrame) -> pd.DataFrame:
    print("[SILVER] Enforcing data types ...")
    try:
        df = df.copy()
        df["count"] = pd.to_numeric(df["count"], errors="coerce").astype("Int64")
        for col in ["min", "max", "median", "variance"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in enforce_data_types: {e}")
        raise

def standardize_date_format(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    print(f"[SILVER] Standardizing date column '{date_col}' to ISO 8601 ...")
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y", errors="coerce")
        return df
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in standardize_date_format: {e}")
        raise

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    print("[SILVER] Removing duplicate rows ...")
    try:
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in remove_duplicates: {e}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    print("[SILVER] Handling missing values ...")
    try:
        df = df.copy()
        key_cols = ["date", "country", "city", "specie"]
        for col in key_cols:
            df = df.dropna(subset=[col])

        count_nulls = df["count"].isna().sum()
        if count_nulls > 0:
            df["count"] = df["count"].fillna(pd.array([1], dtype="Int64")[0])

        for col in ["min", "max", "median"]:
            group_median = df.groupby(["city", "specie"])[col].transform("median")
            global_median = df[col].median()
            df[col] = df[col].fillna(group_median).fillna(global_median)

        df["variance"] = df["variance"].fillna(0.0)
        return df
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in handle_missing_values: {e}")
        raise

def validate_and_remove_invalid_records(df: pd.DataFrame) -> pd.DataFrame:
    print("[SILVER] Validating records and removing invalid rows ...")
    try:
        df = df.copy()
        df = df[~(df["min"] > df["max"])].reset_index(drop=True)

        mask_r2 = df["specie"].str.lower().isin(NON_NEGATIVE_SPECIES) & (df["min"] < 0)
        df = df[~mask_r2].reset_index(drop=True)
        df = df[~(df["count"].fillna(0) < 1)].reset_index(drop=True)
        df = df[~(df["variance"] < 0)].reset_index(drop=True)

        q4_start, q4_end = pd.Timestamp("2023-10-01"), pd.Timestamp("2023-12-31")
        df = df[df["date"].between(q4_start, q4_end)].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in validate_and_remove_invalid_records: {e}")
        raise

def standardize_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    print("[SILVER] Standardizing categorical string values ...")
    try:
        df = df.copy()
        df["country"] = df["country"].str.upper()
        df["city"]    = df["city"].str.title()
        df["specie"]  = df["specie"].str.lower().str.strip()

        df["specie"] = df["specie"].replace({
            'wind-gust': 'wind gust',
            'wind_gust': 'wind gust',
            'windgust': 'wind gust'
        })
        return df
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in standardize_categorical_values: {e}")
        raise

def transform_and_calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("[SILVER/TRANSFORM] Reshaping data and calculating AQI & Time features ...")
    try:
        df = df.copy()
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'o3', 'o3_norm']
        weather_species = ['temperature', 'pressure', 'humidity', 'dew', 'precipitation', 'wind gust', 'uvi']

        df_filtered = df[df['specie'].isin(pollutants + weather_species)].copy()

        df_wide = df_filtered.pivot_table(
            index=['date', 'country', 'city'],
            columns='specie',
            values='median',
            aggfunc='mean'
        ).reset_index()
        df_wide.columns.name = None

        if 'precipitation' in df_wide.columns:
            df_wide['precipitation'] = df_wide['precipitation'].fillna(0)

        weather_cols = ['temperature', 'pressure', 'humidity', 'dew', 'wind gust', 'uvi']
        existing_weather = [col for col in weather_cols if col in df_wide.columns]

        df_wide = df_wide.sort_values(by=['country', 'city', 'date']).reset_index(drop=True)
        
        if existing_weather:
            df_wide[existing_weather] = df_wide.groupby(['country', 'city'])[existing_weather].ffill().bfill()

        if 'pm25' in df_wide.columns: df_wide['pm25_norm'] = df_wide['pm25'] / 15
        if 'pm10' in df_wide.columns: df_wide['pm10_norm'] = df_wide['pm10'] / 45
        if 'no2' in df_wide.columns: df_wide['no2_norm'] = df_wide['no2'] / 25
        if 'so2' in df_wide.columns: df_wide['so2_norm'] = df_wide['so2'] / 40

        o3_source = 'o3' if 'o3' in df_wide.columns else ('o3_norm' if 'o3_norm' in df_wide.columns else None)
        if o3_source: df_wide['o3_norm'] = df_wide[o3_source] / 100

        norm_columns = ['pm25_norm', 'pm10_norm', 'no2_norm', 'so2_norm', 'o3_norm']
        existing_norm_cols = [col for col in norm_columns if col in df_wide.columns]

        if existing_norm_cols:
            df_wide['AQI'] = df_wide[existing_norm_cols].mean(axis=1)
        else:
            df_wide['AQI'] = np.nan

        conditions = [df_wide['AQI'] <= 1.0, (df_wide['AQI'] > 1.0) & (df_wide['AQI'] < 2.0), df_wide['AQI'] >= 2.0]
        choices = ['Safe / Healthy', 'Polluted / Unhealthy', 'Hazardous']
        df_wide['AQI_label'] = np.select(conditions, choices, default='Unknown')
        df_wide['AQI_sort_order'] = np.select(conditions, [1, 2, 3], default=0)

        df_wide = df_wide.drop(columns=pollutants + existing_norm_cols, errors='ignore')

        df_wide['date'] = pd.to_datetime(df_wide['date'])
        df_wide['is_weekday'] = np.where(df_wide['date'].dt.dayofweek.isin([5, 6]), 0, 1)
        df_wide['month_name'] = df_wide['date'].dt.month_name()

        month_numbers = df_wide['date'].dt.month
        conditions_season = [month_numbers.isin([12, 1, 2]), month_numbers.isin([3, 4, 5]), month_numbers.isin([6, 7, 8]), month_numbers.isin([9, 10, 11])]
        df_wide['season'] = np.select(conditions_season, ['Winter', 'Spring', 'Summer', 'Autumn'], default='Unknown')

        actual_weather_cols = [c for c in weather_cols + ['precipitation'] if c in df_wide.columns]
        calculated_features = ['AQI', 'AQI_label', 'AQI_sort_order', 'is_weekday', 'month_name', 'season']
        final_column_order = ['date', 'country', 'city'] + actual_weather_cols + calculated_features

        df_wide = df_wide[final_column_order]
        return df_wide
    except Exception as e:
        print(f"[SILVER] ✗ ERROR in transform_and_calculate_metrics: {e}")
        raise

def add_silver_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    df["silver_processed_date"] = ts
    df["data_quality_flag"] = "VALID"
    return df

def run_silver_pipeline():
    print("\n============================================================")
    print("  SILVER LAYER PIPELINE — START")
    print("============================================================\n")
    try:
        df = load_bronze_data(BRONZE_FILE)
        df = standardize_column_names(df)
        df = strip_string_whitespace(df)
        df = enforce_data_types(df)
        df = standardize_date_format(df)
        df = remove_duplicates(df)
        df = handle_missing_values(df)
        df = validate_and_remove_invalid_records(df)
        df = standardize_categorical_values(df)

        df_transformed = transform_and_calculate_metrics(df)
        df_transformed = add_silver_metadata(df_transformed)

        df_transformed.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"[SILVER] ✓ Success! Saved single file: {OUTPUT_FILE}")

        return df_transformed
    except Exception as e:
        print(f"\n[SILVER] ✗ PIPELINE FAILED: {e}")
        raise

if __name__ == "__main__":
    run_silver_pipeline()