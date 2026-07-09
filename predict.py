import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from dashboard import build_dashboard

MODELS_DIR = "models"


# ============================================================================
# LOAD ARTEFACTS
# ============================================================================

def load_artefacts():
    """Load all saved model objects. Raises clear errors if any are missing."""
    required = ["lgbm_model.pkl", "scaler.pkl", "encoders.pkl",
                "feature_names.pkl", "uvi_median.pkl"]
    for fname in required:
        path = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artefact: {path}\n"
                "Please run model_training.py first to generate all model files."
            )

    model         = joblib.load(os.path.join(MODELS_DIR, "lgbm_model.pkl"))
    scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    encoders      = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    uvi_median    = joblib.load(os.path.join(MODELS_DIR, "uvi_median.pkl"))

    print("[PREDICT] ✓ All model artefacts loaded successfully.")
    return model, scaler, encoders, feature_names, uvi_median


# ============================================================================
# PREPROCESSING (mirrors model_training.py logic)
# ============================================================================

def preprocess_input(df: pd.DataFrame, encoders: dict,
                     feature_names: list, uvi_median: float) -> np.ndarray:
    """
    Apply the exact same transformations used during training.
    Accepts a DataFrame with Silver-layer columns (without AQI).
    """
    df = df.copy()

    # Parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"]        = df["date"].dt.month
        df["day_of_week"]  = df["date"].dt.dayofweek
        df = df.drop(columns=["date"])

    # Drop columns not used in training
    for col in ["AQI", "AQI_label", "AQI_sort_order",
                "silver_processed_date", "data_quality_flag", "month_name"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # UVI imputation
    if "uvi" in df.columns:
        df["uvi"] = df["uvi"].fillna(uvi_median)

    # Encode categoricals
    for col, le in encoders.items():
        if col in df.columns:
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_
                else -1   # -1 for completely unseen categories
            )

    # Reorder columns to match training feature order
    missing_cols = set(feature_names) - set(df.columns)
    for mc in missing_cols:
        df[mc] = 0   # Fill with zero if column absent
    df = df[feature_names]

    return df.values


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_aqi(input_data, model, scaler, encoders,
                feature_names, uvi_median) -> pd.DataFrame:
    """
    Accepts a dict (single record) or a DataFrame (batch).
    Returns a DataFrame with original data + predicted AQI + AQI label.
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise TypeError("input_data must be a dict or a pandas DataFrame.")

    X = preprocess_input(df_input, encoders, feature_names, uvi_median)
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Attach predictions
    df_result = df_input.copy()
    df_result["Predicted_AQI"] = predictions

    # Re-apply AQI label logic (same thresholds as Silver layer)
    df_result["Predicted_AQI_Label"] = pd.cut(
        predictions,
        bins=[-np.inf, 1.0, 2.0, np.inf],
        labels=["Safe / Healthy", "Polluted / Unhealthy", "Hazardous"]
    )

    return df_result


# ============================================================================
# DISPLAY RESULTS
# ============================================================================

def print_results(df_result: pd.DataFrame):
    """Pretty-print prediction results to the terminal."""
    print("\n" + "=" * 70)
    print("  AIR QUALITY PREDICTION RESULTS")
    print("=" * 70)
    for i, row in df_result.iterrows():
        print(f"\n  Record #{i + 1}")
        print(f"  {'Country':<22}: {row.get('country', 'N/A')}")
        print(f"  {'City':<22}: {row.get('city', 'N/A')}")
        print(f"  {'Date':<22}: {row.get('date', 'N/A')}")
        print(f"  {'Temperature (°C)':<22}: {row.get('temperature', 'N/A')}")
        print(f"  {'Humidity (%)':<22}: {row.get('humidity', 'N/A')}")
        print(f"  {'Pressure (hPa)':<22}: {row.get('pressure', 'N/A')}")
        print(f"  {'→ Predicted AQI':<22}: {row['Predicted_AQI']:.4f}")
        print(f"  {'→ AQI Category':<22}: {row['Predicted_AQI_Label']}")
    print("\n" + "=" * 70)


# ============================================================================
# DEMO DATA
# ============================================================================

DEMO_RECORDS = [
    {   # Example 1 — Clean European city in autumn
        "date": "2023-11-15",
        "country": "MX",
        "city": "Aguascalie",
        "temperature": 10.0,
        "pressure": 500.0,
        "humidity": 2.0,
        "dew": 3.5,
        "wind gust": 5.2,
        "uvi": 1.0,
        "precipitation": 0.0,
        "is_weekday": 1,
        "season": "Autumn",
    },
    {   # Example 2 — Potentially polluted winter city
        "date": "2023-12-20",
        "country": "PK",
        "city": "Lahore",
        "temperature": 14.0,
        "pressure": 1000.0,
        "humidity": 2.0,
        "dew": 3.5,
        "wind gust": 5.2,
        "uvi": 1.0,
        "precipitation": 0.0,
        "is_weekday": 0,
        "season": "Winter",
    },
    
     
]


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Air Quality AQI Predictor")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a CSV file with new records to predict.")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Skip generating/opening the HTML dashboard.")
    args = parser.parse_args()

    # Load model
    model, scaler, encoders, feature_names, uvi_median = load_artefacts()

    if args.input:
        print(f"[PREDICT] Reading input file: {args.input}")
        df_new = pd.read_csv(args.input)
    else:
        print("[PREDICT] No --input file provided. Running built-in demo records ...")
        df_new = pd.DataFrame(DEMO_RECORDS)

    # Predict
    df_result = predict_aqi(df_new, model, scaler, encoders, feature_names, uvi_median)

    # Display
    print_results(df_result)

    # Save results
    out_path = "outputs/predictions.csv"
    df_result.to_csv(out_path, index=False)
    print(f"\n[PREDICT] ✓ Predictions saved to: {out_path}")

    # Build the HTML dashboard and open it in the default browser
    if not args.no_dashboard:
        dashboard_path = build_dashboard(df_result, out_dir="outputs")
        print(f"[PREDICT] ✓ Dashboard ready — opening in your browser:")
        print(f"           file://{dashboard_path}\n")


if __name__ == "__main__":
    main()
