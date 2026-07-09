# Global Air Quality Prediction — DEPI Graduation Project

## Project Overview

This project builds a production-grade Machine Learning pipeline to **predict the Air Quality Index (AQI)** for cities around the world using meteorological and pollutant data from Q4 2023.

| Attribute         | Detail |
|-------------------|--------|
| **Problem Type**  | Regression |
| **Target**        | AQI (continuous, 0 – ~21) |
| **Best Model**    | LightGBM (Tuned) |
| **Test R²**       | **0.885** |
| **Test RMSE**     | **0.418** |
| **Data Source**   | WAQI Covid-19 Air Quality Dataset (Medallion Architecture) |
| **Records Used**  | 19,518 (Silver Layer after cleaning) |

---

## Project Structure

```
air_quality_project/
│
├── data/
│   └── silver_layer.csv          ← Cleaned input data (Medallion Silver Layer)
│
├── models/
│   ├── lgbm_model.pkl            ← Trained LightGBM model
│   ├── scaler.pkl                ← StandardScaler
│   ├── encoders.pkl              ← LabelEncoders for country/city/season
│   ├── feature_names.pkl         ← Ordered list of training features
│   └── uvi_median.pkl            ← Imputation value for UVI
│
├── outputs/
│   ├── 01_correlation_heatmap.png
│   ├── 02_model_comparison.png
│   ├── 03_pred_vs_actual.png
│   ├── 04_residuals.png
│   ├── 05_feature_importance.png
│   ├── 06_shap_summary.png
│   └── predictions.csv
│
├── model_training.py             ← Full ML pipeline (train + evaluate + save)
├── predict.py                    ← Inference script
├── requirements.txt              ← Python dependencies
└── README.md
```

---

## Quick Start (Windows + VS Code)

### Prerequisites
- Python 3.10 or higher
- VS Code with Python extension

### Step 1 — Install dependencies

Open a terminal in VS Code inside this folder and run:

```bash
pip install -r requirements.txt
```

### Step 2 — Place the data file

Ensure `silver_layer.csv` is inside the `data/` folder.

### Step 3 — Train the model

```bash
python model_training.py
```

Expected output:
- All 6 visualisation charts saved to `outputs/`
- All model artefacts saved to `models/`
- Final model metrics printed to terminal (R² ≈ 0.885)

### Step 4 — Run inference (demo)

```bash
python predict.py
```

This runs two built-in demo predictions (Berlin & Lahore) and prints results.

### Step 5 — Predict from your own CSV

Create a CSV with columns matching Silver Layer format, then:

```bash
python predict.py --input path/to/your_new_data.csv
```

Results are saved to `outputs/predictions.csv`.

---

## ML Pipeline Decisions

### Why Silver Layer?

The Gold Layer splits data into a Star Schema (fact + dimension tables), which is optimised for BI reporting. For ML, a single flat feature matrix is required, which is exactly what the Silver Layer provides after all cleaning and feature engineering.

### Why Regression?

AQI is a continuous numeric variable (ranging from ~0.002 to ~20.9 in this dataset). Predicting its exact value gives far more actionable information than a 3-class classification, and allows downstream re-labelling at any threshold.

### Why LightGBM?

| Model             | R²    | Notes |
|-------------------|-------|-------|
| Linear Regression | 0.170 | Too simple; AQI relationship is non-linear |
| Decision Tree     | 0.728 | Overfits without pruning |
| Random Forest     | 0.868 | Strong but slower and larger memory footprint |
| XGBoost           | 0.860 | Close competitor |
| **LightGBM**      | **0.885** | Best after tuning; fastest training; handles categoricals natively |
| CatBoost          | 0.858 | Good but slower to tune |

### Feature Engineering

From the raw `date` column:
- `day_of_year`, `week_of_year`, `month`, `day_of_week`

These capture seasonal and cyclical pollution patterns without leaking future information.

---

## Evaluation Metrics (Final Tuned Model)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MAE    | 0.233 | Average prediction error is 0.23 AQI units |
| RMSE   | 0.418 | Penalises large errors more; still low |
| R²     | 0.885 | Model explains 88.5% of AQI variance |

---

## Visualisations

| File | Description |
|------|-------------|
| `01_correlation_heatmap.png` | Pearson correlations between all features |
| `02_model_comparison.png`    | R² scores for all 8 candidate models |
| `03_pred_vs_actual.png`      | Scatter: true vs predicted AQI |
| `04_residuals.png`           | Residual plot + error distribution |
| `05_feature_importance.png`  | Top-20 most important features (LightGBM gain) |
| `06_shap_summary.png`        | SHAP beeswarm — individual feature impact |

---

## Retraining the Model

To retrain from scratch (e.g., with new data):

1. Replace `data/silver_layer.csv` with the new Silver Layer output.
2. Run `python model_training.py`.
3. All model files and charts are regenerated automatically.
