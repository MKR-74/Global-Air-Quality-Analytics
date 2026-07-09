import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

DATA_DIR   = "data"
MODELS_DIR = "models"
OUT_DIR    = "outputs"
for d in [DATA_DIR, MODELS_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

def load_silver_data(path: str) -> pd.DataFrame:
    print(f"\n[LOAD] Reading data from: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"[LOAD] ✓  {len(df):,} rows × {df.shape[1]} columns loaded.")
    return df

def prepare_features(df: pd.DataFrame):
    print("\n[PREP] Starting feature engineering ...")

    df = df.copy()  #df=data frame copy

    before = len(df)
    df = df.dropna(subset=["AQI"]).reset_index(drop=True)
    print(f"[PREP] Dropped {before - len(df):,} rows with missing AQI.")

    drop_cols = ["AQI_label", "AQI_sort_order", "silver_processed_date", "data_quality_flag"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["day_of_week"]  = df["date"].dt.dayofweek

    uvi_median = df["uvi"].median()
    df["uvi"] = df["uvi"].fillna(uvi_median)
    print(f"[PREP] UVI NaN filled with median = {uvi_median:.2f}")

    df = df.drop(columns=["date", "month_name"])

    cat_cols = ["country", "city", "season"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        print(f"[PREP] Encoded '{col}' → {le.classes_.shape[0]} unique values.")

    X = df.drop(columns=["AQI"])
    y = df["AQI"]

    print(f"[PREP] ✓ Final shape: X={X.shape}, y={y.shape}")
    print(f"[PREP] Features: {X.columns.tolist()}")
    return X, y, encoders, uvi_median

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler

def evaluate_model(model, X_tr, X_te, y_tr, y_te, name: str) -> dict:
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)
    cv   = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2").mean()
    print(f"  {name:<22} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  CV-R²={cv:.4f}")
    return {"model": model, "name": name, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv}

def compare_models(X_train, X_test, y_train, y_test) -> list:
    print("\n[MODEL SELECTION] Comparing algorithms ...")
    print(f"  {'Algorithm':<22} {'MAE':>8}  {'RMSE':>8}  {'R²':>7}  {'CV-R²':>7}")
    print("  " + "-" * 60)

    candidates = [
        ("Linear Regression",  LinearRegression()),
        ("Ridge",              Ridge(alpha=1.0)),
        ("Lasso",              Lasso(alpha=0.01, max_iter=5000)),
        ("Decision Tree",      DecisionTreeRegressor(random_state=42)),
        ("Random Forest",      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ("XGBoost",            XGBRegressor(n_estimators=200, random_state=42, verbosity=0)),
        ("LightGBM",           LGBMRegressor(n_estimators=200, random_state=42, verbosity=-1)),
        ("CatBoost",           CatBoostRegressor(iterations=200, random_seed=42, verbose=0)),
    ]

    results = []
    for name, model in candidates:
        result = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(result)

    return results

def tune_best_model(X_train, y_train) -> LGBMRegressor:
    print("\n[TUNING] Running RandomizedSearchCV on LightGBM ...")

    param_grid = {
        "n_estimators":      [200, 400, 600],
        "max_depth":         [4, 6, 8, -1],
        "learning_rate":     [0.01, 0.05, 0.1],
        "num_leaves":        [31, 63, 127],
        "subsample":         [0.7, 0.8, 1.0],
        "colsample_bytree":  [0.7, 0.8, 1.0],
        "min_child_samples": [10, 20, 50],
    }

    lgbm = LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1)
    search = RandomizedSearchCV(
        lgbm, param_distributions=param_grid,
        n_iter=30, cv=5, scoring="r2",
        random_state=42, n_jobs=-1, verbose=1
    )
    search.fit(X_train, y_train)

    print(f"[TUNING] Best CV R² = {search.best_score_:.4f}")
    print(f"[TUNING] Best params: {search.best_params_}")
    return search.best_estimator_

def full_evaluation(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    mse    = mean_squared_error(y_test, y_pred)
    rmse   = np.sqrt(mse)
    r2     = r2_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(f"  FINAL MODEL EVALUATION — {model_name}")
    print("=" * 55)
    print(f"  MAE   (Mean Absolute Error)  : {mae:.4f}")
    print(f"  MSE   (Mean Squared Error)   : {mse:.4f}")
    print(f"  RMSE  (Root MSE)             : {rmse:.4f}")
    print(f"  R²    (Coefficient of Det.)  : {r2:.4f}")
    print("=" * 55)
    return {"y_pred": y_pred, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def plot_model_comparison(results: list, save_path: str):               #من هنا لحد  252  دوال الرسم (plot_*) — كل واحدة بترسم شارت معين وتحفظه
    names = [r["name"] for r in results]
    r2s   = [r["R2"]   for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, r2s, color=plt.cm.RdYlGn(
        [(v - min(r2s)) / (max(r2s) - min(r2s) + 1e-9) for v in r2s]
    ))
    ax.set_xlabel("Test R² Score")
    ax.set_title("Model Comparison — R² Score", fontsize=14, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
    for bar, val in zip(bars, r2s):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {save_path}")

def plot_feature_importance(model, feature_names: list, save_path: str):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(9, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Top-20 Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {save_path}")

def plot_pred_vs_actual(y_test, y_pred, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="royalblue", label="Predictions")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title("Predicted vs Actual AQI", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {save_path}")

def plot_residuals(y_test, y_pred, save_path: str):
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="darkorange")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Predicted AQI")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title("Residual Plot", fontsize=13, fontweight="bold")

    axes[1].hist(residuals, bins=60, color="mediumseagreen", edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Distribution of Residuals", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {save_path}")

def plot_correlation_heatmap(X_df: pd.DataFrame, save_path: str):
    corr = X_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved: {save_path}")

def plot_shap_summary(model, X_test_sc, feature_names: list, save_path: str):
    print("[SHAP] Computing SHAP values (this may take ~30 seconds) ...")
    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test_sc[:2000])

    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_test_sc[:2000],
                      feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title("SHAP Feature Impact (Beeswarm)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved: {save_path}")

def save_artefacts(model, scaler, encoders: dict,
                    uvi_median: float, feature_names: list):
    joblib.dump(model,        os.path.join(MODELS_DIR, "lgbm_model.pkl"))
    joblib.dump(scaler,       os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders,     os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(feature_names,os.path.join(MODELS_DIR, "feature_names.pkl"))
    joblib.dump(uvi_median,   os.path.join(MODELS_DIR, "uvi_median.pkl"))

    print("\n[SAVE] ✓ Saved to models/:")
    for f in os.listdir(MODELS_DIR):
        print(f"       - {f}")

def main():
    print("\n" + "=" * 65)
    print("  AIR QUALITY PREDICTION — ML PIPELINE")
    print("=" * 65)

    df = load_silver_data(os.path.join(DATA_DIR, "silver_layer.csv"))

    X, y, encoders, uvi_median = prepare_features(df)
    feature_names = X.columns.tolist()

    plot_correlation_heatmap(X, os.path.join(OUT_DIR, "01_correlation_heatmap.png"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[SPLIT] Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    results = compare_models(X_train_sc, X_test_sc, y_train, y_test)
    plot_model_comparison(results, os.path.join(OUT_DIR, "02_model_comparison.png"))

    best_model = tune_best_model(X_train_sc, y_train)

    eval_dict = full_evaluation(best_model, X_test_sc, y_test, "LightGBM (Tuned)")
    y_pred = eval_dict["y_pred"]

    plot_pred_vs_actual(y_test, y_pred, os.path.join(OUT_DIR, "03_pred_vs_actual.png"))
    plot_residuals(y_test, y_pred, os.path.join(OUT_DIR, "04_residuals.png"))
    plot_feature_importance(best_model, feature_names, os.path.join(OUT_DIR, "05_feature_importance.png"))
    plot_shap_summary(best_model, X_test_sc, feature_names, os.path.join(OUT_DIR, "06_shap_summary.png"))

    save_artefacts(best_model, scaler, encoders, uvi_median, feature_names)

    metrics = {
        "model_name":   "LightGBM (Tuned)",
        "MAE":          float(eval_dict["MAE"]),
        "RMSE":         float(eval_dict["RMSE"]),
        "R2":           float(eval_dict["R2"]),
        "n_train_rows": int(X_train.shape[0]),
        "n_test_rows":  int(X_test.shape[0]),
        "trained_at":   pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] ✓ Saved training metrics to {os.path.join(MODELS_DIR, 'metrics.json')}")

    print("\n[DONE] ✓ Pipeline completed. Check outputs/ and models/ folders.\n")

if __name__ == "__main__":
    main()