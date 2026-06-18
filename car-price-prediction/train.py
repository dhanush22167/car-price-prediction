"""
train.py
--------
Trains a used-car price prediction model.

Pipeline:
1. Load raw data
2. Select & clean relevant features
3. Build a preprocessing + model Pipeline (impute -> encode/scale -> regressor)
4. Train & compare Linear Regression, Random Forest, and XGBoost
5. Save the best pipeline (model.pkl) + UI metadata (options.json) + metrics (metrics.json)
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = "data/car_data.csv"
MODEL_PATH = "models/model.pkl"
OPTIONS_PATH = "models/options.json"
METRICS_PATH = "models/metrics.json"

NUMERIC_FEATURES = ["Year", "Engine HP", "Engine Cylinders", "Number of Doors",
                     "highway MPG", "city mpg"]
CATEGORICAL_FEATURES = ["Make", "Transmission Type", "Driven_Wheels",
                         "Vehicle Size", "Vehicle Style"]
TARGET = "MSRP"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]].copy()
    # remove a handful of extreme outliers (e.g. hyper-cars) so the model
    # isn't dragged around by a few $1M+ listings
    df = df[df[TARGET] <= 200_000]
    return df


def build_pipeline(model):
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def evaluate(pipeline, X_test, y_test_log):
    pred_log = pipeline.predict(X_test)
    pred = np.expm1(pred_log)
    actual = np.expm1(y_test_log)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    r2 = float(r2_score(actual, pred))
    return rmse, r2


def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_log = np.log1p(df[TARGET])  # log-transform: price is right-skewed

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ),
    }

    results = {}
    best_name, best_pipeline, best_rmse = None, None, float("inf")

    for name, model in candidates.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        rmse, r2 = evaluate(pipe, X_test, y_test)
        results[name] = {"rmse": round(rmse, 2), "r2": round(r2, 4)}
        print(f"{name:18s}  RMSE: ${rmse:,.0f}   R2: {r2:.4f}")
        if rmse < best_rmse:
            best_name, best_pipeline, best_rmse = name, pipe, rmse

    print(f"\nBest model: {best_name}")

    joblib.dump(best_pipeline, MODEL_PATH)

    # Save dropdown options + numeric ranges so the Streamlit app can
    # build its input form directly from the training data
    options = {
        "best_model": best_name,
        "numeric_ranges": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
            }
            for col in NUMERIC_FEATURES
        },
        "categorical_options": {
            col: sorted(df[col].dropna().unique().tolist())
            for col in CATEGORICAL_FEATURES
        },
    }
    with open(OPTIONS_PATH, "w") as f:
        json.dump(options, f, indent=2)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved UI options -> {OPTIONS_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
