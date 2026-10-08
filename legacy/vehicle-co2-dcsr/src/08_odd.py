# %% OOD ROBUSTNESS EVALUATION
# Models: DCSR, Standard GP, XGBoost, LightGBM,
#         CatBoost, Random Forest, Ridge Regression

import os
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


# ---------------------------------------------------------
# 1. Paths and settings
# ---------------------------------------------------------

S_old_path = "../data/processed/S_old_temporal_train.csv"
S_new_path = "../data/processed/S_new_temporal_test.csv"

results_dir = "../results/tables"
predictions_dir = "../results/predictions"

os.makedirs(results_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

TARGET = "co2"

NUMERIC_FEATURES = [
    "engine_size",
    "cylinders"
]

CATEGORICAL_FEATURES = [
    "make",
    "model",
    "vehicle_class",
    "transmission",
    "fuel_type"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

OOD_PERCENTILE = 99


# ---------------------------------------------------------
# 2. Load the same temporal train/test datasets
# ---------------------------------------------------------

train_df = pd.read_csv(S_old_path)
test_df = pd.read_csv(S_new_path)

print("Training shape:", train_df.shape)
print("Test shape:", test_df.shape)

required_columns = ALL_FEATURES + [TARGET]

missing_train = [
    col for col in required_columns if col not in train_df.columns
]
missing_test = [
    col for col in required_columns if col not in test_df.columns
]

if missing_train or missing_test:
    raise ValueError(
        f"Missing columns. Train: {missing_train}; Test: {missing_test}"
    )

X_train = train_df[ALL_FEATURES].copy()
X_test = test_df[ALL_FEATURES].copy()

y_train = train_df[TARGET].to_numpy()
y_test = test_df[TARGET].to_numpy()


# ---------------------------------------------------------
# 3. Define OOD subset using training data only
# ---------------------------------------------------------

engine_size_threshold = train_df["engine_size"].quantile(
    OOD_PERCENTILE / 100
)

test_engine_size = test_df["engine_size"].to_numpy()

# Only test observations with a known engine size can be
# classified using this threshold.
ood_mask = (
    np.isfinite(test_engine_size)
    & (test_engine_size >= engine_size_threshold)
)

print("\nOOD threshold")
print(f"{OOD_PERCENTILE}th percentile of training engine_size: "
      f"{engine_size_threshold:.3f} L")

print("Full temporal test observations:", len(test_df))
print("OOD test observations:", int(ood_mask.sum()))

if ood_mask.sum() == 0:
    raise ValueError(
        "No OOD observations found. Check the engine_size values "
        "and the training-derived threshold."
    )


# ---------------------------------------------------------
# 4. Preprocessing for conventional ML models
# ---------------------------------------------------------

# Numeric columns: replace missing values with training medians.
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical columns: impute missing values, then one-hot encode.
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
    ]
)


# ---------------------------------------------------------
# 5. Define conventional ML models
# ---------------------------------------------------------

conventional_models = {
    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    ),

    "Random_Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1
    ),

    "Ridge": Ridge(
        alpha=1.0
    )
}


# ---------------------------------------------------------
# 6. Fit conventional models on S_old and predict S_new
# ---------------------------------------------------------

predictions = {}
training_times = {}
prediction_times = {}
fitted_models = {}

for model_name, estimator in conventional_models.items():

    print(f"\nTraining {model_name}...")

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ]
    )

    start_time = time.time()
    model_pipeline.fit(X_train, y_train)
    training_times[model_name] = time.time() - start_time

    start_time = time.time()
    y_pred = model_pipeline.predict(X_test)
    prediction_times[model_name] = time.time() - start_time

    predictions[model_name] = np.asarray(y_pred)
    fitted_models[model_name] = model_pipeline

    print(f"{model_name} training time: "
          f"{training_times[model_name]:.2f} seconds")


# ---------------------------------------------------------
# 7. Predict with symbolic models
# ---------------------------------------------------------
# These objects must already be fitted on S_old:
#     dcsr_model
#     gp_model
#
# They use only engine_size and cylinders, so prepare those
# inputs separately and apply training-only median imputation.

symbolic_train = (
    train_df[NUMERIC_FEATURES]
    .replace([np.inf, -np.inf], np.nan)
)

symbolic_test = (
    test_df[NUMERIC_FEATURES]
    .replace([np.inf, -np.inf], np.nan)
)

symbolic_medians = symbolic_train.median()

X_test_symbolic = (
    symbolic_test
    .fillna(symbolic_medians)
    .to_numpy()
)

# DCSR prediction
start_time = time.time()
predictions["DCSR"] = np.asarray(
    dcsr_model.predict(X_test_symbolic)
)
prediction_times["DCSR"] = time.time() - start_time

# Standard GP prediction
start_time = time.time()
predictions["Standard_GP"] = np.asarray(
    gp_model.predict(X_test_symbolic)
)
prediction_times["Standard_GP"] = time.time() - start_time

# Training times should be entered from your model-training
# output, if you want them included in the final table.
training_times["DCSR"] = np.nan
training_times["Standard_GP"] = np.nan


# ---------------------------------------------------------
# 8. Metric calculation
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    n = len(y_true)

    if n == 0:
        return {
            "N": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan
        }

    r2 = r2_score(y_true, y_pred) if n >= 2 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {
        "N": n,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }


# ---------------------------------------------------------
# 9. Evaluate all models on overall test and OOD subset
# ---------------------------------------------------------

results = []
prediction_rows = []

for model_name, y_pred in predictions.items():

    y_pred = np.asarray(y_pred)

    # Overall S_new test results
    overall_metrics = calculate_metrics(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Evaluation_Set": "Overall_S_new",
        **overall_metrics,
        "Training_Time_sec": training_times.get(model_name, np.nan),
        "Prediction_Time_sec": prediction_times.get(model_name, np.nan)
    })

    # OOD subset results
    ood_metrics = calculate_metrics(
        y_test[ood_mask],
        y_pred[ood_mask]
    )

    results.append({
        "Model": model_name,
        "Evaluation_Set": "OOD_Engine_Size_99th_Percentile",
        **ood_metrics,
        "Training_Time_sec": training_times.get(model_name, np.nan),
        "Prediction_Time_sec": prediction_times.get(model_name, np.nan)
    })

    # Save predictions and residuals
    model_prediction_df = pd.DataFrame({
        "Model": model_name,
        "engine_size": test_engine_size,
        "Actual_CO2": y_test,
        "Predicted_CO2": y_pred,
        "Residual": y_test - y_pred,
        "Absolute_Error": np.abs(y_test - y_pred),
        "Is_OOD": ood_mask
    })

    prediction_rows.append(model_prediction_df)


# ---------------------------------------------------------
# 10. Save tables and predictions
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

predictions_df = pd.concat(
    prediction_rows,
    ignore_index=True
)

results_path = os.path.join(
    results_dir,
    "ood_robustness_all_models.csv"
)

predictions_path = os.path.join(
    predictions_dir,
    "ood_predictions_all_models.csv"
)

results_df.to_csv(results_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

print("\n========== OOD ROBUSTNESS RESULTS ==========")
print(results_df.to_string(index=False))

print("\nResults saved:")
print(results_path)

print("\nPredictions saved:")
print(predictions_path)