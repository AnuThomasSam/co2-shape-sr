# %%
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import optuna
from lightgbm import early_stopping

optuna.logging.set_verbosity(optuna.logging.WARNING)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
# Load the cleaned dataset
data_clean = pd.read_csv("../data/processed/vehicle_co2_clean.csv")

print("Dataset shape:", data_clean.shape)
print("\nColumns:")
print(data_clean.columns.tolist())

# %%
# Feature Engineering

USE_ENGINEERED_FEATURES = True  # Set to True to use engineered features, False to use only original features 

categorical_features = [
    "make",
    "model",
    "vehicle_class",
    "transmission",
    "fuel_type"
]

original_numeric_features = [
    "engine_size",
    "cylinders",
]

engineered_features = [
    "engine_per_cylinder",
    "engine_x_cylinders",
    "engine_size_sq",
    "cylinders_sq"
]


if USE_ENGINEERED_FEATURES:

    numeric_features = (
        original_numeric_features
        + engineered_features
    )

else:
    numeric_features = original_numeric_features

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# %%
# Check required columns

required_columns = (
    numeric_features
    + categorical_features
    + ["co2"]
)

missing_columns = [
    col for col in required_columns
    if col not in data_clean.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nAll required columns are available.")

# %%
# Define X and y

X = data_clean[numeric_features + categorical_features].copy()
y = data_clean["co2"].copy()

print("\nX shape:", X.shape)
print("y shape:", y.shape)

# %%
# Train / test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# %%
# Standard preprocessing pipeline (one-hot encoding)

preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# %%
# OPTUNA Helper

def run_optuna(objective, n_trials=100):

    sampler = optuna.samplers.TPESampler(
        seed=42
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler
    )

    study.optimize(
        objective,
        n_trials=n_trials
    )

    return study

# %% [markdown]
# Evaluation Function

# %%
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, cv_scores):
    
    # Fit and time training
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    # Predict and time prediction
    start = time.time()
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    prediction_time = time.time() - start

    # Metrics
    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return r2_score(y_true, y_pred), mse, np.sqrt(mse), mean_absolute_error(y_true, y_pred)

    train_r2, train_mse, train_rmse, train_mae = metrics(y_train, y_train_pred)
    test_r2,  test_mse,  test_rmse,  test_mae  = metrics(y_test, y_test_pred)

    cv_mean = np.mean(cv_scores)
    cv_std  = np.std(cv_scores)

    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Test  R²: {test_r2:.4f}  MSE: {test_mse:.4f}  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"CV    R²: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Train time: {training_time:.4f}s  |  Predict time: {prediction_time:.4f}s")

    return {
        "Model":                    model_name,
        "Train_R2":                 train_r2,
        "Test_R2":                  test_r2,
        "Train_MSE":                train_mse,
        "Test_MSE":                 test_mse,
        "Train_RMSE":               train_rmse,
        "Test_RMSE":                test_rmse,
        "Train_MAE":                train_mae,
        "Test_MAE":                 test_mae,
        "CV_R2_Mean":               cv_mean,
        "CV_R2_Std":                cv_std,
        "Training_Time_Seconds":    training_time,
        "Prediction_Time_Seconds":  prediction_time
    }

def evaluate_catboost(model_name, model, X_train, X_test, y_train, y_test,
                      cat_indices, cv_scores):
    
    start = time.time()
    model.fit(X_train, y_train, cat_features=cat_indices)
    training_time = time.time() - start

    start = time.time()
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    prediction_time = time.time() - start

    def metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return r2_score(y_true, y_pred), mse, np.sqrt(mse), mean_absolute_error(y_true, y_pred)

    train_r2, train_mse, train_rmse, train_mae = metrics(y_train, y_train_pred)
    test_r2,  test_mse,  test_rmse,  test_mae  = metrics(y_test, y_test_pred)

    cv_mean = np.mean(cv_scores)
    cv_std  = np.std(cv_scores)

    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    print(f"Test  R²: {test_r2:.4f}  MSE: {test_mse:.4f}  RMSE: {test_rmse:.4f}  MAE: {test_mae:.4f}")
    print(f"Train R²: {train_r2:.4f}")
    print(f"CV    R²: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Train time: {training_time:.4f}s  |  Predict time: {prediction_time:.4f}s")

    return {
        "Model":                    model_name,
        "Train_R2":                 train_r2,
        "Test_R2":                  test_r2,
        "Train_MSE":                train_mse,
        "Test_MSE":                 test_mse,
        "Train_RMSE":               train_rmse,
        "Test_RMSE":                test_rmse,
        "Train_MAE":                train_mae,
        "Test_MAE":                 test_mae,
        "CV_R2_Mean":               cv_mean,
        "CV_R2_Std":                cv_std,
        "Training_Time_Seconds":    training_time,
        "Prediction_Time_Seconds":  prediction_time
    }

# %%
# Random Forest

def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }

    model = RandomForestRegressor(**params, random_state=42)

    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
    return scores.mean()

# Run Optuna
rf_study = run_optuna(objective_rf)
print("Best Params:", rf_study.best_params)

# Build tuned model
best_rf_model = RandomForestRegressor(
    **rf_study.best_params,
    random_state=42
)

best_rf_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', best_rf_model)
])

rf_cv_scores = cross_val_score(
    best_rf_pipeline,
    X_train,
    y_train,
    cv=kf,
    scoring="r2"
)

# Evaluate tuned model
rf_results = evaluate_model(
    "Random Forest (Tuned)",
    best_rf_pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    rf_cv_scores
)

# %%
# XGBoost Optuna Tuning

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5)
    }

    model = XGBRegressor(**params, random_state=42, verbosity=0)

    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
    return scores.mean()


# Run Optuna
xgb_study = run_optuna(objective_xgb)
print("Best Params:", xgb_study.best_params)

# Build Tuned XGBoost Model

best_xgb_model = XGBRegressor(
    **xgb_study.best_params,
    random_state=42,
    verbosity=0
)

best_xgb_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', best_xgb_model)
])

xgb_cv_scores = cross_val_score(
    best_xgb_pipeline,
    X_train,
    y_train,
    cv=kf,
    scoring="r2"
)

# Evaluate Tuned XGBoost Model

xgb_results = evaluate_model(
    "XGBoost (Tuned)",
    best_xgb_pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    xgb_cv_scores
)

# %%
# Ridge Regression Optuna Tuning

# 1. Optuna Objective Function

def objective_ridge(trial):

    params = {"alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)}

    # Create Ridge model
    model = Ridge(
        **params
    )

    # Create pipeline
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model)
    ])

    # 5-fold cross-validation
    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=kf,
        scoring="r2"
    )

    # Return mean CV R²
    return scores.mean()

# 2. Run Optuna

ridge_study = run_optuna(objective_ridge)

print("Best Ridge Parameters:")
print(ridge_study.best_params)

# 3. Build Tuned Ridge Model

best_ridge_model = Ridge(
    **ridge_study.best_params
)

# 4. Create Ridge Pipeline

best_ridge_pipeline = Pipeline([
    ("prep", preprocess),
    ("model", best_ridge_model)
])


# 5. Fit Ridge Model

ridge_cv_scores = cross_val_score(
    best_ridge_pipeline,
    X_train,
    y_train,
    cv=kf,
    scoring="r2"
)

# 6. Evaluate Ridge Model

ridge_results = evaluate_model(
    "Ridge Regression(Tuned)",
    best_ridge_pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    ridge_cv_scores
)

# %%
# LightGBM - Native Categorical Handling

X_train_lgbm = X_train.copy()
X_test_lgbm  = X_test.copy()

for col in categorical_features:
    cats = X_train[col].astype("category").cat.categories
    X_train_lgbm[col] = pd.Categorical(X_train[col], categories=cats)
    X_test_lgbm[col]  = pd.Categorical(X_test[col],  categories=cats)

def objective_lgbm_native(trial):
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "num_leaves":    trial.suggest_int("num_leaves", 15, 63)
    }
    model = LGBMRegressor(**params, random_state=42, verbosity=-1, n_jobs=1)
    return cross_val_score(model, X_train_lgbm, y_train, cv=kf, scoring="r2").mean()

lgbm_native_study = run_optuna(objective_lgbm_native)
print("Best LightGBM (Native) params:", lgbm_native_study.best_params)

best_lgbm_native = LGBMRegressor(
    **lgbm_native_study.best_params, random_state=42, verbosity=-1, n_jobs=1)

lgbm_native_cv_scores = cross_val_score(
    best_lgbm_native, X_train_lgbm, y_train, cv=kf, scoring="r2")

lgbm_native_results = evaluate_model(
    "LightGBM (Native categorical)",
    best_lgbm_native,
    X_train_lgbm, X_test_lgbm,
    y_train, y_test,
    lgbm_native_cv_scores)

# %%
# LightGBM - Tuned

def objective_lgbm(trial):
    params = {
        "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth":     trial.suggest_int("max_depth", 3, 10),
        "num_leaves":    trial.suggest_int("num_leaves", 15, 63)
    }
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", LGBMRegressor(**params, random_state=42, verbosity=-1))
    ])
    return cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2").mean()

lgbm_study = run_optuna(objective_lgbm)
print("Best LightGBM", lgbm_study.best_params)

best_lgbm_ohe_pipeline = Pipeline([
    ("prep", preprocess),
    ("model", LGBMRegressor(**lgbm_study.best_params, random_state=42, verbosity=-1))
])

lgbm_ohe_cv_scores = cross_val_score(
    best_lgbm_ohe_pipeline, X_train, y_train, cv=kf, scoring="r2")

lgbm_ohe_results = evaluate_model(
    "LightGBM (Tuned)",
    best_lgbm_ohe_pipeline, X_train, X_test, y_train, y_test, lgbm_ohe_cv_scores)

# %%
# CatBoost - Native Categorical Handling

X_train_catboost = X_train.copy()
X_test_catboost = X_test.copy()

for col in categorical_features:
    X_train_catboost[col] = X_train_catboost[col].fillna("Missing").astype(str)
    X_test_catboost[col]  = X_test_catboost[col].fillna("Missing").astype(str)


cat_feature_indices = [
    X_train_catboost.columns.get_loc(col)
    for col in categorical_features
]

print("CatBoost categorical columns:")
print(categorical_features)

print("\nCatBoost categorical indices:")
print(cat_feature_indices)

print("\nCatBoost feature order:")
print(X_train_catboost.columns.tolist())

# %%
# CatBoost Optuna Tuning

def objective_catboost_native(trial):
    params = {
        "iterations":    trial.suggest_int("iterations", 100, 500),
        "depth":         trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1, 10)
    }

    scores = []
    for train_idx, val_idx in kf.split(X_train_catboost):
        X_tr  = X_train_catboost.iloc[train_idx]
        X_val = X_train_catboost.iloc[val_idx]
        y_tr  = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        model = CatBoostRegressor(**params, random_seed=42, verbose=False, thread_count=-1, loss_function="RMSE")
        model.fit(X_tr, y_tr, cat_features=cat_feature_indices, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)
        scores.append(r2_score(y_val, model.predict(X_val)))

    return np.mean(scores)

# Run Optuna
catboost_native_study = run_optuna(objective_catboost_native)
print("Best CatBoost (Native) params:", catboost_native_study.best_params)

# CV scores with best params
catboost_native_cv_scores = []
for train_idx, val_idx in kf.split(X_train_catboost):
    X_tr  = X_train_catboost.iloc[train_idx]
    X_val = X_train_catboost.iloc[val_idx]
    y_tr  = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    m = CatBoostRegressor(
        **catboost_native_study.best_params, random_seed=42, verbose=False)
    m.fit(X_tr, y_tr, cat_features=cat_feature_indices)
    catboost_native_cv_scores.append(r2_score(y_val, m.predict(X_val)))

best_catboost_native = CatBoostRegressor(
    **catboost_native_study.best_params, random_seed=42, verbose=False)

catboost_native_results = evaluate_catboost(
    "CatBoost (Native categorical)",
    best_catboost_native,
    X_train_catboost, X_test_catboost,
    y_train, y_test,
    cat_feature_indices,
    catboost_native_cv_scores
)

# %%
# CatBoost - Tuned

def objective_catboost(trial):
    params = {
        "iterations":    trial.suggest_int("iterations", 100, 500),
        "depth":         trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg":   trial.suggest_float("l2_leaf_reg", 1, 10)
    }
    pipe = Pipeline([
        ("prep", preprocess),
        ("model", CatBoostRegressor(**params, random_seed=42, verbose=False))
    ])
    return cross_val_score(pipe, X_train, y_train, cv=kf, scoring="r2").mean()

catboost_study = run_optuna(objective_catboost)
print("Best CatBoost (OHE) params:", catboost_study.best_params)

best_catboost_ohe_pipeline = Pipeline([
    ("prep", preprocess),
    ("model", CatBoostRegressor(
        **catboost_study.best_params, random_seed=42, verbose=False))
])

catboost_ohe_cv_scores = cross_val_score(
    best_catboost_ohe_pipeline, X_train, y_train, cv=kf, scoring="r2")

catboost_ohe_results = evaluate_model(
    "CatBoost (Tuned)",
    best_catboost_ohe_pipeline,
    X_train, X_test, y_train, y_test,
    catboost_ohe_cv_scores
)

# %%
# Summary table for all models

summary_df = pd.DataFrame([
    rf_results,
    xgb_results,
    ridge_results,
    lgbm_ohe_results,
    lgbm_native_results,
    catboost_ohe_results,
    catboost_native_results
])

# Drop CV columns
# summary_df_clean = summary_df.drop(columns=['CV_R2_Mean', 'CV_R2_Std'])

# Sort by Test_R2 (best model first)
summary_df_clean = summary_df.sort_values(by='Test_R2', ascending=False)

# Round for cleaner presentation
summary_df_clean = summary_df_clean.round(4)

# Reset index
summary_df_clean.reset_index(drop=True, inplace=True)

if USE_ENGINEERED_FEATURES:
    filename = "model_comparison_engineered_features.csv"
    print("Saving results: AFTER engineered features")
else:
    filename = "model_comparison_original_features.csv"
    print("Saving results: BEFORE engineered features")


# Save table
summary_df_clean.to_csv(
    f"../results/tables/{filename}",
    index=False
)

summary_df_clean


