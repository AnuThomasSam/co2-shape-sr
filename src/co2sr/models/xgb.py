"""XGBoost, unconstrained or with a monotone constraint on engine size.

Hyperparameters are tuned with Optuna on the inner validation set only; the
final model is refitted on fit + validation rows with the selected values.
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel


class XGBModel(BaseModel):
    def _make(self, params):
        from xgboost import XGBRegressor
        kw = dict(params)
        if self.spec.get("monotone", False):
            cons = [1] + [0] * (len(self.kept_columns) - 1)     # engine size is column 0
            kw["monotone_constraints"] = "(" + ",".join(map(str, cons)) + ")"
        return XGBRegressor(**kw, random_state=self.seed, n_jobs=1, verbosity=0, tree_method="hist")

    def fit(self, fit_df, val_df, full_df, domain=None):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._set_columns(full_df)
        Xf, yf = self._X(fit_df), fit_df["co2"].to_numpy()
        Xv, yv = self._X(val_df), val_df["co2"].to_numpy()

        def objective(trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            m = self._make(p).fit(Xf, yf)
            return float(np.sqrt(np.mean((m.predict(Xv) - yv) ** 2)))

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=int(self.spec.get("n_trials", 40)))
        self.params = dict(study.best_params)
        self.params_val_rmse = float(study.best_value)
        self.model = self._make(self.params).fit(self._X(full_df), full_df["co2"].to_numpy())
        return self

    def predict(self, df):
        return self.model.predict(self._X(df)).astype(float)

    def describe(self):
        return {"equation": None, "n_parameters": None,
                "params": {**self.params, "inner_val_rmse": self.params_val_rmse},
                "columns": self.kept_columns}
