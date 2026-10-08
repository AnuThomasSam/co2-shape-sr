"""OLS and log-linear (power-law) baselines."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression

from .base import BaseModel


class OLSModel(BaseModel):
    deterministic_slope = True

    def fit(self, fit_df, val_df, full_df, domain=None):
        self._set_columns(full_df)
        self.lr = LinearRegression().fit(self._X(full_df), full_df["co2"].to_numpy())
        return self

    def predict(self, df):
        return self.lr.predict(self._X(df))

    def slope(self, df):
        return np.full(len(df), float(self.lr.coef_[0]))

    def describe(self):
        terms = " + ".join(f"{c:.4g}*{n}" for c, n in zip(self.lr.coef_, self.kept_columns))
        return {"equation": f"{self.lr.intercept_:.4g} + {terms}",
                "n_parameters": len(self.lr.coef_) + 1, "params": {}, "columns": self.kept_columns}


class LogLinearModel(BaseModel):
    """log(CO2) = c + beta * log(engine size) + gamma' z, with Duan's smearing correction."""
    deterministic_slope = True

    def _Z(self, df):
        X = self._X(df)
        X = X.copy()
        X[:, 0] = np.log(X[:, 0])
        return X

    def fit(self, fit_df, val_df, full_df, domain=None):
        self._set_columns(full_df)
        y = np.log(full_df["co2"].to_numpy())
        self.lr = LinearRegression().fit(self._Z(full_df), y)
        resid = y - self.lr.predict(self._Z(full_df))
        self.smearing = float(np.mean(np.exp(resid)))
        return self

    def predict(self, df):
        return np.exp(self.lr.predict(self._Z(df))) * self.smearing

    def slope(self, df):
        return self.predict(df) * self.lr.coef_[0] / df["engine_size"].to_numpy(float)

    def describe(self):
        names = ["log(engine_size)"] + self.kept_columns[1:]
        terms = " + ".join(f"{c:.4g}*{n}" for c, n in zip(self.lr.coef_, names))
        return {"equation": f"{self.smearing:.4g} * exp({self.lr.intercept_:.4g} + {terms})",
                "n_parameters": len(self.lr.coef_) + 2, "params": {"beta_log_engine": float(self.lr.coef_[0])},
                "columns": self.kept_columns}
