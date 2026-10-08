"""PySR (Cranmer, 2023) as an external, unconstrained SR baseline.

PySR needs a Julia runtime, which it installs on first import. If PySR cannot be
imported, the training step records the run as skipped and continues.
The equation is chosen from PySR's hall of fame by inner-validation RMSE.
"""
from __future__ import annotations

import numpy as np

from .base import BaseModel


def pysr_available() -> tuple[bool, str]:
    try:
        import pysr  # noqa: F401
        return True, ""
    except Exception as exc:                       # ImportError or Julia install failure
        return False, f"{type(exc).__name__}: {exc}"


class PySRModel(BaseModel):
    def fit(self, fit_df, val_df, full_df, domain=None):
        from pysr import PySRRegressor
        self._set_columns(full_df)
        Xf, yf = self._X(fit_df), fit_df["co2"].to_numpy()
        Xv, yv = self._X(val_df), val_df["co2"].to_numpy()
        self.model = PySRRegressor(
            niterations=int(self.spec.get("niterations", 40)),
            binary_operators=["+", "-", "*", "/"],
            unary_operators=[],
            maxsize=int(self.spec.get("maxsize", 30)),
            timeout_in_seconds=int(self.spec.get("timeout_s", 600)),
            random_state=self.seed, deterministic=True, parallelism="serial",
            progress=False, verbosity=0, model_selection="accuracy",
        )
        self.model.fit(Xf, yf, variable_names=[c.replace("-", "_") for c in self.kept_columns])
        eqs = self.model.equations_
        rmse = []
        for i in range(len(eqs)):
            p = self.model.predict(Xv, index=i)
            rmse.append(float(np.sqrt(np.mean((p - yv) ** 2))) if np.all(np.isfinite(p)) else np.inf)
        self.index = int(np.argmin(rmse))
        self.params = {"selected_index": self.index, "val_rmse": rmse[self.index]}
        return self

    def predict(self, df):
        return np.asarray(self.model.predict(self._X(df), index=self.index), float)

    def describe(self):
        eq = self.model.equations_.iloc[self.index]
        return {"equation": str(eq["equation"]), "n_parameters": None,
                "params": {**self.params, "complexity": int(eq["complexity"])},
                "columns": self.kept_columns}
