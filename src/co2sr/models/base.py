"""Common model interface.

Every method is wrapped so that the training step can call
    fit(fit_df, val_df, full_df, domain)  ->  self
    predict(df)                           ->  predicted CO2 (g/km)
    slope(df)                             ->  d prediction / d engine size (g/km per litre)
    describe()                            ->  dict with equation, complexity, parameters
and save/load it from a run directory. fit_df is the fitting part of the training
window, val_df the inner validation part, full_df their union (used for the
final refit after tuning or selection).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features import design_matrix, ml_variable_names


class BaseModel:
    deterministic_slope = False

    def __init__(self, spec: dict, cfg: dict, seed: int = 0):
        self.spec = spec
        self.cfg = cfg
        self.seed = int(seed)
        self.params: dict = {}
        self.kept_columns: list[str] = []

    # ------------------------------------------------------------ helpers
    def _names(self) -> list[str]:
        return ml_variable_names(self.cfg, self.spec["feature_set"])

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return design_matrix(df, self.kept_columns)

    def _set_columns(self, fit_df: pd.DataFrame) -> None:
        """Drop columns that are constant in the fitting rows (e.g. a fuel absent from training)."""
        names = self._names()
        X = design_matrix(fit_df, names)
        keep = [n for j, n in enumerate(names) if j == 0 or np.ptp(X[:, j]) > 0]
        self.kept_columns = keep

    # ------------------------------------------------------------ interface
    def fit(self, fit_df, val_df, full_df, domain=None):
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def slope(self, df: pd.DataFrame) -> np.ndarray:
        h = float(self.cfg["evaluation"]["finite_difference_h"])
        up, dn = df.copy(), df.copy()
        up["engine_size"] = up["engine_size"] + h
        dn["engine_size"] = dn["engine_size"] - h
        return (self.predict(up) - self.predict(dn)) / (2 * h)

    def describe(self) -> dict:
        return {"equation": None, "n_parameters": None, "params": self.params,
                "columns": self.kept_columns}

    def save(self, run_dir: Path) -> None:
        joblib.dump(self, run_dir / "model.joblib")

    @staticmethod
    def load(run_dir: Path) -> "BaseModel":
        return joblib.load(Path(run_dir) / "model.joblib")
