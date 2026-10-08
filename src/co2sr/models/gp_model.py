"""Adapter between the data frames of the pipeline and the GP engine."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import design_matrix, fuel_group_index, fuel_levels, uses_fuel, variable_names
from ..gp.engine import GPRegressor
from ..io import write_json
from .base import BaseModel


class GPModel(BaseModel):
    deterministic_slope = True

    def _setup(self):
        s = self.spec
        fuel_scaling = s["fuel_mode"] == "scaling" and uses_fuel(self.cfg, s["feature_set"])
        self.fuel_scaling = fuel_scaling
        self.names = variable_names(self.cfg, s["feature_set"], s["fuel_mode"])
        self.n_groups = len(fuel_levels(self.cfg)) if fuel_scaling else 1

    def _groups(self, df):
        if self.fuel_scaling:
            return fuel_group_index(df, self.cfg)
        return np.zeros(len(df), int)

    def _M(self, df):
        return design_matrix(df, self.names)

    def fit(self, fit_df, val_df, full_df, domain=None):
        self._setup()
        kw = {}
        mode = self.spec["constraint"]
        if mode in ("sample", "feasibility"):
            q = domain["sample"]
            kw.update(X_q=self._M(q), g_q=self._groups(q))
        if mode == "interval":
            lo, hi = domain["boxes_lo"], domain["boxes_hi"]
            kw.update(LO=self._M(lo), HI=self._M(hi), g_box=self._groups(lo))
        if mode == "verified":
            lo, hi = domain["boxes_verify_lo"], domain["boxes_verify_hi"]
            kw.update(LO=self._M(lo), HI=self._M(hi), g_box=self._groups(lo))
        self.gp = GPRegressor(self.spec, self.seed).fit(
            self._M(fit_df), fit_df["co2"].to_numpy(float), self._groups(fit_df), self.n_groups,
            X_val=self._M(val_df), y_val=val_df["co2"].to_numpy(float), g_val=self._groups(val_df),
            X_full=self._M(full_df), y_full=full_df["co2"].to_numpy(float), g_full=self._groups(full_df),
            **kw)
        self.kept_columns = self.names
        self.params = dict(self.gp.info)
        return self

    def predict(self, df):
        return self.gp.predict(self._M(df), self._groups(df))

    def slope(self, df):
        return self.gp.slope(self._M(df), self._groups(df))

    def certify(self, lo: pd.DataFrame, hi: pd.DataFrame) -> dict:
        return self.gp.certify(self._M(lo), self._M(hi), self._groups(lo))

    def verify_final(self, lo: pd.DataFrame, hi: pd.DataFrame, max_boxes: int, max_depth: int) -> dict:
        return self.gp.verify_final(self._M(lo), self._M(hi), self._groups(lo), max_boxes, max_depth)

    def history(self) -> pd.DataFrame:
        return pd.DataFrame(self.gp.history)

    def equation(self) -> str:
        return self.gp.tree.to_string(self.names)

    def describe(self):
        t = self.gp.tree
        levels = fuel_levels(self.cfg) if self.fuel_scaling else ["all"]
        return {
            "equation": self.equation(),
            "scaling": {lv: {"a": float(a), "b": float(b)} for lv, a, b in
                        zip(levels, self.gp.scaling.a, self.gp.scaling.b)},
            "scaling_pooled": {"a": self.gp.scaling.a0, "b": self.gp.scaling.b0},
            "n_nodes": t.size(), "depth": t.depth(), "n_constants": t.n_constants(),
            "variables_used": sorted(self.names[i] for i in t.variables()),
            "n_parameters": t.n_constants() + 2 * self.n_groups,
            "params": self.params, "columns": self.names,
        }

    def save(self, run_dir: Path) -> None:
        super().save(run_dir)
        write_json({"names": self.names, "fuel_scaling": self.fuel_scaling,
                    "fuel_levels": fuel_levels(self.cfg), **self.gp.to_dict(),
                    "equation": self.equation()}, run_dir / "model.json")
