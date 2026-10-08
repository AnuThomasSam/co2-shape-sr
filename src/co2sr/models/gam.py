"""Shape-constrained GAM: monotone increasing spline in engine size, linear terms otherwise."""
from __future__ import annotations

import numpy as np

from .base import BaseModel


class MonotoneGAM(BaseModel):
    def _build(self, lam):
        from pygam import LinearGAM, l, s
        terms = s(0, constraints="monotonic_inc", n_splines=int(self.spec.get("n_splines", 12)))
        for j in range(1, len(self.kept_columns)):
            terms = terms + l(j)
        return LinearGAM(terms, lam=lam)

    def fit(self, fit_df, val_df, full_df, domain=None):
        self._set_columns(full_df)
        Xf, yf = self._X(fit_df), fit_df["co2"].to_numpy()
        Xv, yv = self._X(val_df), val_df["co2"].to_numpy()
        scores = {}
        for lam in self.spec.get("lam_grid", [0.1, 1, 10, 100]):
            try:
                g = self._build(lam).fit(Xf, yf)
                scores[lam] = float(np.sqrt(np.mean((g.predict(Xv) - yv) ** 2)))
            except Exception:                      # numerical failure for this lam
                scores[lam] = np.inf
        best = min(scores, key=scores.get)
        self.params = {"lam": best, "val_rmse_by_lam": scores}
        self.gam = self._build(best).fit(self._X(full_df), full_df["co2"].to_numpy())
        return self

    def predict(self, df):
        return self.gam.predict(self._X(df))

    def describe(self):
        return {"equation": None, "n_parameters": int(round(self.gam.statistics_["edof"])),
                "params": self.params, "columns": self.kept_columns}
