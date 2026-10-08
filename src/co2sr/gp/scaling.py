"""Group-wise linear scaling (Keijzer, 2003), one group per fuel type.

For an evolved expression g(x) the prediction is a_k + b_k g(x), where a_k, b_k
are the least-squares coefficients on the training rows of fuel group k.
Groups with too few rows, and groups unseen in training, use pooled
coefficients. The sign of b_k matters for the monotonicity constraint.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scaling:
    a: np.ndarray          # per group intercept
    b: np.ndarray          # per group slope
    a0: float              # pooled intercept
    b0: float              # pooled slope

    def coefficients(self, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        groups = np.asarray(groups)
        valid = (groups >= 0) & (groups < len(self.a))
        idx = np.where(valid, groups, 0)
        a = np.where(valid, self.a[idx], self.a0)
        b = np.where(valid, self.b[idx], self.b0)
        return a, b

    def apply(self, g: np.ndarray, groups: np.ndarray) -> np.ndarray:
        a, b = self.coefficients(groups)
        return a + b * g

    def to_dict(self):
        return {"a": self.a.tolist(), "b": self.b.tolist(), "a0": float(self.a0), "b0": float(self.b0)}

    @staticmethod
    def from_dict(d):
        return Scaling(np.asarray(d["a"], float), np.asarray(d["b"], float), float(d["a0"]), float(d["b0"]))


def _ls(sg, sy, sgg, sgy, n):
    with np.errstate(all="ignore"):
        mg, my = sg / n, sy / n
        var = sgg / n - mg * mg
        cov = sgy / n - mg * my
        ok = var > 1e-12 * np.maximum(1.0, mg * mg)
        b = np.where(ok, cov / np.where(ok, var, 1.0), 0.0)
        a = my - b * mg
    return a, b


def fit_scaling(g: np.ndarray, y: np.ndarray, groups: np.ndarray, n_groups: int,
                min_rows: int, enabled: bool = True, sign_consistent: bool = False) -> Scaling:
    """sign_consistent: a group whose slope b_k has the opposite sign to the pooled slope b0
    takes b0 and a refitted intercept. Fuel type then only rescales a shared shape g(x)
    and cannot reverse it (e.g. more cylinders giving less CO2 for one fuel)."""
    if not enabled:
        return Scaling(np.zeros(n_groups), np.ones(n_groups), 0.0, 1.0)
    with np.errstate(all="ignore"):
        return _fit(g, y, groups, n_groups, min_rows, sign_consistent)


def _fit(g, y, groups, n_groups, min_rows, sign_consistent=False) -> Scaling:
    n = len(g)
    a0, b0 = _ls(g.sum(), y.sum(), (g * g).sum(), (g * y).sum(), n)
    gi = np.where(groups >= 0, groups, n_groups)       # unknown group -> extra bin, ignored
    nb = n_groups + 1
    cnt = np.bincount(gi, minlength=nb)[:n_groups].astype(float)
    sg = np.bincount(gi, weights=g, minlength=nb)[:n_groups]
    sy = np.bincount(gi, weights=y, minlength=nb)[:n_groups]
    sgg = np.bincount(gi, weights=g * g, minlength=nb)[:n_groups]
    sgy = np.bincount(gi, weights=g * y, minlength=nb)[:n_groups]
    a, b = _ls(sg, sy, sgg, sgy, np.maximum(cnt, 1.0))
    small = cnt < min_rows
    a = np.where(small, a0, a)
    b = np.where(small, b0, b)
    if sign_consistent:
        flip = (~small) & (np.sign(b) != np.sign(b0)) & (b != 0)
        n = np.maximum(cnt, 1.0)
        a = np.where(flip, sy / n - b0 * sg / n, a)
        b = np.where(flip, b0, b)
    return Scaling(a, b, float(a0), float(b0))
