"""Shape constraints on the scaled prediction (default: dCO2/d(engine size) >= 0).

A constraint is a pair (j, s) meaning s * df/dx_j >= 0. The CO2 study uses the
single constraint (0, +1); the synthetic benchmark (step 13) uses several.

The constrained quantity is b_k * dg/dx_0, where b_k is the linear-scaling
slope of the point's fuel group and x_0 is engine size. Engine size enters the
tree only through column 0, so dg/dx_0 is the total derivative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scaling import Scaling
from .tree import Node
from .verify import DEFAULT_CONSTRAINTS, _bounds, signed_slopes

BIG = 1.0e6


@dataclass
class PointCheck:
    violation: float       # mean of max(0, -slope), in units of std(y) per litre
    feasible: bool
    share_negative: float
    min_slope: float


def check_points(tree: Node, scaling: Scaling, X: np.ndarray, groups: np.ndarray,
                 y_std: float, tol: float, dv: np.ndarray | None = None,
                 constraints=DEFAULT_CONSTRAINTS) -> PointCheck:
    if dv is not None and tuple(map(tuple, constraints)) == DEFAULT_CONSTRAINTS:
        if not np.all(np.isfinite(dv)):
            return PointCheck(BIG, False, 1.0, -np.inf)
        _, b = scaling.coefficients(groups)
        slope = b * dv
    else:
        slope = signed_slopes(tree, scaling, X, groups, constraints)
        if not np.all(np.isfinite(slope)):
            return PointCheck(BIG, False, 1.0, -np.inf)
    neg = np.maximum(0.0, -slope)
    return PointCheck(float(neg.mean() / y_std), bool(np.all(slope >= -tol)),
                      float(np.mean(slope < -tol)), float(slope.min()) if len(slope) else 0.0)


def check_interval(tree: Node, scaling: Scaling, LO: np.ndarray, HI: np.ndarray,
                   groups: np.ndarray, y_std: float, tol: float, constraints=DEFAULT_CONSTRAINTS) -> PointCheck:
    """Certified check on a fixed partition: the lower bound of the slope over every box must be >= -tol."""
    lb = _bounds(tree, scaling, np.asarray(LO, float), np.asarray(HI, float), np.asarray(groups, int), constraints)
    finite = np.isfinite(lb)
    neg = np.where(finite, np.maximum(0.0, -lb), BIG)
    return PointCheck(float(neg.mean() / y_std), bool(np.all(lb >= -tol)),
                      float(np.mean(~(lb >= -tol))), float(lb.min()) if len(lb) else 0.0)
