"""Metrics (plan, Section 8)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def accuracy(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = np.isfinite(yhat)
    if len(y) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "n_nonfinite": 0}
    e = np.where(ok, y - yhat, np.nan)
    ss = np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.nanmean(e ** 2))) if ok.any() else np.nan
    return {"n": int(len(y)), "rmse": rmse, "mae": float(np.nanmean(np.abs(e))) if ok.any() else np.nan,
            "r2": float(1 - np.nansum(e ** 2) / ss) if ss > 0 and ok.all() else np.nan,
            "n_nonfinite": int((~ok).sum())}


def slope_summary(slope: np.ndarray, tol: float) -> dict:
    s = np.asarray(slope, float)
    fin = np.isfinite(s)
    if len(s) == 0:
        return {"violation_rate": np.nan, "min_slope": np.nan}
    return {"violation_rate": float(np.mean(~(s >= -tol))),       # non-finite counts as violation
            "min_slope": float(np.min(s[fin])) if fin.any() else np.nan}


def information_ceiling(test: pd.DataFrame, keys: list[str]) -> float:
    """R^2 of the best possible predictor that uses only `keys`: the group mean on the test set itself."""
    if len(test) < 2:
        return np.nan
    t = test.copy()
    t["gears"] = t["gears"].fillna(-1)
    pred = t.groupby(keys, dropna=False)["co2"].transform("mean")
    ss = ((t["co2"] - t["co2"].mean()) ** 2).sum()
    return float(1 - ((t["co2"] - pred) ** 2).sum() / ss) if ss > 0 else np.nan


CEILING_KEYS = {
    "F1": ["engine_size", "cylinders"],
    "F2": ["engine_size", "cylinders", "transmission_type", "gears", "fuel_type"],
    "F3": ["engine_size", "cylinders", "transmission_type", "gears", "fuel_type", "body_group"],
}


def _to_sympy_aq(tree, names):
    """SymPy expression with the analytic quotient kept as one unevaluated function aq(a, b).
    Expanding aq(a, b) into a / sqrt(1 + b**2) adds four or five SymPy nodes per quotient and made
    the "simplified" equation larger than the evolved tree."""
    import sympy as sp
    AQ = sp.Function("aq")
    if tree.op == "const":
        return sp.Float(tree.value, 17)
    if tree.op == "var":
        return sp.Symbol(names[tree.value])
    c = [_to_sympy_aq(ch, names) for ch in tree.children]
    return {"add": lambda: c[0] + c[1], "sub": lambda: c[0] - c[1], "mul": lambda: c[0] * c[1],
            "aq": lambda: AQ(c[0], c[1]), "exp": lambda: sp.exp(c[0]), "log": lambda: sp.log(c[0]),
            "sqrt": lambda: sp.sqrt(c[0])}[tree.op]()


def _count_nodes(expr) -> int:
    """Nodes of a SymPy expression counted like a GP tree: n-ary sums and products count as n-1
    binary operators, a negation (-1 * u) counts as one subtraction."""
    import sympy as sp
    if expr.is_Atom:
        return 1
    args = expr.args
    if isinstance(expr, (sp.Add, sp.Mul)):
        if isinstance(expr, sp.Mul) and len(args) == 2 and args[0] == -1:
            return _count_nodes(args[1]) + 1
        return sum(_count_nodes(a) for a in args) + (len(args) - 1)
    return 1 + sum(_count_nodes(a) for a in args)


def simplify_gp(tree_dict: dict, names: list[str], timeout_nodes: int = 60) -> dict:
    """Simplified form of g(x) and its size.

    The simplification keeps aq as a function, so it only merges constants and cancels terms; it
    cannot make the equation look larger than the evolved tree. n_nodes (the tree size) remains the
    primary complexity measure; n_nodes_simplified is never larger than it except by rounding.
    """
    import sympy as sp

    from .gp.tree import Node
    t = Node.from_dict(tree_dict)
    expr = _to_sympy_aq(t, names)
    try:
        simp = sp.simplify(expr) if t.size() <= timeout_nodes else expr
        if _count_nodes(simp) > _count_nodes(expr):
            simp = expr
    except Exception:
        simp = expr
    simp_r = simp.xreplace({n: sp.Float(round(float(n), 4)) for n in simp.atoms(sp.Float)})
    return {"equation_simplified": str(simp_r),
            "n_nodes_simplified": int(min(_count_nodes(simp), t.size())),
            "n_ops_simplified": int(sp.count_ops(simp))}
