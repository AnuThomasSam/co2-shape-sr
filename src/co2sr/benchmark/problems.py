"""Synthetic shape-constrained regression problems for step 13.

The expressions are equations from the Feynman Symbolic Regression Database
(Udrescu and Tegmark, 2020), the source also used by Kronberger et al. (2022)
and Haider et al. (2023) for shape-constrained SR. The input ranges below are
chosen for this study (they are not copied from those papers): every range
keeps the expression real and finite.

The shape constraints are NOT typed in by hand. derive_constraints()
differentiates each expression symbolically and keeps (j, s) only if
s * df/dx_j >= 0 at every point of a dense sample of the full domain plus all
its corners; variables whose partial derivative changes sign carry no
constraint. The derivation is checked in tests/test_benchmark.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass
class Problem:
    name: str
    expr: str                      # sympy expression in the variable names
    variables: list[str]
    lo: list[float]
    hi: list[float]


PROBLEMS = [
    Problem("I.6.20a", "exp(-theta**2/2)/sqrt(2*pi)", ["theta"], [1.0], [3.0]),
    Problem("I.12.2", "q1*q2/(4*pi*epsilon*r**2)", ["q1", "q2", "epsilon", "r"], [1, 1, 1, 1], [5, 5, 5, 5]),
    Problem("I.15.3x", "(x - u*t)/sqrt(1 - u**2/c**2)", ["x", "u", "c", "t"], [5, 1, 3, 1], [10, 2, 20, 2]),
    Problem("I.18.4", "(m1*r1 + m2*r2)/(m1 + m2)", ["m1", "m2", "r1", "r2"], [1, 1, 1, 1], [5, 5, 5, 5]),
    Problem("I.30.5", "asin(lambd/(n*d))", ["lambd", "d", "n"], [1, 3, 1], [2, 5, 5]),
    Problem("I.39.22", "n*T/V", ["n", "T", "V"], [1, 1, 1], [5, 5, 5]),
    Problem("I.48.20", "m*c**2/sqrt(1 - v**2/c**2)", ["m", "v", "c"], [1, 1, 3], [5, 2, 20]),
]


def by_name(name: str) -> Problem:
    for p in PROBLEMS:
        if p.name == name:
            return p
    raise KeyError(name)


def _sym(p: Problem):
    syms = sp.symbols(p.variables, real=True)
    f = sp.sympify(p.expr, locals=dict(zip(p.variables, syms)))
    return syms, f


def function(p: Problem):
    syms, f = _sym(p)
    fn = sp.lambdify(syms, f, "numpy")
    return lambda X: np.asarray(fn(*[X[:, i] for i in range(X.shape[1])]), float) * np.ones(len(X))


def derive_constraints(p: Problem, n: int = 20000, seed: int = 0) -> list[list[float]]:
    syms, f = _sym(p)
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(p.lo, float), np.asarray(p.hi, float)
    X = lo + rng.random((n, len(lo))) * (hi - lo)
    corners = np.array(np.meshgrid(*[[a, b] for a, b in zip(lo, hi)])).reshape(len(lo), -1).T
    X = np.vstack([X, corners])
    out = []
    for j, s in enumerate(syms):
        d = sp.lambdify(syms, sp.diff(f, s), "numpy")
        v = np.asarray(d(*[X[:, i] for i in range(X.shape[1])]), float) * np.ones(len(X))
        if not np.all(np.isfinite(v)):
            continue
        if np.all(v >= 0) and np.any(v > 0):
            out.append([j, 1.0])
        elif np.all(v <= 0) and np.any(v < 0):
            out.append([j, -1.0])
    return out


def make_data(p: Problem, n_train: int, n_test: int, noise: float, train_fraction: float, seed: int):
    """Training points in the lower part of every range (extrapolation upwards); test points on the full box."""
    rng = np.random.default_rng(seed)
    f = function(p)
    lo, hi = np.asarray(p.lo, float), np.asarray(p.hi, float)
    thi = lo + train_fraction * (hi - lo)
    Xtr = lo + rng.random((n_train, len(lo))) * (thi - lo)
    ytrue = f(Xtr)
    ytr = ytrue + rng.normal(0.0, noise * np.std(ytrue), n_train)
    Xte = lo + rng.random((n_test, len(lo))) * (hi - lo)
    yte = f(Xte)
    extrap = np.any(Xte > thi, axis=1)
    return Xtr, ytr, Xte, yte, extrap


def uniform_boxes(lo, hi, max_boxes: int):
    """Fixed partition with k segments per dimension, k^d <= max_boxes (the DCSR-IA setting)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    d = len(lo)
    k = max(1, int(np.floor(max_boxes ** (1.0 / d) + 1e-9)))
    edges = [np.linspace(a, b, k + 1) for a, b in zip(lo, hi)]
    idx = np.array(np.meshgrid(*[np.arange(k)] * d, indexing="ij")).reshape(d, -1).T
    LO = np.column_stack([edges[j][idx[:, j]] for j in range(d)])
    HI = np.column_stack([edges[j][idx[:, j] + 1] for j in range(d)])
    return LO, HI
