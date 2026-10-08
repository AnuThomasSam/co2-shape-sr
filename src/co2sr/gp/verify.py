"""Budgeted branch-and-bound verification of shape constraints (DCSR-V).

A shape constraint is a pair (j, s): s * d f / d x_j >= 0 on the declared
domain D, where f(x) = a_k + b_k g(x) is the scaled model of group k and D is
a union of axis-aligned root boxes (for the CO2 data: one box per categorical
context, spanning the admissible engine-size range).

For a box B, interval arithmetic on the forward-mode derivative gives a lower
bound L(B) of s * b_k * dg/dx_j over B. Then

  L(B) >= -tol                 -> B is proved (sound: interval arithmetic
                                  encloses the true range, so no point of B
                                  violates the constraint);
  a probe point x in B with
  s * b_k * dg/dx_j(x) < -tol  -> a counterexample (exact, not an estimate);
  otherwise                    -> B is bisected along its widest dimension
                                  (relative to the root box), until a depth or
                                  box budget is exhausted.

The verdict for a model is three-valued:

  refuted    at least one counterexample was found;
  certified  every box was proved (a proof for the whole of D);
  undecided  no counterexample, but some boxes remained unproved when the
             budget ran out.

Plain interval arithmetic on a fixed partition (DCSR-IA) is the special case
max_depth = 0 without probing: it cannot distinguish "refuted" from
"undecided", so it rejects every model whose bound is loose. Bisection shrinks
the overestimation caused by the dependency problem, because the width of the
interval enclosure of a Lipschitz expression decreases with the box width.
Termination is guaranteed by the budgets; completeness is not claimed (a model
whose minimum slope is exactly zero on a set of positive measure may stay
undecided).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scaling import Scaling
from .tree import Node

DEFAULT_CONSTRAINTS = ((0, 1.0),)


@dataclass
class Verdict:
    status: str                       # certified | refuted | undecided
    n_boxes: int = 0                  # interval evaluations (boxes visited)
    n_bisections: int = 0
    max_depth: int = 0
    min_lower_bound: float = np.inf   # smallest proved-or-not lower bound seen at the root level
    worst_slope: float = 0.0          # most negative signed slope at a counterexample
    undecided_fraction: float = 0.0   # share of the root volume left unproved
    cx_X: np.ndarray | None = None    # counterexample points (rows of the design matrix)
    cx_g: np.ndarray | None = None
    cx_slope: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _bounds(tree, scaling, LO, HI, groups, constraints):
    """Smallest lower bound over constraints of s * b * dg/dx_j on each box (NaN -> -inf)."""
    _, b = scaling.coefficients(groups)
    out = np.full(len(LO), np.inf)
    for j, s in constraints:
        _, _, dlo, dhi = tree.evaluate_interval(LO, HI, int(j))
        c = s * b
        with np.errstate(all="ignore"):
            lb = np.where(c > 0, c * dlo, np.where(c < 0, c * dhi, 0.0))
        lb = np.where(np.isnan(lb), -np.inf, lb)
        out = np.minimum(out, lb)
    return out


def signed_slopes(tree, scaling, X, groups, constraints=DEFAULT_CONSTRAINTS):
    """Smallest over constraints of s * b * dg/dx_j at each point (NaN -> -inf)."""
    _, b = scaling.coefficients(groups)
    out = np.full(len(X), np.inf)
    for j, s in constraints:
        _, dv = tree.evaluate_dual(X, int(j))
        with np.errstate(all="ignore"):
            v = s * b * dv
        out = np.minimum(out, np.where(np.isnan(v), -np.inf, v))     # +inf satisfies the inequality
    return out


def verify(tree: Node, scaling: Scaling, LO: np.ndarray, HI: np.ndarray, groups: np.ndarray,
           tol: float = 1e-9, max_boxes: int = 4000, max_depth: int = 10,
           constraints=DEFAULT_CONSTRAINTS, probe: bool = True, max_cx: int = 5) -> Verdict:
    LO = np.asarray(LO, float)
    HI = np.asarray(HI, float)
    groups = np.asarray(groups, int)
    n_root = len(LO)
    root_w = HI - LO
    lo, hi, g = LO.copy(), HI.copy(), groups.copy()
    origin = np.arange(n_root)
    depth = np.zeros(n_root, int)
    vol = np.full(n_root, 1.0 / max(n_root, 1))      # share of the root volume (roots weighted equally)
    v = Verdict(status="certified")
    undecided_vol = 0.0
    cx_X, cx_g, cx_s = [], [], []
    first = True
    while len(lo):
        v.n_boxes += len(lo)
        lb = _bounds(tree, scaling, lo, hi, g, constraints)
        if first:
            v.min_lower_bound = float(lb.min()) if len(lb) else np.inf
            first = False
        open_ = ~(lb >= -tol)
        if not open_.any():
            break
        lo, hi, g, origin, depth, vol = (a[open_] for a in (lo, hi, g, origin, depth, vol))
        if probe:
            # probe the midpoint and both corners of every open box
            pts = np.concatenate([(lo + hi) / 2.0, lo, hi])
            gp = np.concatenate([g, g, g])
            sl = signed_slopes(tree, scaling, pts, gp, constraints)
            bad = sl < -tol
            if bad.any():
                order = np.argsort(sl)[: max_cx]
                order = order[bad[order]]
                cx_X.append(pts[order]); cx_g.append(gp[order]); cx_s.append(sl[order])
                break
        # bisect what remains, within the budgets
        can = depth < max_depth
        n_split = int(can.sum())
        if v.n_boxes + 2 * n_split > max_boxes:
            # budget exhausted: everything still open is undecided
            undecided_vol += float(vol.sum())
            break
        undecided_vol += float(vol[~can].sum())
        lo, hi, g, origin, depth, vol = (a[can] for a in (lo, hi, g, origin, depth, vol))
        if not len(lo):
            break
        with np.errstate(all="ignore"):
            rel = np.where(root_w[origin] > 0, (hi - lo) / np.where(root_w[origin] > 0, root_w[origin], 1.0), 0.0)
        dim = np.argmax(rel, axis=1)
        if np.all(rel[np.arange(len(lo)), dim] == 0):      # degenerate boxes cannot be split
            undecided_vol += float(vol.sum())
            break
        mid = (lo[np.arange(len(lo)), dim] + hi[np.arange(len(lo)), dim]) / 2.0
        lo2, hi1 = lo.copy(), hi.copy()
        hi1[np.arange(len(lo)), dim] = mid
        lo2[np.arange(len(lo)), dim] = mid
        lo = np.concatenate([lo, lo2]); hi = np.concatenate([hi1, hi])
        g = np.concatenate([g, g]); origin = np.concatenate([origin, origin])
        depth = np.concatenate([depth + 1, depth + 1]); vol = np.concatenate([vol / 2, vol / 2])
        v.n_bisections += n_split
        v.max_depth = int(depth.max())
    if cx_X:
        v.status = "refuted"
        v.cx_X, v.cx_g, v.cx_slope = np.concatenate(cx_X), np.concatenate(cx_g), np.concatenate(cx_s)
        v.worst_slope = float(v.cx_slope.min())
    elif undecided_vol > 0:
        v.status = "undecided"
    v.undecided_fraction = float(undecided_vol)
    return v
