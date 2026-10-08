"""Primitive functions with value, forward-mode derivative and interval rules.

Every primitive is defined three times, consistently:
  dual(a, da, b, db)                      -> (value, derivative)
  interval(alo, ahi, dalo, dahi, ...)     -> (lo, hi, dlo, dhi)
where derivatives are with respect to one input variable (engine size).

No primitive is "protected" by clipping. The analytic quotient is smooth
everywhere. The extended operators (exp, log, sqrt) are strict: outside their
domain they return non-finite values, and the candidate is then invalid. This
keeps the forward value, the derivative and the interval bounds consistent,
which protected operators with clipping do not.
"""
from __future__ import annotations

import numpy as np

ARITY = {"add": 2, "sub": 2, "mul": 2, "aq": 2, "exp": 1, "log": 1, "sqrt": 1}
FUNCTION_SETS = {
    "basic": ["add", "sub", "mul", "aq"],
    "extended": ["add", "sub", "mul", "aq", "exp", "log", "sqrt"],
}
SYMBOL = {"add": "+", "sub": "-", "mul": "*"}

_EXP_MAX = 700.0   # exp overflows float64 above about 709


# ----------------------------------------------------------------------------
# Point evaluation with forward-mode derivative
# ----------------------------------------------------------------------------

def dual(op, a, da, b=None, db=None):
    with np.errstate(all="ignore"):
        if op == "add":
            return a + b, da + db
        if op == "sub":
            return a - b, da - db
        if op == "mul":
            return a * b, da * b + a * db
        if op == "aq":
            s = 1.0 + b * b
            r = np.sqrt(s)
            v = a / r
            d = da / r - a * b * db / (s * r)
            return v, d
        if op == "exp":
            v = np.where(a > _EXP_MAX, np.inf, np.exp(np.minimum(a, _EXP_MAX)))
            return v, v * da
        if op == "log":
            v = np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), np.nan)
            return v, da / a
        if op == "sqrt":
            v = np.where(a >= 0, np.sqrt(np.abs(a)), np.nan)
            return v, da / (2.0 * v)
    raise ValueError(f"Unknown primitive {op}")


def value_only(op, a, b=None):
    with np.errstate(all="ignore"):
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "aq":
            return a / np.sqrt(1.0 + b * b)
        if op == "exp":
            return np.where(a > _EXP_MAX, np.inf, np.exp(np.minimum(a, _EXP_MAX)))
        if op == "log":
            return np.where(a > 0, np.log(np.where(a > 0, a, 1.0)), np.nan)
        if op == "sqrt":
            return np.where(a >= 0, np.sqrt(np.abs(a)), np.nan)
    raise ValueError(f"Unknown primitive {op}")


# ----------------------------------------------------------------------------
# Interval arithmetic (vectorised over boxes)
# ----------------------------------------------------------------------------

def _fix(lo, hi):
    """Replace NaN bounds (e.g. 0 * inf) by the widest interval."""
    lo = np.where(np.isnan(lo), -np.inf, lo)
    hi = np.where(np.isnan(hi), np.inf, hi)
    return lo, hi


def imul(alo, ahi, blo, bhi):
    with np.errstate(all="ignore"):
        p = np.stack(np.broadcast_arrays(alo * blo, alo * bhi, ahi * blo, ahi * bhi))
    lo = np.min(np.where(np.isnan(p), -np.inf, p), axis=0)
    hi = np.max(np.where(np.isnan(p), np.inf, p), axis=0)
    return lo, hi


def isq(lo, hi):
    """Interval of x^2."""
    a, b = lo * lo, hi * hi
    sq_lo = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(a, b))
    return sq_lo, np.maximum(a, b)


def irecip_pos(lo, hi):
    """1/x for a strictly positive interval."""
    with np.errstate(all="ignore"):
        return 1.0 / hi, 1.0 / lo


def interval(op, alo, ahi, dalo, dahi, blo=None, bhi=None, dblo=None, dbhi=None):
    with np.errstate(all="ignore"):
        if op == "add":
            return _fix(alo + blo, ahi + bhi) + _fix(dalo + dblo, dahi + dbhi)
        if op == "sub":
            return _fix(alo - bhi, ahi - blo) + _fix(dalo - dbhi, dahi - dblo)
        if op == "mul":
            v = imul(alo, ahi, blo, bhi)
            t1 = imul(dalo, dahi, blo, bhi)
            t2 = imul(alo, ahi, dblo, dbhi)
            return v + _fix(t1[0] + t2[0], t1[1] + t2[1])
        if op == "aq":
            s_lo, s_hi = isq(blo, bhi)
            s_lo, s_hi = 1.0 + s_lo, 1.0 + s_hi                  # s = 1 + b^2 >= 1
            r_lo, r_hi = np.sqrt(s_lo), np.sqrt(s_hi)             # r = sqrt(s) >= 1
            inv_r = irecip_pos(r_lo, r_hi)
            v = imul(alo, ahi, *inv_r)
            # derivative: da / r - a * b * db / (s * r)
            t1 = imul(dalo, dahi, *inv_r)
            ab = imul(alo, ahi, blo, bhi)
            abdb = imul(*ab, dblo, dbhi)
            sr = (s_lo * r_lo, s_hi * r_hi)                       # both factors positive
            t2 = imul(*abdb, *irecip_pos(*sr))
            return v + _fix(t1[0] - t2[1], t1[1] - t2[0])
        if op == "exp":
            vlo = np.where(alo > _EXP_MAX, np.inf, np.exp(np.minimum(alo, _EXP_MAX)))
            vhi = np.where(ahi > _EXP_MAX, np.inf, np.exp(np.minimum(ahi, _EXP_MAX)))
            return (vlo, vhi) + imul(vlo, vhi, dalo, dahi)
        if op == "log":
            ok = alo > 0
            vlo = np.where(ok, np.log(np.where(ok, alo, 1.0)), -np.inf)
            vhi = np.where(ahi > 0, np.log(np.where(ahi > 0, ahi, 1.0)), np.nan)
            inv = (np.where(ok, 1.0 / ahi, -np.inf), np.where(ok, 1.0 / np.where(ok, alo, 1.0), np.inf))
            dlo, dhi = imul(dalo, dahi, *inv)
            return vlo, np.where(ok, vhi, np.inf), np.where(ok, dlo, -np.inf), np.where(ok, dhi, np.inf)
        if op == "sqrt":
            ok = alo > 0
            vlo = np.sqrt(np.maximum(alo, 0.0))
            vhi = np.sqrt(np.maximum(ahi, 0.0))
            inv = (np.where(ok, 0.5 / vhi, -np.inf), np.where(ok, 0.5 / np.where(ok, vlo, 1.0), np.inf))
            dlo, dhi = imul(dalo, dahi, *inv)
            valid = alo >= 0
            return (np.where(valid, vlo, -np.inf), np.where(valid, vhi, np.inf),
                    np.where(ok, dlo, -np.inf), np.where(ok, dhi, np.inf))
    raise ValueError(f"Unknown primitive {op}")
