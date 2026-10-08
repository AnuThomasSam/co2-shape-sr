"""Declared input domain D, its sample Q_D, the evaluation grid and interval boxes.

D is defined a priori in the configuration (engine-size range, admissible
cylinder counts, litres per cylinder). The categorical context of a point
(transmission, gears, body group, fuel) is taken from combinations observed in
the TRAINING rows of a block only; test rows are never used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONTEXT_COLS = ["transmission_type", "gears", "gears_missing", "body_group", "fuel_type"]


def build_contexts(train: pd.DataFrame, dcfg: dict) -> pd.DataFrame:
    cats = train[CONTEXT_COLS].copy()
    cats["gears"] = cats["gears"].fillna(-1.0)          # -1 marks "no fixed gear count"
    cats = cats.drop_duplicates().reset_index(drop=True)
    cyl = pd.DataFrame({"cylinders": dcfg["cylinders"]})
    ctx = cats.merge(cyl, how="cross")
    ctx["engine_lo"] = np.maximum(dcfg["engine_size_min"], dcfg["litres_per_cylinder_min"] * ctx["cylinders"])
    ctx["engine_hi"] = np.minimum(dcfg["engine_size_max"], dcfg["litres_per_cylinder_max"] * ctx["cylinders"])
    ctx = ctx[ctx["engine_hi"] > ctx["engine_lo"]].reset_index(drop=True)
    ctx["gears"] = ctx["gears"].replace(-1.0, np.nan)
    ctx.insert(0, "context_id", np.arange(len(ctx)))
    return ctx


def _subset(ctx: pd.DataFrame, n_max: int, seed: int) -> pd.DataFrame:
    if len(ctx) <= n_max:
        return ctx
    return ctx.sample(n=n_max, random_state=seed).sort_values("context_id").reset_index(drop=True)


def domain_sample(ctx: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ctx), size)
    s = ctx.iloc[idx].reset_index(drop=True).copy()
    u = rng.random(size)
    s["engine_size"] = s["engine_lo"] + u * (s["engine_hi"] - s["engine_lo"])
    return s.drop(columns=["engine_lo", "engine_hi"])


def domain_grid(ctx: pd.DataFrame, points: int, max_contexts: int, seed: int) -> pd.DataFrame:
    c = _subset(ctx, max_contexts, seed)
    rep = c.loc[c.index.repeat(points)].reset_index(drop=True)
    frac = np.tile(np.linspace(0.0, 1.0, points), len(c))
    rep["engine_size"] = rep["engine_lo"] + frac * (rep["engine_hi"] - rep["engine_lo"])
    return rep.drop(columns=["engine_lo", "engine_hi"])


def domain_boxes(ctx: pd.DataFrame, segments: int, max_contexts: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Boxes over engine size for every context: returns (lower corner, upper corner) frames."""
    c = _subset(ctx, max_contexts, seed)
    rep = c.loc[c.index.repeat(segments)].reset_index(drop=True)
    k = np.tile(np.arange(segments), len(c))
    width = (rep["engine_hi"] - rep["engine_lo"]) / segments
    lo = rep.drop(columns=["engine_lo", "engine_hi"]).copy()
    hi = lo.copy()
    lo["engine_size"] = rep["engine_lo"] + k * width
    hi["engine_size"] = rep["engine_lo"] + (k + 1) * width
    return lo, hi
