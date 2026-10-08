"""Empirical support for the monotone prior (plan, Section 4.3).

Matched pairs: records identical on the match keys but with different engine
size. For each group, CO2 is averaged per engine size; every pair of engine
sizes in the group is compared. A pair "violates" the prior when the larger
engine has strictly lower CO2.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def matched_pairs(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    keys = [k for k in keys if k in df.columns and df[k].notna().any()]
    d = df.copy()
    d["gears"] = d["gears"].fillna(-1)
    rows = []
    for gkey, g in d.groupby(keys, dropna=False):
        means = g.groupby("engine_size")["co2"].mean()
        if len(means) < 2:
            continue
        gdict = dict(zip(keys, gkey if isinstance(gkey, tuple) else (gkey,)))
        for (e1, c1), (e2, c2) in itertools.combinations(sorted(means.items()), 2):
            rows.append({**gdict, "engine_small": e1, "engine_large": e2, "co2_small": c1, "co2_large": c2,
                         "violation": int(c2 < c1), "tie": int(c2 == c1)})
    return pd.DataFrame(rows)


def _row(name, g):
    slope = (g["co2_large"] - g["co2_small"]) / (g["engine_large"] - g["engine_small"])
    return {"group": name, "pairs": len(g), "violation_share": g["violation"].mean(),
            "tie_share": g["tie"].mean(),
            # matched-pair estimate of the engine-size effect, compared with model slopes in step 09
            "median_slope_g_per_km_per_litre": float(np.median(slope)),
            "q1_slope": float(np.quantile(slope, 0.25)), "q3_slope": float(np.quantile(slope, 0.75))}


def support_summary(pairs: pd.DataFrame, by: str | None) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame([{"group": "all", "pairs": 0, "violation_share": np.nan, "tie_share": np.nan}])
    out = [_row("all", pairs)]
    if by and by in pairs:
        for k, g in pairs.groupby(by):
            out.append(_row(f"{by}={k}", g))
    if "fuel_type" in pairs:
        for k, g in pairs.groupby("fuel_type"):
            out.append(_row(f"fuel_type={k}", g))
    return pd.DataFrame(out)
