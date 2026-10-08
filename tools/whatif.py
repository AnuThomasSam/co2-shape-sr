#!/usr/bin/env python
"""What-if tool (secondary objective S2).

Loads a saved model from a run directory, sweeps one input while holding the
others fixed, and writes a table (and optionally a figure) of predicted CO2.
Points outside the declared domain are refused, not extrapolated silently.

Example
  python tools/whatif.py --run outputs/07_models/nrcan/P1/T2023/dcsr_s/seed00 \\
      --vary engine_size --start 1.0 --stop 5.0 --step 0.25 \\
      --set cylinders=4 transmission_type=AS gears=8 body_group=car fuel_type=X \\
      --out results/whatif_engine_size_4cyl.csv --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from co2sr.config import load_config  # noqa: E402
from co2sr.models import load_model  # noqa: E402

DEFAULTS = {"engine_size": 2.0, "cylinders": 4, "transmission_type": "AS", "gears": 8.0, "gears_missing": 0,
            "body_group": "car", "fuel_type": "X"}
NUMERIC = {"engine_size", "cylinders", "gears", "gears_missing"}


def in_domain(df: pd.DataFrame, dc: dict) -> np.ndarray:
    r = df["engine_size"] / df["cylinders"]
    return (df["engine_size"].between(dc["engine_size_min"], dc["engine_size_max"])
            & df["cylinders"].isin(dc["cylinders"])
            & r.between(dc["litres_per_cylinder_min"], dc["litres_per_cylinder_max"])).to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--run", required=True, help="run directory containing model.joblib")
    ap.add_argument("--vary", required=True, choices=["engine_size", "cylinders", "gears"])
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--stop", type=float, required=True)
    ap.add_argument("--step", type=float, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="fixed values, e.g. cylinders=4 fuel_type=X")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    model = load_model(ROOT / a.run if not Path(a.run).is_absolute() else a.run)
    base = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in base:
            ap.error(f"unknown variable '{k}'; choose from {sorted(base)}")
        base[k] = float(v) if k in NUMERIC else v
    if base["transmission_type"] == "AV" and "gears" not in [kv.split("=")[0] for kv in a.set]:
        base["gears"], base["gears_missing"] = np.nan, 1
    values = np.arange(a.start, a.stop + 1e-9, a.step)
    df = pd.DataFrame([{**base, a.vary: v} for v in values])
    ok = in_domain(df, cfg["domain"])
    if not ok.all():
        bad = df.loc[~ok, a.vary].tolist()
        print(f"Refused: {len(bad)} point(s) outside the declared domain ({a.vary} = {bad}). "
              f"Narrow the sweep or change the fixed values.", file=sys.stderr)
        df = df[ok].reset_index(drop=True)
        if df.empty:
            sys.exit(1)
    df["predicted_co2_g_per_km"] = model.predict(df)
    df["slope_g_per_km_per_litre"] = model.slope(df)
    print(df.to_string(index=False))
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        if a.plot:
            from co2sr.plotting import apply_style, save
            import matplotlib.pyplot as plt
            apply_style(cfg)
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            ax.plot(df[a.vary], df["predicted_co2_g_per_km"], color="#2a78d6", marker="o", markersize=3)
            ax.set_xlabel({"engine_size": "Engine size (L)", "cylinders": "Cylinders", "gears": "Gears"}[a.vary])
            ax.set_ylabel("Predicted CO$_2$ (g/km)")
            save(fig, out.parent, out.stem, cfg)


if __name__ == "__main__":
    main()
