"""Step 10: manufacturer residual audit (secondary objective S1).

For the audited method, one run per block is used (the seed with the median
validation RMSE) so that seeds do not count as independent observations.
Make was not a model input; a significant result means systematic under- or
over-prediction relative to the recorded specifications.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import read_csv, write_csv, write_json
from ..stats import bootstrap_median_ci, kruskal_dunn
from .common import load_clean, run_dir, sdir

CODE_DEPS = ['co2sr.stats']
NAME = "10_audit"


def config_part(cfg):
    return {"audit": cfg["audit"]}


def inputs(cfg):
    return [sdir(cfg, "08") / "run_metrics.csv"]


def run(cfg, log, **_):
    ac = cfg["audit"]
    met = read_csv(sdir(cfg, "08") / "run_metrics.csv")
    sel = met[(met["dataset"] == ac["dataset"]) & (met["protocol"] == ac["protocol"])
              & (met["method"] == ac["method"]) & (met["set"] == "val")]
    if sel.empty:
        log.warning("no runs for the audit configuration %s", ac)
        return []
    chosen = []
    for b, g in sel.groupby("block"):
        g = g.sort_values("rmse")
        chosen.append(g.iloc[(len(g) - 1) // 2])
    df = load_clean(cfg, ac["dataset"]).set_index("row_id")
    res = []
    for r in chosen:
        p = read_csv(run_dir(cfg, r["dataset"], r["protocol"], r["block"], r["method"], int(r["seed"]))
                     / "predictions.csv")
        p = p[~p["set"].isin(["val", "new_config"])]
        p["residual"] = p["y"] - p["yhat"]
        p["make"] = df.loc[p["row_id"], "make"].to_numpy()
        p["block"] = r["block"]
        p["seed"] = int(r["seed"])
        res.append(p)
    res = pd.concat(res, ignore_index=True)
    counts = res["make"].value_counts()
    keep = counts[counts >= ac["min_rows_per_make"]].index
    rr = res[res["make"].isin(keep)]
    groups = {m: g["residual"].to_numpy() for m, g in rr.groupby("make")}
    out_dir = sdir(cfg, "10")
    outs = [write_csv(res, out_dir / "audit_residuals.csv")]
    if len(groups) >= 2:
        kw, dunn = kruskal_dunn(groups)
        per = []
        for i, (m, x) in enumerate(sorted(groups.items())):
            lo, hi = bootstrap_median_ci(x, ac["bootstrap"], cfg["project"]["random_seed"] + i)
            per.append({"make": m, "n": len(x), "median_residual": float(np.median(x)),
                        "ci_low": lo, "ci_high": hi, "mean_residual": float(np.mean(x))})
        outs.append(write_csv(pd.DataFrame(per), out_dir / "audit_by_make.csv"))
        outs.append(write_csv(dunn, out_dir / "audit_dunn_pairwise.csv"))
        outs.append(write_json({**kw, **ac, "makes_excluded": int((counts < ac["min_rows_per_make"]).sum())},
                               out_dir / "audit_kruskal.json"))
        log.info("Kruskal-Wallis H=%.2f p=%.3g over %d makes", kw["H"], kw["p_value"], len(groups))
    return outs
