"""Step 08: evaluate every trained run from its saved files.

Per run (cached as evaluation.json and curves.csv in the run directory, redone
only when the run or the evaluation settings change):
  accuracy on validation and every test set, test-point slope violations,
  domain-grid violation rate and most negative slope, interval certification
  (GP), complexity (simplified equation for GP), and engine-size response curves.
The per-run files are then collected into tables in outputs/08_evaluation/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import stable_hash
from ..io import code_hash, read_csv, read_json, write_csv, write_json
from ..metrics import accuracy, simplify_gp, slope_summary
from ..models import load_model
from .common import block_frames, load_clean, load_domain, load_splits, sdir

CODE_DEPS = ['co2sr.metrics']
NAME = "08_evaluation"
INCREMENTAL = True
KEYS = ["dataset", "protocol", "block", "method", "seed"]


def config_part(cfg):
    return {"evaluation": cfg["evaluation"]}


def inputs(cfg):
    return []


def _reference_curves(full: pd.DataFrame, dcfg: dict, points: int) -> pd.DataFrame:
    """Engine-size sweeps at the most common training context for 4, 6 and 8 cylinders, fuel X."""
    rows = []
    base = full[full["fuel_type"] == "X"] if (full["fuel_type"] == "X").any() else full
    for c in (4, 6, 8):
        sub = base[base["cylinders"] == c]
        if sub.empty:
            sub = base
        mode = sub.assign(gears=sub["gears"].fillna(-1)).groupby(
            ["transmission_type", "gears", "gears_missing", "body_group", "fuel_type"]).size().idxmax()
        lo = max(dcfg["engine_size_min"], dcfg["litres_per_cylinder_min"] * c)
        hi = min(dcfg["engine_size_max"], dcfg["litres_per_cylinder_max"] * c)
        for e in np.linspace(lo, hi, points):
            rows.append({"curve_cylinders": c, "engine_size": e, "cylinders": c,
                         "transmission_type": mode[0], "gears": np.nan if mode[1] == -1 else mode[1],
                         "gears_missing": mode[2], "body_group": mode[3], "fuel_type": mode[4]})
    return pd.DataFrame(rows)


def evaluate_run(cfg, rd: Path, run: dict, cache: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ev = cfg["evaluation"]
    tol = ev["slope_tol"]
    ds, p, b = run["dataset"], run["protocol"], run["block"]
    model = load_model(rd)
    pred = read_csv(rd / "predictions.csv")
    base = {k: run[k] for k in KEYS}
    rows = []
    for s, g in pred.groupby("set"):
        rows.append({**base, "set": s, **accuracy(g["y"], g["yhat"]),
                     **{f"points_{k}": v for k, v in slope_summary(g["slope"], tol).items()}})
    bkey = (ds, p, b)
    clean_cache = cache.setdefault("clean", {})
    dom_cache = cache.setdefault("domain", {})
    if ds not in clean_cache:
        clean_cache[ds] = load_clean(cfg, ds)
    if bkey not in dom_cache:
        if len(dom_cache) > 4:
            dom_cache.clear()
        block = next(x for x in load_splits(cfg, ds, p)["blocks"] if x["block_id"] == b)
        dom = load_domain(cfg, ds, p, b, which=("grid", "boxes_eval", "boxes_verify"))
        dom["curves"] = _reference_curves(block_frames(clean_cache[ds], block)["full"], cfg["domain"],
                                          ev["response_curve_points"])
        dom_cache[bkey] = dom
    dom = dom_cache[bkey]
    grid_slope = model.slope(dom["grid"])
    summ = {**base, "status": run["status"], "fit_time_s": run.get("fit_time_s"),
            **{f"domain_{k}": v for k, v in slope_summary(grid_slope, tol).items()}}
    fin = np.isfinite(grid_slope)
    summ["domain_mean_slope"] = float(np.mean(grid_slope[fin])) if fin.any() else np.nan
    summ["domain_share_zero_slope"] = float(np.mean(np.abs(grid_slope[fin]) <= tol)) if fin.any() else np.nan
    # A model that ignores engine size satisfies the constraint trivially; record it.
    summ["uses_engine_size"] = int(fin.any() and np.mean(np.abs(grid_slope[fin]) > tol) > 0)
    for f, idx in dom["grid"].groupby("fuel_type").groups.items():
        gs = grid_slope[np.asarray(dom["grid"].index.get_indexer(idx))]
        gs = gs[np.isfinite(gs)]
        summ[f"domain_mean_slope_{f}"] = float(np.mean(gs)) if len(gs) else np.nan
    desc = run.get("description", {})
    summ["n_parameters"] = desc.get("n_parameters")
    summ["equation"] = desc.get("equation")
    if run["spec"]["type"] == "gp":
        cert = model.certify(dom["boxes_eval_lo"], dom["boxes_eval_hi"])
        summ.update({"certified": int(cert["certified"]), "share_boxes_failing": cert["share_boxes_failing"],
                     "min_slope_lower_bound": cert["min_slope_lower_bound"],
                     **model.verify_final(dom["boxes_verify_lo"], dom["boxes_verify_hi"],
                                          int(cfg["domain"].get("verify_eval_max_boxes", 200000)),
                                          int(cfg["domain"].get("verify_eval_max_depth", 16))),
                     "n_nodes": desc.get("n_nodes"), "depth": desc.get("depth"),
                     "n_constants": desc.get("n_constants"),
                     "variables_used": ";".join(desc.get("variables_used", [])),
                     "feasible_on_constraint_set": int(desc.get("params", {}).get("feasible_on_constraint_set", 0)),
                     "n_fitness_evaluations": desc.get("params", {}).get("n_fitness_evaluations"),
                     "final_verdict_in_search": desc.get("params", {}).get("final_verdict", ""),
                     **{k: v for k, v in desc.get("params", {}).items() if k.startswith("verify_")}})
        if ev.get("simplify_equations", True):
            mj = read_json(rd / "model.json")
            summ.update(simplify_gp(mj["tree"], mj["names"]))
    curves = dom["curves"].copy()
    curves["yhat"] = model.predict(curves)
    curves = curves[["curve_cylinders", "engine_size", "yhat"]]
    for k, v in base.items():
        curves[k] = v
    return summ, pd.DataFrame(rows), curves


def run(cfg, log, force=False, **_):
    root = sdir(cfg, "07")
    ev_hash = stable_hash({"evaluation": cfg["evaluation"], "domain": cfg["domain"],
                           "code": code_hash(CODE_DEPS + [__name__])})
    run_files = sorted(root.rglob("run.json"))
    summaries, metrics, curves, hist = [], [], [], []
    cache: dict = {}
    n_new = 0
    for rj in run_files:
        rd = rj.parent
        run = read_json(rj)
        if run.get("status") != "ok":
            continue
        ej, cj, mj = rd / "evaluation.json", rd / "curves.csv", rd / "metrics.csv"
        key = stable_hash([run["run_hash"], ev_hash])
        if not force and ej.exists() and cj.exists() and mj.exists() and read_json(ej).get("eval_hash") == key:
            summ = read_json(ej)
            met, cur = read_csv(mj), read_csv(cj)
        else:
            try:
                summ, met, cur = evaluate_run(cfg, rd, run, cache)
            except Exception as exc:
                log.warning("evaluation failed for %s: %s", rd, exc)
                continue
            summ["eval_hash"] = key
            write_json(summ, ej)
            write_csv(met, mj)
            write_csv(cur, cj)
            n_new += 1
        summaries.append(summ)
        metrics.append(met)
        curves.append(cur)
        if (rd / "history.csv").exists():
            h = read_csv(rd / "history.csv")
            for k in KEYS:
                h[k] = run[k]
            hist.append(h)
    log.info("%d run(s) evaluated, %d newly", len(summaries), n_new)
    d = sdir(cfg, "08")
    outs = []
    if summaries:
        outs.append(write_csv(pd.DataFrame(summaries), d / "run_summary.csv"))
        outs.append(write_csv(pd.concat(metrics, ignore_index=True), d / "run_metrics.csv"))
        outs.append(write_csv(pd.concat(curves, ignore_index=True), d / "response_curves.csv"))
    if hist:
        outs.append(write_csv(pd.concat(hist, ignore_index=True), d / "gp_convergence.csv"))
    return outs
