"""Step 07: train every (experiment, protocol, block, method, seed) and save model + predictions.

Incremental: a run is skipped when its run.json exists with status "ok" (or
"skipped") and the same run hash. The hash covers the method specification,
the block's row ids and, for constrained GP, the domain configuration. Change
one method in the config and only that method's runs are redone.

Outputs per run (outputs/07_models/<dataset>/<protocol>/<block>/<method>/seedNN/):
  model.joblib     the fitted model (all methods)
  model.json       the GP tree and scaling coefficients (GP methods)
  predictions.csv  row_id, set (val or test-set name), y, yhat, slope
  history.csv      per-generation statistics (GP methods)
  run.json         status, specification, timings, description of the model
"""
from __future__ import annotations

import time
import traceback

import pandas as pd

from ..config import method_spec, stable_hash
from ..io import code_hash, read_json, write_csv, write_json
from ..models import build_model, pysr_available
from .common import block_frames, clean_file, load_clean, load_domain, load_splits, run_dir

CODE_DEPS = ['co2sr.models.base', 'co2sr.models.linear', 'co2sr.models.gam', 'co2sr.models.xgb', 'co2sr.models.pysr_model', 'co2sr.models.gp_model', 'co2sr.gp.engine', 'co2sr.gp.tree', 'co2sr.gp.primitives', 'co2sr.gp.scaling', 'co2sr.gp.constraints', 'co2sr.features']
NAME = "07_models"
INCREMENTAL = True


def config_part(cfg):
    return {"experiments": cfg["experiments"]}


def inputs(cfg):
    return []


_CACHE: dict = {}


def _clean_cached(cfg, ds):
    p = clean_file(cfg, ds)
    key = (str(p), p.stat().st_mtime)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = load_clean(cfg, ds)
    return _CACHE[key]


def _seeds(n_or_list):
    return list(range(n_or_list)) if isinstance(n_or_list, int) else list(n_or_list)


def plan_tasks(cfg, experiments=None, methods=None, protocols=None, blocks=None, seeds=None):
    tasks = []
    for ename, e in cfg["experiments"].items():
        if experiments and ename not in experiments:
            continue
        ds = e["dataset"]
        if not cfg["datasets"][ds].get("enabled", True):
            continue
        for p in e["protocols"]:
            if protocols and p not in protocols:
                continue
            try:
                sp = load_splits(cfg, ds, p)
            except FileNotFoundError:
                continue
            for b in sp["blocks"]:
                bid = b["block_id"]
                if e.get("blocks") and bid not in e["blocks"]:
                    continue
                if blocks and bid not in blocks:
                    continue
                for m in e["methods"]:
                    if methods and m not in methods:
                        continue
                    spec = method_spec(cfg, m)
                    ss = _seeds(e["seeds"]) if spec["stochastic"] else [0]
                    if seeds is not None:
                        ss = [s for s in ss if s in seeds] if spec["stochastic"] else [0]
                    for s in ss:
                        tasks.append((ds, p, bid, m, s))
    return sorted(set(tasks))


def run_hash(cfg, ds, p, block, spec):
    h = {"spec": spec, "dataset": ds, "protocol": p,
         "rows": stable_hash([block["fit"], block["val"], block["tests"]], 16),
         "code": code_hash(CODE_DEPS)}
    if spec["type"] == "gp" and spec.get("constraint", "none") != "none":
        h["domain"] = cfg["domain"]
    if spec["type"] in ("xgb", "gam", "ols", "loglinear", "pysr"):
        h["fd"] = cfg["evaluation"]["finite_difference_h"]
    return stable_hash(h, 16)


def train_one(cfg, task, force=False):
    ds, p, bid, m, seed = task
    rd = run_dir(cfg, ds, p, bid, m, seed)
    rd.mkdir(parents=True, exist_ok=True)
    spec = method_spec(cfg, m)
    block = next(b for b in load_splits(cfg, ds, p)["blocks"] if b["block_id"] == bid)
    h = run_hash(cfg, ds, p, block, spec)
    rj = rd / "run.json"
    if rj.exists() and not force:
        old = read_json(rj)
        if old.get("run_hash") == h and old.get("status") in ("ok", "skipped"):
            return "cached"
    rec = {"dataset": ds, "protocol": p, "block": bid, "method": m, "seed": seed, "run_hash": h,
           "spec": spec, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if spec["type"] == "pysr":
        ok, why = pysr_available()
        if not ok:
            rec.update({"status": "skipped", "reason": f"PySR not available: {why}"})
            write_json(rec, rj)
            return "skipped"
    try:
        df = _clean_cached(cfg, ds)
        fr = block_frames(df, block)
        domain = None
        if spec["type"] == "gp" and spec["constraint"] in ("sample", "feasibility", "interval"):
            domain = load_domain(cfg, ds, p, bid, which=("sample", "boxes_train"))
        elif spec["type"] == "gp" and spec["constraint"] == "verified":
            domain = load_domain(cfg, ds, p, bid, which=("boxes_verify",))
            n_root = len(domain["boxes_verify_lo"])
            budget = int(spec.get("verify_max_boxes", 4000))
            if budget <= n_root:
                # With a budget at or below the number of root boxes, the first interval pass uses it
                # up and no box is ever bisected: the run silently becomes the no-bisection ablation.
                raise ValueError(f"verify_max_boxes = {budget} is not larger than the {n_root} root boxes "
                                 f"of {ds}/{p}/{bid}; no bisection would be possible")
        model = build_model(spec, cfg, seed)
        t0 = time.perf_counter()
        model.fit(fr["fit"], fr["val"], fr["full"], domain)
        fit_time = time.perf_counter() - t0
        preds = []
        t1 = time.perf_counter()
        for name, part in [("val", fr["val"])] + list(fr["tests"].items()):
            if len(part) == 0:
                continue
            preds.append(pd.DataFrame({"row_id": part["row_id"], "set": name, "y": part["co2"],
                                       "yhat": model.predict(part), "slope": model.slope(part)}))
        predict_time = time.perf_counter() - t1
        model.save(rd)
        write_csv(pd.concat(preds, ignore_index=True), rd / "predictions.csv")
        if spec["type"] == "gp":
            write_csv(model.history(), rd / "history.csv")
        rec.update({"status": "ok", "fit_time_s": fit_time, "predict_time_s": predict_time,
                    "n_fit": len(fr["fit"]), "n_val": len(fr["val"]), "description": model.describe()})
        result = "ok"
    except Exception as exc:
        rec.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()})
        result = "failed"
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(rec, rj)
    return result


def run(cfg, log, force=False, experiments=None, methods=None, protocols=None, blocks=None,
        seeds=None, n_jobs=None, **_):
    tasks = plan_tasks(cfg, experiments, methods, protocols, blocks, seeds)
    n_jobs = n_jobs or cfg["project"].get("n_jobs", 1)
    log.info("%d run(s) planned, n_jobs=%d", len(tasks), n_jobs)
    if n_jobs == 1:
        results = []
        for i, t in enumerate(tasks):
            results.append(train_one(cfg, t, force))
            if results[-1] != "cached":
                log.info("[%d/%d] %s -> %s", i + 1, len(tasks), "/".join(map(str, t)), results[-1])
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(train_one)(cfg, t, force) for t in tasks)
    counts = pd.Series(results).value_counts().to_dict() if results else {}
    log.info("run outcomes: %s", counts)
    if counts.get("failed"):
        log.warning("Some runs failed; see the 'error' field in their run.json files.")
    return []
