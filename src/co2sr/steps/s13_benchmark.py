"""Step 13: synthetic shape-constrained benchmark (independent of the CO2 data).

For every problem in co2sr.benchmark.problems, method and seed:
  1. generate training data in the lower part of the domain (noise added) and
     test data on the whole domain;
  2. fit the GP engine with the method's constraint handling; the declared
     domain D is the full box; DCSR-IA uses a fixed partition of D, DCSR-V
     starts from the single root box D and refines it adaptively;
  3. evaluate test NMSE (whole domain and extrapolation part), the share of
     test points that violate a constraint, the branch-and-bound verdict of the
     final model (large budget), the plain interval verdict, size and time.

Each run is written to outputs/13_benchmark/runs/<problem>/<method>/seedNN.json
and is reused while its hash (spec, problem, data settings, code) is unchanged.
All runs are collected in outputs/13_benchmark/benchmark_runs.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..benchmark.problems import by_name, derive_constraints, make_data, uniform_boxes
from ..config import method_spec, stable_hash
from ..gp.engine import GPRegressor
from ..gp.verify import signed_slopes
from ..io import code_hash, read_json, write_csv, write_json
from .common import sdir

CODE_DEPS = ["co2sr.benchmark.problems", "co2sr.gp.engine", "co2sr.gp.verify", "co2sr.gp.constraints",
             "co2sr.gp.tree", "co2sr.gp.primitives", "co2sr.gp.scaling"]
NAME = "13_benchmark"
INCREMENTAL = True


def config_part(cfg):
    return {"benchmark": cfg["benchmark"], "gp_defaults": cfg["gp_defaults"],
            "methods": {m: cfg["methods"][m] for m in cfg["benchmark"]["methods"]}}


def inputs(cfg):
    return []


def _spec(cfg, method, constraints):
    bc = cfg["benchmark"]
    spec = method_spec(cfg, method)
    spec.update(bc.get("gp_overrides", {}))
    spec.update({"require_engine_size": False, "constraints": constraints, "fuel_mode": "none"})
    return spec


def run_one(cfg, problem, method, seed, force=False):
    bc = cfg["benchmark"]
    p = by_name(problem)
    cons = derive_constraints(p)
    spec = _spec(cfg, method, cons)
    path = sdir(cfg, "13") / "runs" / problem / method / f"seed{seed:02d}.json"
    h = stable_hash({"spec": spec, "problem": problem, "data": {k: bc[k] for k in
                     ("n_train", "n_test", "noise", "train_fraction", "ia_boxes", "eval_max_boxes",
                      "eval_max_depth")}, "code": code_hash(CODE_DEPS)}, 16)
    if path.exists() and not force:
        old = read_json(path)
        if old.get("run_hash") == h:
            return old
    data_seed = 1000 + seed
    Xtr, ytr, Xte, yte, extrap = make_data(p, bc["n_train"], bc["n_test"], bc["noise"], bc["train_fraction"], data_seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ytr))
    n_val = int(round(bc["validation_fraction"] * len(ytr)))
    vi, fi = perm[:n_val], perm[n_val:]
    lo, hi = np.asarray(p.lo, float)[None, :], np.asarray(p.hi, float)[None, :]
    kw = {}
    if spec["constraint"] in ("sample", "feasibility"):
        q = np.random.default_rng(bc["domain_sample_seed"])
        kw.update(X_q=lo + q.random((bc["domain_sample_size"], lo.shape[1])) * (hi - lo),
                  g_q=np.zeros(bc["domain_sample_size"], int))
    elif spec["constraint"] == "interval":
        LO, HI = uniform_boxes(p.lo, p.hi, bc["ia_boxes"])
        kw.update(LO=LO, HI=HI, g_box=np.zeros(len(LO), int))
    elif spec["constraint"] == "verified":
        kw.update(LO=lo, HI=hi, g_box=np.zeros(1, int))
    g0 = lambda n: np.zeros(n, int)  # noqa: E731
    t0 = time.perf_counter()
    gp = GPRegressor(spec, seed).fit(Xtr[fi], ytr[fi], g0(len(fi)), 1, X_val=Xtr[vi], y_val=ytr[vi], g_val=g0(len(vi)),
                                     X_full=Xtr, y_full=ytr, g_full=g0(len(ytr)), **kw)
    fit_time = time.perf_counter() - t0
    with np.errstate(all="ignore"):
        pred = gp.predict(Xte, g0(len(Xte)))
    var = float(np.var(yte))
    ok = np.isfinite(pred)
    nmse = lambda m: float(np.mean((yte[m] - pred[m]) ** 2) / var) if m.any() and ok[m].all() else np.inf  # noqa: E731
    sl = signed_slopes(gp.tree, gp.scaling, Xte, g0(len(Xte)), gp.constraints)
    LOi, HIi = uniform_boxes(p.lo, p.hi, bc["ia_boxes"])
    ia = gp.certify(LOi, HIi, g0(len(LOi)))
    t1 = time.perf_counter()
    bb = gp.verify_final(lo, hi, g0(1), bc["eval_max_boxes"], bc["eval_max_depth"])
    rec = {"problem": problem, "method": method, "seed": seed, "run_hash": h, "n_constraints": len(cons),
           "n_variables": len(p.variables), "fit_time_s": fit_time,
           "test_nmse": nmse(np.ones(len(yte), bool)), "extrap_nmse": nmse(extrap), "interp_nmse": nmse(~extrap),
           "test_violation_share": float(np.mean(~(sl >= -1e-9))),
           "ia_certified": int(ia["certified"]), "eval_verify_time_s": time.perf_counter() - t1, **bb,
           "n_nodes": gp.tree.size(), "equation": gp.tree.to_string(p.variables),
           "n_fitness_evaluations": gp.info.get("n_fitness_evaluations"),
           "final_verdict_in_search": gp.info.get("final_verdict", ""),
           **{k: v for k, v in gp.info.items() if k.startswith("verify_")}}
    write_json(rec, path)
    return rec


def run(cfg, log, force=False, n_jobs=None, seeds=None, methods=None, **_):
    bc = cfg["benchmark"]
    if not bc.get("enabled", True):
        log.info("benchmark disabled in the configuration")
        return []
    meths = methods or bc["methods"]
    sds = seeds if seeds is not None else list(range(bc["seeds"]))
    tasks = [(pb, m, s) for pb in bc["problems"] for m in meths for s in sds]
    n_jobs = n_jobs or cfg["project"].get("n_jobs", 1)
    log.info("%d benchmark run(s), n_jobs=%d", len(tasks), n_jobs)
    if n_jobs == 1:
        recs = [run_one(cfg, *t, force=force) for t in tasks]
    else:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=n_jobs, verbose=2)(delayed(run_one)(cfg, *t, force) for t in tasks)
    # collect every run on disk (also those from earlier calls with other seeds or methods)
    rows = [read_json(f) for f in sorted((sdir(cfg, "13") / "runs").rglob("seed*.json"))]
    df = pd.DataFrame(rows)
    df = df[df["problem"].isin(bc["problems"]) & df["method"].isin(bc["methods"])]
    cons = pd.DataFrame([{"problem": pb, "constraints": str(derive_constraints(by_name(pb))),
                          "variables": ";".join(by_name(pb).variables), "expression": by_name(pb).expr}
                         for pb in bc["problems"]])
    return [write_csv(df, sdir(cfg, "13") / "benchmark_runs.csv"),
            write_csv(cons, sdir(cfg, "13") / "benchmark_problems.csv")]
