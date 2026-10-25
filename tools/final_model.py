"""The final model of the thesis: DCSR-V trained on the most recent window, tested on model year 2024.

Which run is used: in the P1 block T2024 (fit on 2021-2022, choose on 2023, test on 2024), the seed
whose validation RMSE is the median of the 30 seeds. Validation decides, so test data play no part
in the choice; the median avoids showing the luckiest run.

What the script prints and writes (results/tables/T6_final_model.tex, outputs/extra/final_model.json):
  1. the evolved shape g(x), its size, and its verification verdict;
  2. the fuel-specific coefficients: prediction = a_k + b_k * g(x) for fuel type k;
  3. the prediction written out for each transmission type, simplified with SymPy. Transmission
     types enter g(x) as 0/1 indicators, so fixing the type turns the equation into a short
     formula in engine size and cylinders. A negative b_k together with a g(x) that decreases in
     engine size gives a prediction that increases in engine size: the negative constants in g(x)
     are not an error.

The model is loaded from outputs/07_models if that folder exists; otherwise it is retrained with
the same seed (about one to two minutes), which gives the same equation.

Run from the repository root, after steps 07 to 09:
    python tools/final_model.py
    python tools/final_model.py --method gp_standard       # same for another GP method
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from co2sr.config import load_config, method_spec  # noqa: E402
from co2sr.models import build_model, load_model  # noqa: E402
from co2sr.steps.common import block_frames, load_clean, load_domain, load_splits, run_dir  # noqa: E402

FUEL = {"X": "regular petrol", "Z": "premium petrol", "D": "diesel", "E": "E85"}
TRANS = {"A": "automatic", "AM": "automated manual", "AS": "automatic with select shift",
         "AV": "continuously variable", "M": "manual"}


def aq(x, y):
    return x / sp.sqrt(1 + y ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--method", default="dcsr_v")
    ap.add_argument("--block", default="T2024")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    df = load_clean(cfg, "nrcan")
    block = next(b for b in load_splits(cfg, "nrcan", "P1")["blocks"] if b["block_id"] == args.block)

    metrics = pd.read_csv(ROOT / "outputs/08_evaluation/run_metrics.csv")
    m = metrics[(metrics.dataset == "nrcan") & (metrics.protocol == "P1") & (metrics.block == args.block)
                & (metrics.method == args.method)]
    val = m[m.set == "val"].set_index("seed")["rmse"].sort_values()
    seed = int(val.index[(len(val) - 1) // 2])
    test = float(m[(m.set == "all") & (m.seed == seed)]["rmse"].iloc[0])

    rd = run_dir(cfg, "nrcan", "P1", args.block, args.method, seed)
    if (rd / "model.joblib").exists():
        model = load_model(rd)
        source = f"loaded from {rd.relative_to(ROOT)}"
    else:
        spec = method_spec(cfg, args.method)
        fr = block_frames(df, block)
        which = ("boxes_verify",) if spec["constraint"] == "verified" else ("sample", "boxes_train")
        dom = load_domain(cfg, "nrcan", "P1", args.block, which=which) if spec["constraint"] != "none" else None
        model = build_model(spec, cfg, seed)
        model.fit(fr["fit"], fr["val"], fr["full"], dom)
        source = "retrained with the same seed"
    d = model.describe()

    summ = pd.read_csv(ROOT / "outputs/08_evaluation/run_summary.csv")
    s = summ[(summ.dataset == "nrcan") & (summ.protocol == "P1") & (summ.block == args.block)
             & (summ.method == args.method) & (summ.seed == seed)].iloc[0]

    # symbolic form of g(x)
    names = d["columns"]
    syms = {n: sp.Symbol(n) for n in names}
    g = sp.sympify(d["equation"], locals={**syms, "aq": aq})
    e, c = syms["engine_size"], syms["cylinders"]

    per_type = []
    data_types = df["transmission_type"].value_counts()
    for t in ["A", "AS", "M", "AM", "AV"]:
        if t not in data_types.index:
            continue
        sub = {syms[n]: 0 for n in names if n.startswith("tt_")}
        if f"tt_{t}" in syms:
            sub[syms[f"tt_{t}"]] = 1
        if "gears_missing" in syms:
            sub[syms["gears_missing"]] = 1 if t in cfg["cleaning"]["cvt_codes_without_gears"] else 0
        gt = sp.simplify(g.subs(sub))
        per_type.append((t, int(data_types[t]), gt))

    print(f"Final model: {args.method}, block {args.block}, seed {seed} ({source})")
    print(f"  validation RMSE {val[seed]:.2f}, test RMSE {test:.2f} g/km, {d['n_nodes']} nodes, "
          f"branch-and-bound verdict: {s.bb_verdict}, smallest slope on the domain grid: {s.domain_min_slope:.2f}")
    print(f"  g(x) = {d['equation']}")
    for k, v in d["scaling"].items():
        print(f"  fuel {k} ({FUEL.get(k, k)}): prediction = {v['a']:.2f} + ({v['b']:.3f}) * g(x)")

    lines = ["% T6. Final model: " + f"{args.method}, P1 block {args.block}, seed {seed}",
             "\\begin{tabular}{llr}", "\\hline",
             "Transmission type & Prediction for regular petrol (g/km) & Rows in data \\\\", "\\hline"]
    out_types = []
    for t, n, gt in per_type:
        a, b = d["scaling"]["X"]["a"], d["scaling"]["X"]["b"]
        pred = sp.simplify(a + b * gt)
        try:   # partial fractions in engine size; works when the formula is rational in engine size
            pred = sp.apart(sp.nsimplify(pred, rational=True), e)
        except Exception:
            pass
        pred = pred.evalf(8)
        pred_r = pred.xreplace({x: sp.Float(round(float(x), 2)) for x in pred.atoms(sp.Float)})
        print(f"  {TRANS[t]:30s} ({n:5d} rows): CO2 = {pred_r}")
        tex = re.sub(r"(?<![\d.])1\.0(?!\d)", "1", sp.latex(pred_r))
        tex = tex.replace("engine_{size}", "d").replace("engine\\_size", "d").replace("cylinders", "c")
        lines.append(f"{TRANS[t]} ({t}) & ${tex}$ & {n} \\\\")
        out_types.append({"type": t, "rows": n, "g": str(gt), "prediction_regular": str(pred_r)})
    lines += ["\\hline", "\\end{tabular}", "% d = engine size (litres), c = number of cylinders", ""]

    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "T6_final_model.tex").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "outputs" / "extra").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "extra" / "final_model.json").write_text(json.dumps({
        "method": args.method, "block": args.block, "seed": seed, "validation_rmse": float(val[seed]),
        "test_rmse": test, "n_nodes": d["n_nodes"], "equation": d["equation"], "scaling": d["scaling"],
        "bb_verdict": s.bb_verdict, "domain_min_slope": float(s.domain_min_slope),
        "by_transmission_type": out_types}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
