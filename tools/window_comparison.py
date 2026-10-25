"""Compare the P1 training window used in the study with a window that uses all earlier years.

P1 (used in the study): for test year t, fit on years t-3 and t-2, choose the final model on year
t-1, test on year t.
Expanding window (comparison only): fit on every year from 2015 to t-2, choose on year t-1, test
on year t. Validation and test rows are identical in both designs, so the only difference is how
many older years the model is trained on.

Run from the repository root, after step 05 (and step 08 if GP methods are compared):
    python tools/window_comparison.py
    python tools/window_comparison.py --gp-methods gp_standard dcsr_v --gp-seeds 0 1 2 --gp-years 2020 2024

The first command trains the five non-GP baselines on all seven test years under the expanding
window (a few minutes in total) and retrains them under the P1 window as a check. The second also
trains the GP methods under the expanding window for the chosen seeds and years (about 1 to 3
minutes per run); their P1-window results are read from outputs/08_evaluation/run_metrics.csv, so
they are not retrained.

Outputs:
    outputs/extra/window_comparison_runs.csv     one row per run
    outputs/extra/window_comparison.csv          mean test RMSE per method and window
    results/tables/T5_window_comparison.tex      the same table in LaTeX
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from co2sr.config import load_config, method_spec  # noqa: E402
from co2sr.data.domain import build_contexts, domain_boxes, domain_sample  # noqa: E402
from co2sr.models import build_model  # noqa: E402
from co2sr.steps.common import block_frames, load_clean  # noqa: E402

BASELINES = ["ols", "loglinear", "gam_mono", "xgb", "xgb_mono"]
LABEL = {"ols": "OLS", "loglinear": "Log-linear OLS", "gam_mono": "Monotone GAM", "xgb": "XGBoost",
         "xgb_mono": "Monotone XGBoost", "gp_standard": "Standard GP", "dcsr_s": "DCSR-S",
         "dcsr_f": "DCSR-F", "dcsr_ia": "DCSR-IA", "dcsr_v": "DCSR-V", "dcsr_dyn": "Dynamic penalty"}


def expanding_block(df, p1_block, first_year):
    """Same validation and test rows as the P1 block; fit rows = every year from first_year to t-2."""
    m = p1_block["meta"]
    fit_years = list(range(first_year, m["val_year"]))
    fit_idx = df.index[df["model_year"].isin(fit_years)]
    val_idx = df.index[df["model_year"] == m["val_year"]]
    test_idx = df.index[df["model_year"] == m["test_year"]]
    seen = set(df.loc[fit_idx.union(val_idx), "spec_key"])
    new_idx = test_idx[~df.loc[test_idx, "spec_key"].isin(seen)]
    ids = lambda ix: df.loc[ix, "row_id"].tolist()  # noqa: E731
    return {"block_id": p1_block["block_id"], "fit": ids(fit_idx), "val": ids(val_idx),
            "tests": {"all": ids(test_idx), "new_config": ids(new_idx)},
            "meta": {**m, "fit_years": fit_years}}


def domain_for(cfg, frames):
    dc = cfg["domain"]
    ctx = build_contexts(frames["full"], dc)
    lo, hi = domain_boxes(ctx, dc["interval_segments_train"], dc["max_interval_contexts_train"], dc["sample_seed"])
    vlo, vhi = domain_boxes(ctx, dc.get("verify_root_segments", 1), len(ctx), dc["sample_seed"])
    return {"sample": domain_sample(ctx, dc["sample_size"], dc["sample_seed"]),
            "boxes_lo": lo, "boxes_hi": hi, "boxes_verify_lo": vlo, "boxes_verify_hi": vhi}


def train(cfg, df, block, method, seed, window):
    spec = method_spec(cfg, method)
    fr = block_frames(df, block)
    dom = domain_for(cfg, fr) if spec["type"] == "gp" and spec["constraint"] != "none" else None
    model = build_model(spec, cfg, seed)
    model.fit(fr["fit"], fr["val"], fr["full"], dom)
    rec = {"window": window, "test_year": block["meta"]["test_year"], "method": method, "seed": seed,
           "n_fit": len(fr["fit"])}
    for name, part in [("val", fr["val"])] + list(fr["tests"].items()):
        y = part["co2"].to_numpy(float)
        rec[f"rmse_{name}"] = float(np.sqrt(np.mean((y - model.predict(part)) ** 2)))
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gp-methods", nargs="*", default=[])
    ap.add_argument("--gp-seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--gp-years", nargs="*", type=int, default=[2020, 2024])
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    df = load_clean(cfg, "nrcan")
    first_year = int(df["model_year"].min())
    p1 = json.loads((ROOT / "outputs/05_splits/nrcan_P1_splits.json").read_text())["blocks"]
    rows = []

    for b in p1:
        eb = expanding_block(df, b, first_year)
        for m in BASELINES:
            for window, blk in [("P1 (2 fit years)", b), ("All earlier years", eb)]:
                rows.append(train(cfg, df, blk, m, 0, window))
                print(json.dumps(rows[-1]), flush=True)

    if args.gp_methods:
        metrics = pd.read_csv(ROOT / "outputs/08_evaluation/run_metrics.csv")
        metrics = metrics[(metrics.dataset == "nrcan") & (metrics.protocol == "P1")]
        for b in p1:
            t = b["meta"]["test_year"]
            if t not in args.gp_years:
                continue
            eb = expanding_block(df, b, first_year)
            for m in args.gp_methods:
                for s in args.gp_seeds:
                    old = metrics[(metrics.block == b["block_id"]) & (metrics.method == m) & (metrics.seed == s)]
                    if len(old):
                        rec = {"window": "P1 (2 fit years)", "test_year": t, "method": m, "seed": s,
                               "n_fit": len(b["fit"])}
                        for name in ["val", "all", "new_config"]:
                            r = old.loc[old["set"] == name, "rmse"]
                            rec[f"rmse_{name}"] = float(r.iloc[0]) if len(r) else np.nan
                        rows.append(rec)
                    else:
                        rows.append(train(cfg, df, b, m, s, "P1 (2 fit years)"))
                    rows.append(train(cfg, df, eb, m, s, "All earlier years"))
                    print(json.dumps(rows[-1]), flush=True)

    runs = pd.DataFrame(rows)
    out = ROOT / "outputs" / "extra"
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "window_comparison_runs.csv", index=False)

    # median over seeds within a test year, then mean over test years
    per_year = runs.groupby(["method", "window", "test_year"])["rmse_all"].median().reset_index()
    summ = per_year.pivot_table(index="method", columns="window", values="rmse_all", aggfunc="mean")
    summ["Test years"] = per_year.groupby("method")["test_year"].nunique() // 1
    summ["Change (%)"] = 100 * (summ["All earlier years"] / summ["P1 (2 fit years)"] - 1)
    wins = per_year.pivot_table(index=["method", "test_year"], columns="window", values="rmse_all")
    summ["Years where P1 window is better"] = (wins["P1 (2 fit years)"] < wins["All earlier years"]).groupby(level=0).sum()
    order = [m for m in BASELINES + args.gp_methods if m in summ.index]
    summ = summ.loc[order].reset_index()
    summ["method"] = summ["method"].map(lambda m: LABEL.get(m, m))
    summ = summ[["method", "Test years", "P1 (2 fit years)", "All earlier years", "Change (%)",
                 "Years where P1 window is better"]].rename(columns={"method": "Method"})
    summ.to_csv(out / "window_comparison.csv", index=False)
    print("\nMean test RMSE (g/km) over test years\n" + summ.to_string(index=False))

    tex = ["% T5. Mean test RMSE (g/km) over P1 test years, P1 window against all earlier years",
           "\\begin{tabular}{lrrrrr}", "\\hline",
           "Method & Test years & P1 window & All earlier years & Change (\\%) & Years P1 window better \\\\",
           "\\hline"]
    for _, r in summ.iterrows():
        tex.append(f"{r['Method']} & {int(r['Test years'])} & {r['P1 (2 fit years)']:.2f} & "
                   f"{r['All earlier years']:.2f} & {r['Change (%)']:+.1f} & {int(r['Years where P1 window is better'])} \\\\")
    tex += ["\\hline", "\\end{tabular}", ""]
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "T5_window_comparison.tex").write_text("\n".join(tex), encoding="utf-8")


if __name__ == "__main__":
    main()
