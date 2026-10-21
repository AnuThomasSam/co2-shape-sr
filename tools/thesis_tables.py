"""Tables the thesis needs that steps 09 and 12 do not produce.

Reads only files written by earlier steps (02, 05, 08, 09, 13) and writes CSV and booktabs LaTeX to
outputs/extra and results/tables/extra:

  checkpoint_P4.csv        Stage 3 checkpoint on the P4 test folds: OLS, Standard GP (median over
                           seeds), GAM, XGBoost and monotone XGBoost per fold, and the share of
                           Standard GP runs that beat OLS.
  new_config_P1.csv        P1 test RMSE on all rows and on new-configuration rows only.
  hypotheses.csv           H1 to H5 of the revised plan, each with the value, the threshold fixed in
                           the plan and a verdict. The verdicts are mechanical; the thesis text must
                           still interpret them.
  interaction_ols.csv      OLS with nine engine-size interaction terms, fitted on fit + validation
                           rows, under P1, P2 and P4 (a diagnostic of how much a linear model with a
                           few product terms gains over plain OLS).

    python tools/thesis_tables.py [--config configs/default.yaml]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from co2sr.config import load_config, outputs_dir, results_dir  # noqa: E402

KEYS = ["dataset", "protocol", "block", "method", "seed"]
GP_CONSTRAINED = ["dcsr_t", "dcsr_s", "dcsr_f", "dcsr_ia", "dcsr_dyn", "dcsr_v"]


def med_by_block(met, ds, prot, sets, methods):
    m = met[(met.dataset == ds) & (met.protocol == prot) & (met.set.isin(sets)) & (met.method.isin(methods))]
    # mean over the test sets of a block (P2 has two tails in one block), then median over seeds
    per_run = m.groupby(KEYS)["rmse"].mean().reset_index()
    return per_run.groupby(["block", "method"])["rmse"].median().unstack()


def checkpoint_p4(met):
    t = med_by_block(met, "nrcan", "P4", ["all"], ["ols", "gp_standard", "gam_mono", "xgb", "xgb_mono"])
    runs = met[(met.dataset == "nrcan") & (met.protocol == "P4") & (met.set == "all")]
    ols = runs[runs.method == "ols"].set_index("block")["rmse"]
    gp = runs[runs.method == "gp_standard"]
    share = gp.assign(beat=gp["rmse"].to_numpy() < ols.reindex(gp["block"]).to_numpy()).groupby("block")["beat"].mean()
    t["gp_runs_beating_ols"] = share
    t.loc["mean"] = t.mean()
    return t.reset_index()


def new_config_p1(met, methods):
    a = med_by_block(met, "nrcan", "P1", ["all"], methods).mean()
    n = med_by_block(met, "nrcan", "P1", ["new_config"], methods).mean()
    t = pd.DataFrame({"rmse_all_rows": a, "rmse_new_configurations": n})
    t["ratio_new_to_all"] = t["rmse_new_configurations"] / t["rmse_all_rows"]
    return t.reset_index().rename(columns={"index": "method"})


def a12_med(pw, prot, a, b):
    q = pw[(pw.dataset == "nrcan") & (pw.protocol == prot) & (pw.method_a == a) & (pw.method_b == b)
           & (~pw.set.isin(["val", "new_config"]))]
    return float(q["A12"].median()) if len(q) else np.nan


def hypotheses(d9, bench):
    proto = pd.read_csv(d9 / "summary_by_protocol.csv")
    proto = proto[proto.dataset == "nrcan"].set_index(["protocol", "method"])
    pw = pd.read_csv(d9 / "pairwise_tests.csv")
    rank = pd.read_csv(d9 / "proxy_vs_temporal_ranking.csv")
    validity = pd.read_csv(d9 / "constraint_validity.csv")
    validity = validity[validity.dataset == "nrcan"].set_index(["protocol", "method"])
    ivb = pd.read_csv(d9 / "interval_vs_branch_and_bound.csv")
    ivb = ivb[ivb.dataset == "nrcan"]
    deg = pd.read_csv(d9 / "degradation_vs_P4.csv")
    deg = deg[deg.dataset == "nrcan"]
    summ = pd.read_csv(d9.parent / "08_evaluation" / "run_summary.csv")
    rows = []

    def add(h, measure, value, threshold, ok):
        ok = None if ok is None else bool(ok)
        rows.append({"hypothesis": h, "measure": measure, "value": value, "threshold": threshold,
                     "verdict": "supported" if ok is True else "not supported" if ok is False else "see text"})

    proxy_file = d9.parent / "04_proxy" / "nrcan_proxy_validity.csv"
    pv = pd.read_csv(proxy_file)
    rho = float(pv.loc[pv.measure.str.startswith("Spearman"), "value"].iloc[0])
    tau = float(rank["kendall_tau"].iloc[0])
    add("H1", "Spearman rho, technology score vs model year", rho, "> 0", rho > 0)
    add("H1", f"Kendall tau, method ranking P3 vs P1 (p = {rank['p_value'].iloc[0]:.3f})", tau, "> 0", tau > 0)

    for m in GP_CONSTRAINED:
        v = a12_med(pw, "P2", m, "gp_standard")
        if not np.isnan(v):
            add("H2a", f"median A12, {m} vs gp_standard, P2", v, ">= 0.64", v >= 0.64)
    gp4 = proto.loc[("P4", "gp_standard"), "rmse_mean_of_medians"]
    for m in GP_CONSTRAINED:
        if ("P4", m) in proto.index:
            inc = proto.loc[("P4", m), "rmse_mean_of_medians"] / gp4 - 1
            add("H2b", f"P4 RMSE increase, {m} vs gp_standard", inc, "< 0.05", inc < 0.05)

    fr = ivb[(ivb.protocol.isin(["P1", "P2"])) & (ivb.method.isin(["gp_standard"] + GP_CONSTRAINED))]
    add("H3a", "max interval false rejection share (P1, P2)", float(fr.share_ia_false_rejection.max()), "> 0",
        fr.share_ia_false_rejection.max() > 0)
    samp = ivb[ivb.method.isin(["dcsr_s", "dcsr_f", "dcsr_dyn"])]
    bfa = np.nan
    if bench is not None:
        b = bench[bench.method.isin(["dcsr_s", "dcsr_f", "dcsr_dyn"])]
        bfa = float(((b.test_violation_share == 0) & (b.bb_verdict == "refuted")).groupby(b.method).mean().max())
    gfa = max(float(samp.share_grid_false_acceptance.max()), 0 if np.isnan(bfa) else bfa)
    add("H3a", "max grid false acceptance share, sampling methods (NRCan and benchmark)", gfa, "> 0", gfa > 0)
    add("H3a", "IA-certified equations refuted by B&B (soundness check)", int(ivb.n_ia_certified_bb_refuted.sum()),
        "= 0", int(ivb.n_ia_certified_bb_refuted.sum()) == 0)

    for prot in ["P1", "P2", "P3", "P4"]:
        if (prot, "dcsr_v") in validity.index:
            v = float(validity.loc[(prot, "dcsr_v"), "share_bb_certified"])
            add("H3b", f"DCSR-V certified share, {prot}", v, "= 1", v == 1.0)
    for m in ["dcsr_s", "dcsr_f", "dcsr_dyn"]:
        if ("P2", m) in validity.index:
            v = float(validity.loc[("P2", m), "share_bb_certified"])
            add("H3b", f"{m} certified share, P2", v, "< 1", v < 1.0)

    for prot in ["P1", "P2"]:
        others = [m for m in ["dcsr_s", "dcsr_f", "dcsr_dyn", "dcsr_ia"] if (prot, m) in proto.index]
        best = min(proto.loc[(prot, m), "rmse_mean_of_medians"] for m in others)
        gap = proto.loc[(prot, "dcsr_v"), "rmse_mean_of_medians"] / best - 1
        a12s = [a12_med(pw, prot, "dcsr_v", m) for m in others]
        ok = gap <= 0.02 or all(0.44 < x < 0.56 for x in a12s if not np.isnan(x))
        add("H3c", f"DCSR-V RMSE relative to best other constrained variant, {prot}", gap, "<= 0.02 or A12 in (0.44, 0.56)", ok)
        v = a12_med(pw, prot, "dcsr_v", "dcsr_ia")
        add("H3c", f"A12 DCSR-V vs DCSR-IA, {prot} (expected improvement)", v, ">= 0.56 (non-negligible)", v >= 0.56)

    vc = pd.read_csv(d9 / "verification_cost.csv")
    vc = vc[vc.dataset == "nrcan"].set_index(["protocol", "method"])
    for prot in ["P1", "P2"]:
        if (prot, "dcsr_v_no_bisection") in vc.index:
            a, b = vc.loc[(prot, "dcsr_v"), "verify_bb_certified"], vc.loc[(prot, "dcsr_v_no_bisection"), "verify_bb_certified"]
            add("H3d", f"certified candidates per run, no bisection vs full, {prot}", b / a - 1, "< 0", b < a)
        if (prot, "dcsr_v_no_feedback") in vc.index:
            a, b = vc.loc[(prot, "dcsr_v"), "verify_bb_calls"], vc.loc[(prot, "dcsr_v_no_feedback"), "verify_bb_calls"]
            add("H3d", f"B&B calls per run, no feedback vs full, {prot}", b / a - 1, "> 0", b > a)
        if (prot, "dcsr_v_undecided_infeasible") in proto.index:
            a = proto.loc[(prot, "dcsr_v"), "rmse_mean_of_medians"]
            b = proto.loc[(prot, "dcsr_v_undecided_infeasible"), "rmse_mean_of_medians"]
            add("H3d", f"RMSE change, undecided infeasible vs default, {prot}", b / a - 1, "> 0", b > a)

    p2 = deg[deg.protocol == "P2"].set_index("method")["degradation_ratio"]
    trees = [m for m in ["xgb", "xgb_mono"] if m in p2.index]
    add("H4", "largest P2/P4 degradation ratio is an unconstrained tree ensemble",
        f"{p2.idxmax()} ({p2.max():.2f})", "xgb", p2.idxmax() == "xgb" if trees else None)
    for prot in ["P1", "P2"]:
        v = proto.loc[(prot, "dcsr_v"), "rmse_mean_of_medians"]
        comp = min(proto.loc[(prot, m), "rmse_mean_of_medians"] for m in ["gam_mono", "xgb_mono"])
        add("H4", f"DCSR-V RMSE relative to best of monotone GAM and monotone XGBoost, {prot}", v / comp - 1,
            "<= 0.02 (competitive)", v / comp - 1 <= 0.02)
    nn = summ[(summ.dataset == "nrcan") & (summ.method == "dcsr_v")]["n_nodes"]
    add("H4", "median tree nodes of DCSR-V equations", float(nn.median()), "< 30", nn.median() < 30)

    if bench is not None:
        cert = bench.assign(c=bench.bb_verdict == "certified").groupby("method")["c"].mean()
        others = [m for m in ["dcsr_s", "dcsr_f", "dcsr_dyn", "dcsr_ia"] if m in cert.index]
        add("H5", "DCSR-V certified share minus best other constrained method", float(cert["dcsr_v"] - cert[others].max()),
            ">= 0", cert["dcsr_v"] >= cert[others].max())
        med = bench.groupby(["problem", "method"])["extrap_nmse"].median().unstack()
        wins = int((med["dcsr_v"] <= med["dcsr_ia"]).sum())
        add("H5", "problems where DCSR-V median extrapolation NMSE <= DCSR-IA", f"{wins} of {len(med)}",
            "majority", wins > len(med) / 2)
    return pd.DataFrame(rows)


def interaction_ols(cfg):
    from sklearn.linear_model import LinearRegression
    od = outputs_dir(cfg)
    df = pd.read_csv(od / "02_clean" / "nrcan_clean.csv").set_index("row_id")
    X = pd.DataFrame(index=df.index)
    X["es"], X["cyl"] = df.engine_size, df.cylinders
    X["gears"], X["cvt"] = df.gears.fillna(0), df.gears_missing
    for t in ["AM", "AS", "AV", "M"]:
        X["tt_" + t] = (df.transmission_type == t).astype(float)
    for f in ["Z", "D", "E"]:
        X["f_" + f] = (df.fuel_type == f).astype(float)
    Xi = X.copy()
    for c in ["f_Z", "f_D", "f_E", "tt_AM", "tt_AS", "tt_AV", "tt_M"]:
        Xi["es_x_" + c] = X.es * X[c]
    Xi["es2"], Xi["es_cyl"] = X.es ** 2, X.es * X.cyl
    y = df.co2.astype(float)
    rows = []
    for prot in ["P1", "P2", "P4"]:
        sp = json.loads((od / "05_splits" / f"nrcan_{prot}_splits.json").read_text())
        for b in sp["blocks"]:
            tr = b["fit"] + b["val"]
            m0 = LinearRegression().fit(X.loc[tr], y.loc[tr])
            m1 = LinearRegression().fit(Xi.loc[tr], y.loc[tr])
            for name, te in b["tests"].items():
                r0 = float(np.sqrt(np.mean((m0.predict(X.loc[te]) - y.loc[te]) ** 2)))
                r1 = float(np.sqrt(np.mean((m1.predict(Xi.loc[te]) - y.loc[te]) ** 2)))
                rows.append({"protocol": prot, "block": b["block_id"], "set": name, "ols": r0, "ols_interactions": r1,
                             "n_parameters": Xi.shape[1] + 1})
    return pd.DataFrame(rows)


def to_tex(df, path, digits=3):
    try:
        s = df.to_latex(index=False, float_format=lambda v: f"{v:.{digits}f}", escape=True)
        s = s.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
        path.write_text(s)
    except Exception as exc:          # LaTeX export needs jinja2 in some pandas versions
        print(f"LaTeX export skipped for {path.name}: {exc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    od = outputs_dir(cfg)
    out, tex = od / "extra", results_dir(cfg) / "tables" / "extra"
    out.mkdir(parents=True, exist_ok=True)
    tex.mkdir(parents=True, exist_ok=True)
    met = pd.read_csv(od / "08_evaluation" / "run_metrics.csv")
    bp = od / "13_benchmark" / "benchmark_runs.csv"
    bench = pd.read_csv(bp) if bp.exists() else None
    main_methods = ["ols", "loglinear", "gam_mono", "xgb", "xgb_mono", "gp_standard"] + GP_CONSTRAINED
    tables = {
        "checkpoint_P4": checkpoint_p4(met),
        "new_config_P1": new_config_p1(met, main_methods),
        "hypotheses": hypotheses(od / "09_statistics", bench),
        "interaction_ols": interaction_ols(cfg),
    }
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False)
        to_tex(t, tex / f"{name}.tex")
        print(f"\n== {name}\n{t.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
