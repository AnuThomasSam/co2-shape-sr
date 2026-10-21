"""Step 12: LaTeX tables (booktabs) generated from the CSV outputs of steps 02-10.

Each table is written as results/tables/<name>.tex (a tabular environment only,
to be wrapped in a table float with its caption in the thesis) and a .csv copy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import enabled_datasets
from ..io import read_csv, read_json, write_csv
from ..plotting import label
from .common import sdir

CODE_DEPS = []
NAME = "12_tables"


def config_part(cfg):
    return {"statistics": cfg["statistics"], "labels": cfg["plotting"]["labels"]}


def inputs(cfg):
    return [p for p in [sdir(cfg, "09") / "summary_by_protocol.csv", sdir(cfg, "08") / "run_summary.csv"]
            if p.exists()]


def _esc(x) -> str:
    s = str(x)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}")]:
        s = s.replace(a, b)
    return s


def _fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return str(x)
    if isinstance(x, (float, np.floating)):
        return f"{x:.{digits}f}"
    return _esc(x)


def to_latex(df: pd.DataFrame, path, align=None, digits=3, raw_cols=()):
    cols = list(df.columns)
    align = align or ("l" + "r" * (len(cols) - 1))
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(_esc(c) for c in cols) + r" \\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(r[c]) if c in raw_cols else _fmt(r[c], digits) for c in cols) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    write_csv(df, path.with_suffix(".csv"))
    return [path, path.with_suffix(".csv")]


def run(cfg, log, **_):
    out = sdir(cfg, "12")
    paths = []
    d2, d4, d9, d10 = sdir(cfg, "02"), sdir(cfg, "04"), sdir(cfg, "09"), sdir(cfg, "10")
    for ds in enabled_datasets(cfg):
        p = d2 / f"{ds}_card_feature_ranges.csv"
        if p.exists():
            t = read_csv(p)[["model_year", "rows", "engine_size_min", "engine_size_max", "cylinders_min",
                             "cylinders_max", "cvt_share", "co2_mean"]]
            t.columns = ["Model year", "Rows", "Engine min (L)", "Engine max (L)", "Cyl. min", "Cyl. max",
                         "CVT share", "Mean CO2 (g/km)"]
            paths += to_latex(t, out / f"tab01_data_by_model_year_{ds}.tex", digits=2)
        p = d2 / f"{ds}_cleaning_log.csv"
        if p.exists():
            t = read_csv(p)
            t = t[["rule", "rows_before", "rows_removed"]]
            t.columns = ["Rule", "Rows before", "Rows removed"]
            paths += to_latex(t, out / f"tab02_cleaning_log_{ds}.tex", align="lrr")
        p = d4 / f"{ds}_proxy_validity.csv"
        if p.exists():
            t = read_csv(p)[["measure", "value"]]
            t["value"] = [(_fmt(float(v)) if _is_num(v) else _esc(v)) for v in t["value"]]
            t.columns = ["Measure", "Value"]
            paths += to_latex(t, out / f"tab03_proxy_validity_{ds}.tex", align="lr", raw_cols=("Value",))

    p = d9 / "summary_by_protocol.csv"
    if p.exists():
        t = read_csv(p)
        cv = read_csv(d9 / "constraint_validity.csv") if (d9 / "constraint_validity.csv").exists() else None
        for (ds, pr), g in t.groupby(["dataset", "protocol"]):
            g = g.sort_values("mean_rank")
            if cv is not None:
                g = g.merge(cv[(cv["dataset"] == ds) & (cv["protocol"] == pr)][["method", "share_runs_domain_valid", "share_runs_using_engine_size"]]
                            .rename(columns={"share_runs_domain_valid": "valid",
                                             "share_runs_using_engine_size": "uses_engine"}),
                            on="method", how="left")
            tab = pd.DataFrame({"Method": [label(cfg, m) for m in g["method"]],
                                "Blocks": g["n_blocks"].astype(int),
                                "RMSE": g["rmse_mean_of_medians"], "MAE": g["mae_mean_of_medians"],
                                "R2": g["r2_mean_of_medians"], "Mean rank": g["mean_rank"],
                                "Domain-valid runs": g["valid"] if "valid" in g else np.nan,
                                "Uses engine size": g["uses_engine"] if "uses_engine" in g else np.nan})
            paths += to_latex(tab, out / f"tab04_results_{ds}_{pr}.tex", digits=3)
    p = d9 / "friedman_summary.csv"
    if p.exists():
        t = read_csv(p)
        if len(t):
            t = t[["group", "dataset", "n_blocks", "k_methods", "friedman_chi2", "p_value", "critical_difference"]]
            t.columns = ["Group", "Dataset", "Blocks", "Methods", "Chi2", "p", "CD"]
            paths += to_latex(t, out / "tab05_friedman.tex", align="llrrrrr")
    p = d9 / "pairwise_tests.csv"
    if p.exists():
        t = read_csv(p)
        if len(t):
            g = t.groupby(["dataset", "protocol", "method_a", "method_b"]).agg(
                tests=("A12", "count"), A12_median=("A12", "median"),
                significant=("significant", "sum")).reset_index()
            g["method_a"] = [label(cfg, m) for m in g["method_a"]]
            g["method_b"] = [label(cfg, m) for m in g["method_b"]]
            g.columns = ["Dataset", "Protocol", "Method A", "Method B", "Blocks", "Median A12",
                         "Significant (Holm)"]
            paths += to_latex(g, out / "tab06_pairwise_a12.tex", align="llllrrr")
    p = sdir(cfg, "08") / "run_summary.csv"
    if p.exists():
        s = read_csv(p)
        met = read_csv(sdir(cfg, "08") / "run_metrics.csv")
        gp = s[s["method"].isin(["gp_standard", "dcsr_t", "dcsr_s", "dcsr_f", "dcsr_ia", "dcsr_v", "dcsr_dyn"])]
        if "equation_simplified" in gp and len(gp):
            keys = ["dataset", "protocol", "block", "method", "seed"]
            val = met[met["set"] == "val"][keys + ["rmse"]].rename(columns={"rmse": "val_rmse"})
            tst = (met[~met["set"].isin(["val", "new_config"])].groupby(keys)["rmse"].mean()
                   .rename("test_rmse").reset_index())
            gp = gp.merge(val, on=keys).merge(tst, on=keys)
            for (ds, pr), g in gp.groupby(["dataset", "protocol"]):
                for block in sorted(g["block"].unique()):
                    rows = []
                    for m, gm in g[g["block"] == block].groupby("method"):
                        # representative run: the seed with the median validation RMSE. Choosing on
                        # validation keeps test data out of the choice; the median avoids reporting
                        # the luckiest seed.
                        gm = gm.sort_values("val_rmse").reset_index(drop=True)
                        rep = gm.iloc[(len(gm) - 1) // 2]
                        rows.append({"Method": _esc(label(cfg, m)),
                                     "Equation g(x)": f"${_latex_eq(rep['equation_simplified'])}$",
                                     "Tree nodes": int(rep["n_nodes"]),
                                     "Val. RMSE": f"{rep['val_rmse']:.2f}", "Test RMSE": f"{rep['test_rmse']:.2f}"})
                    if rows:
                        paths += to_latex(pd.DataFrame(rows), out / f"tab07_equations_{ds}_{pr}_{block}.tex",
                                          align="lp{8cm}rrr",
                                          raw_cols=("Method", "Equation g(x)", "Tree nodes", "Val. RMSE", "Test RMSE"))
    p = d9 / "interval_vs_branch_and_bound.csv"
    if p.exists():
        t = read_csv(p)
        if len(t):
            t = t.assign(method=[label(cfg, m) for m in t["method"]])[
                ["dataset", "protocol", "method", "n_runs", "share_ia_certified", "share_bb_certified",
                 "share_ia_false_rejection", "share_grid_false_acceptance", "bb_boxes_median"]]
            t.columns = ["Dataset", "Protocol", "Method", "Runs", "IA certified", "B&B certified",
                         "IA false rejection", "Grid false acceptance", "B&B boxes (median)"]
            paths += to_latex(t, out / "tab09_interval_vs_branch_and_bound.tex", align="lllrrrrrr",
                              raw_cols=())
    p = d9 / "verification_cost.csv"
    if p.exists():
        t = read_csv(p)
        if len(t):
            keep = [c for c in ["dataset", "protocol", "method", "fit_time_s", "verify_bb_calls", "verify_bb_boxes",
                                "verify_prefilter_refuted", "verify_bb_certified", "verify_bb_refuted",
                                "verify_bb_undecided", "verify_cx_added"] if c in t]
            t = t[keep].assign(method=[label(cfg, m) for m in t["method"]])
            t.columns = [c.replace("verify_", "").replace("_", " ").capitalize() for c in keep]
            paths += to_latex(t, out / "tab10_verification_cost.tex", digits=1)
    p = d10 / "audit_kruskal.json"
    if p.exists():
        k = read_json(p)
        t = read_csv(d10 / "audit_by_make.csv").sort_values("median_residual")
        t = t[["make", "n", "median_residual", "ci_low", "ci_high"]]
        t.columns = ["Make", "n", "Median residual", "CI low", "CI high"]
        paths += to_latex(t, out / "tab08_audit_by_make.tex", digits=2)
        (out / "tab08_audit_kruskal.txt").write_text(
            f"Kruskal-Wallis H = {k['H']:.3f}, p = {k['p_value']:.3g}, groups = {k['n_groups']}, N = {k['N']}\n")
    log.info("%d table file(s) written to %s", len(paths), out)
    return paths


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _latex_eq(s: str) -> str:
    try:
        import sympy as sp
        return sp.latex(sp.sympify(s))
    except Exception:
        return _esc(s)
